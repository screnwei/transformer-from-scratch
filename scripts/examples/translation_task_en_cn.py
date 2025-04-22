"""
Now we consider a real-world example using the Opus Books English-Chinese Translation task.
This task is much smaller than the WMT task considered in the paper, but it illustrates the whole system.
We also show how to use hardware acceleration to make it really fast.

现在我们考虑一个使用 Opus Books 英中翻译任务的现实世界示例。这个任务比论文中考虑的 WMT 任务要小得多，但它说明了整个系统。我们还展示了如何使用硬件加速来使其真正快速。

执行脚本：PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=$PYTHONPATH:. python scripts/examples/translation_task_modified.py
"""
import os
from os.path import exists

import GPUtil
import spacy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.functional import pad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
import jieba

from scripts.train import LabelSmoothing, rate, TrainState, run_epoch, Batch, SimpleLossCompute, DummyOptimizer, \
    DummyScheduler
from src.models.transformer import make_model

# 检查是否可以使用 MPS
def get_device():
    if torch.backends.mps.is_available():
        # return torch.device("mps")
        return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_tokenizers():
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    # 中文不需要加载 spacy 模型，我们使用 jieba
    return spacy_en


def tokenize_english(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def tokenize_chinese(text):
    return list(jieba.cut(text))


class OpusBooksDataset(Dataset):
    def __init__(self, split, tokenizer_en):
        full_dataset = load_dataset("opus100", "en-zh", split="train")
        # 将数据集分割成训练集(80%)和验证集(20%)
        if split == "train":
            self.dataset = full_dataset.train_test_split(test_size=0.2, seed=42)["train"]
        elif split == "validation":
            self.dataset = full_dataset.train_test_split(test_size=0.2, seed=42)["test"]
        else:
            raise ValueError(f"Unknown split {split}")
            
        self.tokenizer_en = tokenizer_en

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item['translation']['en'], item['translation']['zh']


def yield_tokens(dataset, tokenizer, index):
    for item in dataset:
        if index == 0:  # English
            yield tokenizer(item[index])
        else:  # Chinese
            yield tokenize_chinese(item[index])


def build_vocabulary(spacy_en):
    def tokenize_en_wrapper(text):
        return tokenize_english(text, spacy_en)

    print("Building English Vocabulary ...")
    train_dataset = OpusBooksDataset("train", tokenize_en_wrapper)
    val_dataset = OpusBooksDataset("validation", tokenize_en_wrapper)
    
    all_data = [(item[0], item[1]) for item in train_dataset] + \
             [(item[0], item[1]) for item in val_dataset]
    
    vocab_src = build_vocab_from_iterator(
        yield_tokens(all_data, tokenize_en_wrapper, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building Chinese Vocabulary ...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(all_data, None, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_en):
    if not exists("vocab_en_zh.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab_en_zh.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab_en_zh.pt")
    print("Finished.\nVocabulary sizes:")
    print(f"English vocabulary size: {len(vocab_src)}")
    print(f"Chinese vocabulary size: {len(vocab_tgt)}")
    return vocab_src, vocab_tgt


spacy_en = load_tokenizers()
vocab_src, vocab_tgt = load_vocab(spacy_en)

def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tokenize_chinese(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            pad(
                processed_src,
                (0, max_padding - len(processed_src)),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    def tokenize_en_wrapper(text):
        return tokenize_english(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_en_wrapper,
            None,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_dataset = OpusBooksDataset("train", tokenize_en_wrapper)
    valid_dataset = OpusBooksDataset("validation", tokenize_en_wrapper)

    train_sampler = (
        DistributedSampler(train_dataset) if is_distributed else None
    )
    valid_sampler = (
        DistributedSampler(valid_dataset) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_en,
    config,
    is_distributed=False,
):
    device = get_device()
    print(f"Train worker process using device: {device} for training", flush=True)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.to(device)
    module = model
    is_main_process = True
    
    # 只有在使用 CUDA 时才使用分布式训练
    if is_distributed and torch.cuda.is_available():
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_en,
        batch_size=config["batch_size"] // (ngpus_per_node if torch.cuda.is_available() else 1),
        max_padding=config["max_padding"],
        is_distributed=is_distributed and torch.cuda.is_available(),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed and torch.cuda.is_available():
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[Device {device}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)

        print(f"[Device {device}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_en, config):
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) for Apple Silicon")
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_en, config, False)
        return
    elif not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU only.")
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_en, config, False)
        return

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_en, config):
    if config["distributed"] and not torch.backends.mps.is_available():
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_en, config, False
        )


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "en_zh_model_",
    }
    model_path = "en_zh_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_en, config)

    device = get_device()
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("en_zh_model_final.pt", map_location=device))
    model.to(device)
    return model


if __name__ == "__main__":
    model = load_trained_model() 