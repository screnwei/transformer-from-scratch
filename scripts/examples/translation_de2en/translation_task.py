"""
Now we consider a real-world example using the Multi30k German-English Translation task.
This task is much smaller than the WMT task considered in the paper, but it illustrates the whole system.
We also show how to use multi-gpu processing to make it really fast.

现在我们考虑一个使用 Multi30k 德英翻译任务的现实世界示例。这个任务比论文中考虑的 WMT 任务要小得多，但它说明了整个系统。我们还展示了如何使用多 GPU 处理来使其真正快速。
"""
import os
from os.path import exists
import glob

import GPUtil
import spacy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.functional import pad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator

from scripts.train import LabelSmoothing, rate, TrainState, run_epoch, Batch, SimpleLossCompute, DummyOptimizer, \
    DummyScheduler
from src.models.transformer import make_model


class Multi30kDataset(Dataset):
    def __init__(self, de_file, en_file):
        with open(de_file, 'r', encoding='utf-8') as f:
            self.de_data = f.readlines()
        with open(en_file, 'r', encoding='utf-8') as f:
            self.en_data = f.readlines()

    def __len__(self):
        return len(self.de_data)

    def __getitem__(self, idx):
        # __getitem__ 是 Python 中的一个特殊方法（magic method），它用于实现对象的索引操作。
        # 当使用 dataset[idx] 这样的索引操作时，Python 会自动调用这个方法
        return self.de_data[idx].strip(), self.en_data[idx].strip()

# Load spacy tokenizer models, download them if they haven't been
# downloaded already
def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(texts, tokenizer, index):
    for text in texts:
        # 使用 yield 关键字定义生成器函数
        # 生成器函数不会一次性执行完所有代码，而是每次执行到 yield 时暂停，并返回一个值
        # 下次调用时会从上次暂停的地方继续执行
        yield tokenizer(text)




def load_local_dataset(root_dir):
    # 加载本地数据集
    # 参数:
    #   root_dir: 数据集根目录路径
    # 返回:
    #   train_dataset: 训练数据集
    #   val_dataset: 验证数据集
    #   test_dataset: 测试数据集

    train_de = os.path.join(root_dir, 'train.de')
    train_en = os.path.join(root_dir, 'train.en')
    val_de = os.path.join(root_dir, 'val.de')
    val_en = os.path.join(root_dir, 'val.en')
    test_de = os.path.join(root_dir, 'test.de')
    test_en = os.path.join(root_dir, 'test.en')
    
    train_dataset = Multi30kDataset(train_de, train_en)
    val_dataset = Multi30kDataset(val_de, val_en)
    test_dataset = Multi30kDataset(test_de, test_en)
    
    return train_dataset, val_dataset, test_dataset


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = load_local_dataset('datasets/Multi30k')
    all_de = []
    for dataset in [train, val, test]:
        for item in dataset:
            all_de.append(item[0])
    
    vocab_src = build_vocab_from_iterator(
        yield_tokens(all_de, tokenize_de, index=0),
        # 表示只保留出现次数大于等于2的词
        min_freq=2,
        # 添加特殊标记：
        # <s>: 句子开始标记
        # </s>: 句子结束标记
        # <blank>: 填充标记
        # <unk>: 未知词标记
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    all_en = []
    for dataset in [train, val, test]:
        for item in dataset:
            all_en.append(item[1])
            
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(all_en, tokenize_en, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    # 将未知词的索引设置为默认索引
    # 当遇到不在词表中的词时，会返回 < unk > 的索引
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print(f"Source vocabulary size: {len(vocab_src)}")
    print(f"Target vocabulary size: {len(vocab_tgt)}")
    return vocab_src, vocab_tgt


def collate_batch(
    batch,                # 输入批次数据
    src_pipeline,         # 源语言（德语）处理管道，对应的就是tokenize_de
    tgt_pipeline,         # 目标语言（英语）处理管道，对应的是tokenize_en
    src_vocab,           # 源语言词表
    tgt_vocab,           # 目标语言词表
    device,              # 训练设备
    max_padding=128,     # 最大填充长度
    pad_id=2,            # 填充标记的索引
):
    """
    批处理函数，用于处理每个批次的数据
    
    参数:
        batch: 原始批次数据，包含源语言和目标语言句子对
        src_pipeline: 源语言处理函数（分词等）
        tgt_pipeline: 目标语言处理函数（分词等）
        src_vocab: 源语言词表
        tgt_vocab: 目标语言词表
        device: 训练设备
        max_padding: 序列最大长度
        pad_id: 填充标记的索引
        
    转换过程示例：
    假设有一个批次包含一个句子对：
    源语言（德语）: "Ein Mann läuft."
    目标语言（英语）: "A man is running."
    
    1. 分词：
       src_tokens = ["Ein", "Mann", "läuft", "."]
       tgt_tokens = ["A", "man", "is", "running", "."]
    
    2. 转换为词表索引（假设词表映射）：
       src_indices = [10, 20, 30, 40]  # 每个词对应的索引
       tgt_indices = [50, 60, 70, 80, 90]
    
    3. 添加特殊标记：
       src_with_markers = [0, 10, 20, 30, 40, 1]  # 0是开始标记，1是结束标记
       tgt_with_markers = [0, 50, 60, 70, 80, 90, 1]
    
    4. 填充到固定长度（假设max_padding=10）：
       src_padded = [0, 10, 20, 30, 40, 1, 2, 2, 2, 2]  # 2是填充标记
       tgt_padded = [0, 50, 60, 70, 80, 90, 1, 2, 2, 2]
    
    5. 转换为张量：
       src_tensor = torch.tensor(src_padded)
       tgt_tensor = torch.tensor(tgt_padded)
    
    最终返回形状为 [batch_size, max_padding] 的张量对
    """
    
    # 创建特殊标记的张量
    bs_id = torch.tensor([0], device=device)  # 句子开始标记 <s>
    eos_id = torch.tensor([1], device=device)  # 句子结束标记 </s>
    
    # 初始化源语言和目标语言的列表
    src_list, tgt_list = [], []
    
    # 处理批次中的每个句子对
    for (_src, _tgt) in batch:
        # 处理源语言句子
        processed_src = torch.cat(
            [
                bs_id,  # 添加开始标记
                torch.tensor(
                    src_vocab(src_pipeline(_src)),  # 将分词后的句子转换为词表索引
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,  # 添加结束标记
            ],
            0,  # 在维度0上连接
        )
        
        # 处理目标语言句子
        processed_tgt = torch.cat(
            [
                bs_id,  # 添加开始标记
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),  # 将分词后的句子转换为词表索引
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,  # 添加结束标记
            ],
            0,  # 在维度0上连接
        )
        
        # 对源语言句子进行填充
        src_list.append(
            pad(
                processed_src,
                (0, max_padding - len(processed_src)),  # 填充到最大长度
                value=pad_id,  # 使用填充标记
            )
        )
        
        # 对目标语言句子进行填充
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),  # 填充到最大长度
                value=pad_id,  # 使用填充标记
            )
        )

    # 将列表转换为张量
    src = torch.stack(src_list)  # 形状: [batch_size, max_padding]
    tgt = torch.stack(tgt_list)  # 形状: [batch_size, max_padding]
    
    return (src, tgt)  # 返回处理后的源语言和目标语言张量

def create_dataloaders(
    device,                # 训练设备（GPU/MPS/CPU）
    vocab_src,            # 源语言（德语）词表
    vocab_tgt,            # 目标语言（英语）词表
    spacy_de,             # 德语分词器
    spacy_en,             # 英语分词器
    batch_size=12000,     # 批次大小
    max_padding=128,      # 最大填充长度
    is_distributed=True,  # 是否使用分布式训练
):
    """
    创建训练和验证数据加载器
    
    参数:
        device: 训练设备
        vocab_src: 源语言词表
        vocab_tgt: 目标语言词表
        spacy_de: 德语分词器
        spacy_en: 英语分词器
        batch_size: 每个批次的样本数量
        max_padding: 序列的最大填充长度
        is_distributed: 是否使用分布式训练
    """
    
    # 定义德语和英语的分词函数
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    # 定义批处理函数，用于处理每个批次的数据
    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],  # 获取填充标记的索引
        )

    # 加载本地数据集
    train_iter, valid_iter, test_iter = load_local_dataset('datasets/Multi30k')

    # 将数据集转换为可映射的格式
    train_iter_map = to_map_style_dataset(train_iter)
    valid_iter_map = to_map_style_dataset(valid_iter)

    # 如果使用分布式训练，创建分布式采样器
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    # 创建训练数据加载器
    train_dataloader = DataLoader(
        train_iter_map,                    # 训练数据集
        batch_size=batch_size,             # 批次大小
        shuffle=(train_sampler is None),   # 是否打乱数据（非分布式时打乱）
        sampler=train_sampler,             # 采样器（分布式训练时使用）
        collate_fn=collate_fn,             # 批处理函数
    )

    # 创建验证数据加载器
    valid_dataloader = DataLoader(
        valid_iter_map,                    # 验证数据集
        batch_size=batch_size,             # 批次大小
        shuffle=(valid_sampler is None),   # 是否打乱数据（非分布式时打乱）
        sampler=valid_sampler,             # 采样器（分布式训练时使用）
        collate_fn=collate_fn,             # 批处理函数
    )

    return train_dataloader, valid_dataloader


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Training the System
def train_worker(
    gpu,                    # GPU 设备编号
    ngpus_per_node,         # 每个节点的 GPU 数量
    vocab_src,              # 源语言（德语）词表
    vocab_tgt,              # 目标语言（英语）词表
    spacy_de,               # 德语分词器
    spacy_en,               # 英语分词器
    config,                 # 训练配置参数
    is_distributed=False,   # 是否使用分布式训练
):
    # 获取训练设备（GPU/MPS/CPU）
    device = get_device()
    print(f"Train worker process using device: {device} for training", flush=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 获取填充标记的索引
    pad_idx = vocab_tgt["<blank>"]
    d_model = 512  # Transformer 模型的维度

    # 创建 Transformer 模型
    model = make_model(len(vocab_src), len(vocab_tgt))
    model.to(device)
    module = model
    is_main_process = True
    
    # 如果使用分布式训练且支持 CUDA
    if is_distributed and torch.cuda.is_available():
        # 初始化分布式训练环境
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        # 使用 DistributedDataParallel 包装模型
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0  # 只有主进程（gpu=0）保存模型

    # 创建标签平滑损失函数
    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.to(device)

    # 创建数据加载器
    train_dataloader, valid_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // (ngpus_per_node if torch.cuda.is_available() else 1),
        max_padding=config["max_padding"],
        is_distributed=is_distributed and torch.cuda.is_available(),
    )

    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["base_lr"], 
        betas=(0.9, 0.98), 
        eps=1e-9
    )

    # 创建学习率调度器
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )

    # 创建训练状态对象
    train_state = TrainState()

    # 开始训练循环
    for epoch in range(config["num_epochs"]):
        # 如果是分布式训练，设置数据加载器的 epoch
        if is_distributed and torch.cuda.is_available():
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        # 训练阶段
        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)


        # 查看训练数据加载器中的第一批数据，（无实际用处，只是为了了解 dataloader中的元素）
        for i, batch in enumerate(train_dataloader):
            if i == 0:  # 只查看第一批数据
                print(f"批次数据形状: 源语言: {batch[0].shape}, 目标语言: {batch[1].shape} | 示例数据: 源语言: {batch[0][0]}, 目标语言: {batch[1][0]}")
                break

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

        # 显示 GPU 使用情况
        GPUtil.showUtilization()
        
        # 如果是主进程，保存模型
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()  # 清理 GPU 缓存

        # 验证阶段
        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
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
        torch.cuda.empty_cache()  # 清理 GPU 缓存

    # 训练结束，如果是主进程，保存最终模型
    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    """
    设置并启动分布式训练
    
    参数:
        vocab_src: 源语言（德语）词表
        vocab_tgt: 目标语言（英语）词表
        spacy_de: 德语分词器
        spacy_en: 英语分词器
        config: 训练配置参数
    """
    
    # 获取可用的 GPU 数量
    ngpus = torch.cuda.device_count()
    
    # 设置分布式训练的主节点地址和端口
    # MASTER_ADDR: 主节点的 IP 地址或主机名
    # MASTER_PORT: 主节点用于通信的端口号
    os.environ["MASTER_ADDR"] = "localhost"  # 使用本地主机作为主节点
    os.environ["MASTER_PORT"] = "12356"      # 使用 12356 端口进行通信
    
    # 打印检测到的 GPU 数量
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    
    # 使用 torch.multiprocessing.spawn 启动多个训练进程
    # 每个 GPU 对应一个训练进程
    mp.spawn(
        train_worker,                    # 要执行的训练函数
        nprocs=ngpus,                    # 启动的进程数量（等于 GPU 数量）
        args=(                           # 传递给 train_worker 的参数
            ngpus,                       # 每个节点的 GPU 数量
            vocab_src,                   # 源语言词表
            vocab_tgt,                   # 目标语言词表
            spacy_de,                    # 德语分词器
            spacy_en,                    # 英语分词器
            config,                      # 训练配置
            True                         # 启用分布式训练
        ),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


# def load_trained_model():
#     model = make_model(len(vocab_src), len(vocab_tgt), N=6)
#     model.load_state_dict(torch.load("multi30k_model_final.pt", map_location=device))
#     model.to(device)
#     return model


if __name__ == '__main__':
    # 加载德语和英语的 tokenizer 模型
    spacy_de, spacy_en = load_tokenizers()
    # 构建德语和英语的词汇表
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    
    # 获取设备
    device = get_device()
    # 配置训练参数
    config = {
        # 批次大小：根据设备类型设置不同的批次大小
        # - 对于 MPS 设备（Apple Silicon）使用较小的批次大小 16
        # - 对于其他设备（如 CUDA）使用较大的批次大小 32
        "batch_size": 16 if device.type == 'mps' else 32,

        # 是否使用分布式训练
        # - False: 单机训练
        # - True: 多机多卡分布式训练
        "distributed": False,

        # 训练轮数：整个数据集将被遍历 8 次
        "num_epochs": 8,

        # 梯度累积步数：每 10 个批次更新一次模型参数
        # 用于模拟更大的批次大小，特别是在显存有限的情况下
        "accum_iter": 10,

        # 基础学习率：优化器的初始学习率
        # 使用较大的学习率 1.0，配合 warmup 策略
        "base_lr": 1.0,

        # 最大填充长度：序列的最大长度
        # 超过此长度的序列将被截断，不足的将被填充
        "max_padding": 72,

        # 预热步数：学习率预热阶段的步数
        # 在前 3000 步中，学习率从 0 逐渐增加到 base_lr
        "warmup": 3000,

        # 模型保存文件的前缀
        # 例如：multi30k_model_0.pt, multi30k_model_1.pt 等
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    # if not exists(model_path):
    train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)