import torch
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
import spacy
from scripts.examples.translation_task import (
    load_tokenizers,
    load_vocab,
    create_dataloaders,
    get_device,
    make_model,
    Batch
)

def load_model(model_path, vocab_src, vocab_tgt):
    """加载训练好的模型"""
    device = get_device()
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """使用贪心算法进行解码"""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(
            memory, src_mask, 
            ys, 
            subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def subsequent_mask(size):
    """创建掩码矩阵"""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def translate(model, src_sentence, vocab_src, vocab_tgt, spacy_de, spacy_en):
    """翻译单个句子"""
    device = get_device()
    model.eval()
    
    # 对输入句子进行分词
    src_tokens = [tok.text for tok in spacy_de.tokenizer(src_sentence)]
    src_tokens = ['<s>'] + src_tokens + ['</s>']
    src_indexes = [vocab_src[token] for token in src_tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = (src_tensor != vocab_src['<blank>']).unsqueeze(-2)
    
    # 进行解码
    out = greedy_decode(
        model, src_tensor, src_mask, 
        max_len=60, start_symbol=vocab_tgt['<s>']
    )
    
    # 将输出转换为文本
    tgt_tokens = [vocab_tgt.get_itos()[i] for i in out[0]]
    # 移除所有的特殊标记并清理多余的空格
    cleaned_tokens = [token for token in tgt_tokens[1:] if token not in ['<s>', '</s>', '<blank>']]
    translation = ' '.join(cleaned_tokens).strip()
    # 清理重复的标点和空格
    while " ." in translation:
        translation = translation.replace(" .", ".")
    while ".." in translation:
        translation = translation.replace("..", ".")
    return translation

def main():
    # 加载必要的组件
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    model = load_model("scripts/examples/multi30k_model_final.pt", vocab_src, vocab_tgt)
    
    # 测试句子
    test_sentences = [
        "Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.",
        "Eine Gruppe von Menschen steht vor einem Iglu .",
        "Ein Mann lächelt einen ausgestopften Löwen an ."
    ]
    
    print("开始翻译测试...")
    for src in test_sentences:
        print(f"\n源句子: {src}")
        translation = translate(model, src, vocab_src, vocab_tgt, spacy_de, spacy_en)
        print(f"翻译结果: {translation}")

if __name__ == "__main__":
    main()