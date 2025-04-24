import torch

from scripts.examples.translation_de2en.translation_task import (
    load_tokenizers,
    get_device,
)
from src.models.transformer import make_model, greedy_decode


def load_vocab(vocab_path):
    vocab_src, vocab_tgt = torch.load(vocab_path)
    print(f"Source vocabulary size: {len(vocab_src)}")
    print(f"Target vocabulary size: {len(vocab_tgt)}")
    return vocab_src, vocab_tgt

def load_model(model_path, vocab_src, vocab_tgt):
    """加载训练好的模型"""
    device = get_device()
    model = make_model(len(vocab_src), len(vocab_tgt))
    # 从指定路径加载模型参数到指定设备
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 将模型移动到指定设备(CPU/GPU)
    model.to(device)
    # 设置模型为评估模式，关闭dropout等训练时使用的功能
    model.eval()
    return model

def translate(model, src_sentence, vocab_src, vocab_tgt, spacy_de, spacy_en):
    """
    参数:
        model: 训练好的Transformer模型
        src_sentence: 待翻译的源语言句子
        vocab_src: 源语言词表
        vocab_tgt: 目标语言词表
        spacy_de: 德语分词器
        spacy_en: 英语分词器
        
    返回:
        translation: 翻译后的目标语言句子
    """
    """翻译单个句子"""
    device = get_device()
    # model.eval()

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
    vocab_src, vocab_tgt = load_vocab("scripts/examples/translation_de2en/vocab.pt")
    model = load_model("scripts/examples/translation_de2en/multi30k_model_final.pt", vocab_src, vocab_tgt)

    # 测试句子
    test_sentences = [
        "Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.",
        "Eine Gruppe von Menschen steht vor einem Iglu .",
        "Ein Mann lächelt einen ausgestopften Löwen an .",
        "Drei Mädchen, von denen eines gerade trinkt, stehen in einer belebten Straße und schneiden Grimassen.",
        "Eine junge Frau und eine ältere Frau in traditionellen Saris spinnen Textilien, während drei weitere Personen in moderner Kleidung nur von der Taille abwärts auf dem Bild zu sehen sind."
    ]

    print("开始翻译测试...")
    for src in test_sentences:
        print(f"\n源句子: {src}")
        translation = translate(model, src, vocab_src, vocab_tgt, spacy_de, spacy_en)
        print(f"翻译结果: {translation}")


if __name__ == "__main__":
    main()
