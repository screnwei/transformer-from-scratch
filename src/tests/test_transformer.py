import torch
import torch.nn as nn
from src.models.transformer import (
    make_model,
    subsequent_mask,
    EncoderDecoder,
    Generator,
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    Embeddings,
    PositionalEncoding,
    attention,
)

def test_model_build():
    "Test model building"
    model = make_model(11, 11, 2)
    assert isinstance(model, EncoderDecoder)
    assert isinstance(model.encoder, Encoder)
    assert isinstance(model.decoder, Decoder)
    assert isinstance(model.generator, Generator)

def test_attention():
    "Test attention mechanism"
    query = torch.randn(1, 1, 512)
    key = torch.randn(1, 1, 512)
    value = torch.randn(1, 1, 512)
    
    output, attn = attention(query, key, value)
    assert output.shape == (1, 1, 512)
    assert attn.shape == (1, 1, 1)

def test_multi_headed_attention():
    "Test multi-headed attention"
    mha = MultiHeadedAttention(8, 512)
    query = torch.randn(1, 1, 512)
    key = torch.randn(1, 1, 512)
    value = torch.randn(1, 1, 512)
    
    output = mha(query, key, value)
    assert output.shape == (1, 1, 512)

def test_positional_encoding():
    "Test positional encoding"
    pe = PositionalEncoding(512, 0.1)
    x = torch.randn(1, 10, 512)
    output = pe(x)
    assert output.shape == (1, 10, 512)

def inference_test():
    "Test the model with a simple example"
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)

def run_tests():
    "Run all tests"
    print("Running model build test...")
    test_model_build()
    print("Running attention test...")
    test_attention()
    print("Running multi-headed attention test...")
    test_multi_headed_attention()
    print("Running positional encoding test...")
    test_positional_encoding()
    print("Running inference test...")
    for _ in range(10):
        inference_test()

if __name__ == "__main__":
    run_tests() 