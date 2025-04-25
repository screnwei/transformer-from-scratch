import copy
import math

import torch
from torch import nn
from torch.nn.functional import log_softmax, pad

class Embeddings(nn.Module):
    """
    词嵌入模块，用于将离散的词索引转换为连续的向量表示。
    
    该模块实现了标准的词嵌入层，并按照 Transformer 论文的要求对嵌入向量进行了缩放。
    缩放因子为 sqrt(d_model)，这有助于保持嵌入向量的方差在合理范围内。
    eg：如果一个sequence有10个词，d_model为512的时候，则我们得到的是一个10*512的矩阵。每一行是512列，代表一个词的dense表示
    
    主要功能：
    1. 将词索引映射到高维向量空间
    2. 对嵌入向量进行缩放，以保持数值稳定性
    """

    def __init__(self, d_model, vocab):
        """
        初始化词嵌入层
        
        参数:
            d_model (int): 嵌入向量的维度，即模型维度
            vocab (int): 词表大小，即需要嵌入的词汇数量
        """
        super(Embeddings, self).__init__()
        # 使用 PyTorch 的 Embedding 层进行词嵌入
        # one-hot转词嵌入，这里有一个待训练的矩阵E，大小是vocab*d_model
        self.lut = nn.Embedding(vocab, d_model)
        # 保存模型维度，用于后续的缩放操作
        self.d_model = d_model

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入词索引，形状为 [batch_size, seq_len]
            
        返回:
            torch.Tensor: 嵌入后的向量，形状为 [batch_size, seq_len, d_model]
        """
        # 1. 通过查找表获取词嵌入
        # 2. 乘以 sqrt(d_model) 进行缩放
        return self.lut(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于为序列中的每个位置添加位置信息。
    
    在 Transformer 中，由于没有循环或卷积结构，模型无法感知序列中词的位置信息。
    因此需要显式地添加位置编码，使模型能够利用序列的顺序信息。
    
    位置编码使用正弦和余弦函数的组合，生成固定维度的位置向量：
    1. 对于偶数位置使用正弦函数
    2. 对于奇数位置使用余弦函数
    
    这种编码方式具有以下优点：
    1. 可以表示任意长度的序列
    2. 不同位置之间的相对位置关系可以通过线性变换表示
    3. 可以很好地泛化到比训练时更长的序列
    """

    def __init__(self, d_model, dropout, max_len=5000):
        """
        初始化位置编码模块

        注意，位置编码不会更新，是写死的，所以这个class里面没有可训练的参数
        
        参数:
            d_model (int): 模型维度，即位置编码的维度， d_model=512
            dropout (float): dropout比率，用于防止过拟合， dropout=0.1,
            max_len (int): 支持的最大序列长度，默认为5000，max_len=5000代表事先准备好长度为5000的序列的位置编码，其实没必要，
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预先计算位置编码矩阵，形状: [max_len, d_model]
        # (5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，每个位置用一个512维度向量来表示其位置编码
        pe = torch.zeros(max_len, d_model)
        
        # 生成位置序列 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # 计算除数项，用于生成不同频率的正弦和余弦波
        # (0,2,…, 4998)一共准备2500个值，供sin, cos调用
        # 此处的PE计算公式写的比较牛B，详见图片 PE.png 中的推导逻辑
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        
        # 对偶数位置使用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对奇数位置使用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度，形状变为: [1, max_len, d_model]
        # (5000, 512) -> (1, 5000, 512) 为batch.size留出位置
        pe = pe.unsqueeze(0)
        
        # 将位置编码矩阵注册为缓冲区，这样它不会被当作模型参数
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]
            
        返回:
            torch.Tensor: 添加了位置编码的张量，形状与输入相同
        """
        # 将位置编码添加到输入张量中
        # pe[:, :x.size(1)] 选择与输入序列长度匹配的位置编码
        # requires_grad_(False) 确保位置编码不参与梯度计算

        # 接受1.Embeddings的词嵌入结果x，
        # 然后把自己的位置编码pe，加上去。
        # 例如，假设x是(30,10,512)的一个tensor， 30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
        # 则该行代码的第二项是(1, min(10, 5000), 512)=(1,10,512)，
        # 在具体相加的时候，会扩展(1,10,512)为(30,10,512)，
        # 保证一个batch中的30个序列，都使用（叠加）一样的位置编码。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        
        # 应用 dropout，防止过拟合， 在训练过程中，随机将一部分位置编码的值置为 0
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制模块，实现 Transformer 中的多头注意力。

    该模块将输入分成多个头，每个头独立计算注意力，然后将结果拼接起来。
    这种设计允许模型同时关注不同位置的不同表示子空间。

    主要功能：
    1. 将输入分成多个头，每个头独立计算注意力
    2. 对每个头的结果进行线性变换
    3. 将所有头的结果拼接起来
    4. 应用最终的线性变换

    参数说明：
    - h: 注意力头的数量
    - d_model: 输入和输出的维度
    - dropout: dropout 比率，用于防止过拟合

    实现细节：
    1. 使用 4 个线性层分别处理查询、键、值和最终输出
    2. 每个头的维度为 d_model // h
    3. 使用 attention 函数计算每个头的注意力
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        初始化多头注意力模块

        参数:
            h (int): 注意力头的数量，通常为 8
            d_model (int): 输入和输出的维度，通常为 512
            dropout (float): dropout 比率，默认为 0.1

        初始化过程：
        1. 确保 d_model 能被 h 整除
        2. 创建 4 个线性层用于查询、键、值和最终输出
        3. 创建 dropout 层

        eg: h=8, d_model=512
        """
        super(MultiHeadedAttention, self).__init__()
        # 确保 d_model 能被 h 整除
        assert d_model % h == 0
        # 每个头的维度
        self.d_k = d_model // h
        # 注意力头的数量
        self.h = h
        # 创建 4 个线性层，用于查询、键、值和最终输出
        # eg: 每个线性层的大小是(512, 512)的，每个Linear network里面有两类可训练参数，Weights，其大小为512*512，以及biases，其大小为512=d_model。
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 用于存储注意力权重，主要用于可视化
        self.attn = None
        # dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播函数

        参数:
            query (torch.Tensor): 查询向量，形状为 [batch_size, seq_len, d_model]
            key (torch.Tensor): 键向量，形状为 [batch_size, seq_len, d_model]
            value (torch.Tensor): 值向量，形状为 [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): 注意力掩码，形状为 [batch_size, 1, seq_len, seq_len]

        返回:
            torch.Tensor: 多头注意力的输出，形状为 [batch_size, seq_len, d_model]

        处理流程:
        1. 对查询、键、值进行线性变换
        2. 将变换后的张量分成多个头
        3. 计算每个头的注意力
        4. 将所有头的结果拼接起来
        5. 应用最终的线性变换
        """

        # eg: 输入query的形状类似于(30, 10, 512)，key.size() ~ (30, 11, 512), 以及value.size() ~ (30, 11, 512)
        if mask is not None:
            # 将掩码扩展到所有头
            mask = mask.unsqueeze(1)
        # 获取批次大小
        nbatches = query.size(0)

        # 1. 对查询、键、值进行线性变换
        # 2. 将变换后的张量分成多个头
        # 3. 调整维度顺序以进行注意力计算
        #  eg:这里是前三个Linear Networks的具体应用，
        #     例如query=(30,10, 512) -> Linear network -> (30, 10, 512) -> view -> (30,10, 8, 64) -> transpose(1,2) -> (30, 8, 10, 64)
        #     其他的key和value也是类似地，从(30, 11, 512) -> (30, 8, 11, 64)。
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 4. 计算每个头的注意力
        # #调用上面定义好的attention函数，输出的x形状为(30, 8, 10, 64)。attn的形状为(30, 8, 10=target.seq.len, 11=src.seq.len)
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 5. 将所有头的结果拼接起来
        # 6. 应用最终的线性变换
        # eg：x ~ (30, 8, 10, 64) -> transpose(1,2) -> (30, 10, 8, 64) -> contiguous() and view -> (30, 10, 8*64) = (30, 10, 512)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        # 清理中间变量以节省内存
        del query
        del key
        del value

        # eg：执行第四个Linear network，把(30, 10, 512)经过一次linear network，得到(30, 10, 512).
        return self.linears[-1](x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络模块，实现 Transformer 中的前馈神经网络部分。

    完全相同的前馈网络独立应用于一个输入句子的每个位置上的单词，所以又称为point-wise feed-forward neural network (这里的Point=word)。

    该模块实现了 Transformer 论文中的前馈神经网络（FFN）方程：
    FFN(x) = max(0, xW1 + b1)W2 + b2

    主要功能：
    1. 对每个位置的特征向量进行独立的非线性变换
    2. 通过两层线性变换和 ReLU 激活函数增加模型的表达能力
    3. 在两层线性变换之间使用 dropout 防止过拟合

    参数说明：
    - d_model: 输入和输出的维度（模型维度）
    - d_ff: 中间层的维度，通常比 d_model 大 4 倍
    - dropout: dropout 比率，用于防止过拟合

    实现细节：
    1. 第一层线性变换将输入从 d_model 维度扩展到 d_ff 维度
    2. 使用 ReLU 激活函数进行非线性变换
    3. 应用 dropout 随机丢弃部分激活值
    4. 第二层线性变换将特征从 d_ff 维度压缩回 d_model 维度
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化位置前馈网络

        参数:
            d_model (int): 输入和输出的维度，通常为 512
            d_ff (int): 中间层的维度，通常为 2048（是 d_model 的 4 倍）
            dropout (float): dropout 比率，默认为 0.1，用于防止过拟合

        初始化过程：
        1. 创建第一层线性变换：d_model -> d_ff
        2. 创建第二层线性变换：d_ff -> d_model
        3. 创建 dropout 层
        """
        super(PositionwiseFeedForward, self).__init__()
        # 第一层线性变换：将输入从 d_model 维度扩展到 d_ff 维度
        self.w_1 = nn.Linear(d_model, d_ff)
        # 第二层线性变换：将特征从 d_ff 维度压缩回 d_model 维度
        self.w_2 = nn.Linear(d_ff, d_model)
        # dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]
                其中：
                - batch_size: 批次大小
                - seq_len: 序列长度
                - d_model: 特征维度

        返回:
            torch.Tensor: 经过前馈网络处理后的张量，形状与输入相同 [batch_size, seq_len, d_model]

        处理流程：
        1. 第一层线性变换：xW1 + b1
        2. ReLU 激活函数：max(0, xW1 + b1)
        3. Dropout：随机丢弃部分激活值
        4. 第二层线性变换：(xW1 + b1)W2 + b2
        """
        # 1. 第一层线性变换：xW1 + b1
        # 2. ReLU 激活函数：max(0, xW1 + b1)
        # 3. Dropout：随机丢弃部分激活值
        # 4. 第二层线性变换：(xW1 + b1)W2 + b2
        return self.w_2(self.dropout(self.w_1(x).relu()))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Generator(nn.Module):
    """
    生成器模块，用于将解码器的输出转换为最终的预测结果。

    该模块实现了标准的线性变换 + softmax 生成步骤：
    1. 首先通过线性层将解码器输出的高维向量投影到词表大小
    2. 然后应用 log_softmax 函数得到每个词的概率分布

    这是 Transformer 模型的最后一步，将解码器的输出转换为实际的预测结果。
    """

    def __init__(self, d_model, vocab):
        """
        初始化生成器

        参数:
            d_model (int): 模型维度，即解码器输出的维度
            vocab (int): 词表大小，即预测的类别数
        """
        super(Generator, self).__init__()
        # 线性投影层，将 d_model 维向量映射到词表大小
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 解码器的输出，形状为 [batch_size, seq_len, d_model]

        返回:
            torch.Tensor: 经过 log_softmax 后的预测结果，形状为 [batch_size, seq_len, vocab]
        """
        return log_softmax(self.proj(x), dim=-1)

class EncoderDecoder(nn.Module):
    """
    Transformer的标准编码器-解码器架构。
    这是Transformer模型的基础架构，也是许多其他序列到序列模型的基础。

    主要组件:
    - encoder: 编码器，用于处理输入序列
    - decoder: 解码器，用于生成输出序列
    - src_embed: 源语言（输入）的词嵌入层
    - tgt_embed: 目标语言（输出）的词嵌入层
    - generator: 生成器，用于将解码器输出转换为最终预测
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        初始化编码器-解码器模型

        参数:
            encoder: 编码器模块
            decoder: 解码器模块
            src_embed: 源语言词嵌入层
            tgt_embed: 目标语言词嵌入层
            generator: 生成器模块
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        前向传播函数

        参数:
            src: 源序列输入
            tgt: 目标序列输入
            src_mask: 源序列的掩码
            tgt_mask: 目标序列的掩码

        返回:
            解码器的输出
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        编码过程：将源序列转换为编码表示

        参数:
            src: 源序列输入
            src_mask: 源序列的掩码

        返回:
            编码后的源序列表示
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        解码过程：基于编码表示生成目标序列

        参数:
            memory: 编码器的输出（记忆）
            src_mask: 源序列的掩码
            tgt: 目标序列输入
            tgt_mask: 目标序列的掩码

        返回:
            解码器的输出
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    """
    构建Transformer模型的主要函数
    
    参数:
        src_vocab (int): 源语言词表大小
        tgt_vocab (int): 目标语言词表大小
        N (int): 编码器和解码器的层数，默认为6
        d_model (int): 模型维度，默认为512
        d_ff (int): 前馈网络中间层维度，默认为2048
        h (int): 注意力头数，默认为8
        dropout (float): dropout比率，默认为0.1
        
    返回:
        model (EncoderDecoder): 构建好的Transformer模型
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Print the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params:,}")
    return model

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    """
    实现 Transformer 中的缩放点积注意力机制（Scaled Dot-Product Attention）。

    该方法计算注意力权重并生成加权和输出，是 Transformer 的核心组件之一。

    数学公式：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    参数:
        query (torch.Tensor): 查询向量，形状为 [batch_size, num_heads, seq_len, d_k]
        key (torch.Tensor): 键向量，形状为 [batch_size, num_heads, seq_len, d_k]
        value (torch.Tensor): 值向量，形状为 [batch_size, num_heads, seq_len, d_v]
        mask (torch.Tensor, optional): 注意力掩码，用于屏蔽某些位置，形状为 [batch_size, 1, seq_len, seq_len]
        dropout (nn.Dropout, optional): dropout 层，用于防止过拟合

    返回:
        tuple: (output, attention_weights)
            - output: 注意力输出，形状为 [batch_size, num_heads, seq_len, d_v]
            - attention_weights: 注意力权重，形状为 [batch_size, num_heads, seq_len, seq_len]

    处理流程:
    1. 计算查询和键的点积，并除以 sqrt(d_k) 进行缩放
    2. 应用掩码（如果提供）
    3. 使用 softmax 计算注意力权重
    4. 应用 dropout（如果提供）
    5. 计算加权和输出

    eg:
         query, key, value的形状类似于(30, 8, 10, 64), (30, 8, 11, 64),(30, 8, 11, 64)。
         其中  30=batch.size，即当前batch中有多少一个序列；
              8=head.num，注意力头的个数；
              10=目标序列中词的个数；
              64=每个词对应的向量表示；
              11=源语言序列传过来的memory中，当前序列的词的个数，
              64是每个词对应的向量表示。
         这里假定query来自target language sequence；key和value都来自source language sequence.
    """
    # 获取键向量的维度 d_k
    # eg：64=d_k
    d_k = query.size(-1)
    
    # 1. 计算查询和键的点积，并除以 sqrt(d_k) 进行缩放
    # 这一步计算注意力分数，表示查询和键之间的相似度
    # 除以 sqrt(d_k) 是为了防止点积结果过大，导致 softmax 梯度消失

    # eg： 先是(30, 8, 10, 64)和(30, 8, 64, 11)相乘，（注意是最后两个维度相乘）得到(30,8,10,11)，代表10个目标语言序列中每个词和11个源语言序列的分别的"亲密度"。
    # 然后除以sqrt(d_k)=8，防止过大的亲密度，scores的shape是(30, 8, 10, 11)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用掩码（如果提供）
    # 使用mask，对已经计算好的scores，按照mask矩阵，填-1e9，使得下一步计算softmax的时候，被设置成-1e9的数对应的值~0,被忽视
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 3. 使用 softmax 计算注意力权重
    # 将注意力分数转换为概率分布
    # eg：对scores的最后一个维度执行softmax，得到的还是一个tensor, (30, 8, 10, 11)
    p_attn = scores.softmax(dim=-1)
    
    # 4. 应用 dropout（如果提供）
    # 随机丢弃部分注意力权重，防止过拟合
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # 5. 计算加权和输出
    # 使用注意力权重对值向量进行加权求和
    # eg：返回的第一项，是(30,8,10, 11)乘以（最后两个维度相乘）value=(30,8,11,64)，得到的tensor是(30,8,10,64)，和query的最初的形状一样。
    # 另外，返回p_attn，形状为(30,8,10,11)，注意，这里返回p_attn主要是用来可视化显示多头注意力机制。
    return torch.matmul(p_attn, value), p_attn

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """使用贪心算法进行解码"""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
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