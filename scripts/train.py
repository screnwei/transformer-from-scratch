import time

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from src.models.transformer import subsequent_mask


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        """

        :param src: 源语言序列，(batch.size, src.seq.len)
                    二维tensor，第一维度是batch.size；第二个维度是源语言句子的长度
                    例如：[ [2,1,3,4], [2,3,1,4] ]这样的二行四列的， 1-4代表每个单词word的id
        :param tgt: 目标语言序列，默认为空，其shape和src类似，(batch.size, trg.seq.len)，
                    二维tensor，第一维度是batch.size；第二个维度是目标语言句子的长度
                    例如tgt=[ [2,1,3,4], [2,3,1,4] ] for a "copy network"
        :param pad: 源语言和目标语言统一使用的 位置填充符号，'<blank>'
        """

        self.src = src
        # 创建源语言序列的掩码
        # src != pad 生成一个布尔张量，其中True表示非填充位置，False表示填充位置
        # unsqueeze(-2) 在倒数第二个维度上增加一个维度，用于后续的注意力计算
        # 例如：如果src的shape是(batch_size, seq_len)，那么src_mask的shape就是(batch_size, 1, seq_len)
        #

        # 创建源语言序列的掩码
        # 例如：src = [[1, 2, 3, 2], [1, 4, 2, 2]]，pad=2
        # src != pad 得到：[[1, 0, 1, 0], [1, 1, 0, 0]]
        # unsqueeze(-2) 在倒数第二个维度上增加一个维度，用于后续的注意力计算
        # 最终得到：[[[1, 0, 1, 0]], [[1, 1, 0, 0]]]
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            
            # 目标序列去掉最后一个token，用于输入到decoder
            # 例如：如果tgt是[2,1,3,4,5]，那么self.tgt就是[2,1,3,4]
            # 这样做的原因是：在训练时，decoder的输入需要比输出少一个token
            # 因为decoder的每个时间步都是基于前一个时间步的输出进行预测
            self.tgt = tgt[:, :-1]
            # 目标序列去掉第一个token，用于作为decoder的预测目标
            # 例如：如果tgt是[2,1,3,4,5]，那么self.tgt_y就是[1,3,4,5]
            # 这样做的原因是：在训练时，decoder的每个时间步都需要预测下一个token
            # 所以目标序列需要比输入序列少一个token，并且从第二个token开始
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # 创建目标序列的掩码，用于隐藏填充位置和未来单词
        # 例如：假设 tgt = [[2,1,3,4,2], [2,5,6,2,2]]，pad=2
        # 1. 首先创建基础掩码，标记非填充位置
        # (tgt != pad) 得到：[[1,1,1,1,0], [1,1,1,0,0]]
        # unsqueeze(-2) 在倒数第二个维度上增加一个维度，得到：
        # [[[1,1,1,1,0]], [[1,1,1,0,0]]]
        tgt_mask = (tgt != pad).unsqueeze(-2)
        
        # 2. 将基础掩码与后续掩码进行按位与操作
        # subsequent_mask(5) 创建5x5的上三角矩阵：
        # [[1,0,0,0,0],
        #  [1,1,0,0,0],
        #  [1,1,1,0,0],
        #  [1,1,1,1,0],
        #  [1,1,1,1,1]]
        # 最终掩码的计算过程：
        # 对于第一个序列 [1,1,1,1,0]：
        # 第1个位置：1 & [1,0,0,0,0] = [1,0,0,0,0]
        # 第2个位置：1 & [1,1,0,0,0] = [1,1,0,0,0]
        # 第3个位置：1 & [1,1,1,0,0] = [1,1,1,0,0]
        # 第4个位置：1 & [1,1,1,1,0] = [1,1,1,1,0]
        # 第5个位置：0 & [1,1,1,1,1] = [0,0,0,0,0]
        # 最终得到：
        # [[1,0,0,0,0],
        #  [1,1,0,0,0],
        #  [1,1,1,0,0],
        #  [1,1,1,1,0],
        #  [0,0,0,0,0]]
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


class LabelSmoothing(nn.Module):
    """标签平滑（Label Smoothing）实现。
    这是一种正则化技术，用于防止模型在训练时对预测结果过于自信。
    通过将一部分概率分配给非目标类别，使模型预测更加平滑。
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        初始化标签平滑模块
        
        Args:
            size: 目标词表大小
            padding_idx: 填充token的索引
            smoothing: 平滑因子，控制标签平滑的程度
        """
        super(LabelSmoothing, self).__init__()
        # 使用KL散度作为损失函数，reduction="sum"表示对所有元素求和
        self.criterion = nn.KLDivLoss(reduction="sum")
        # 填充索引，用于标记序列中的填充位置， '<blank>' 的id
        self.padding_idx = padding_idx
        # 置信度，表示正确标签的权重，等于1减去平滑因子
        self.confidence = 1.0 - smoothing
        # 平滑因子，用于标签平滑，均分出去的概率值，得分 e.g. 0.4
        self.smoothing = smoothing
        # target vocab size 目标语言词表大小
        self.size = size
        # 存储真实分布，初始化为None
        self.true_dist = None

    def forward(self, x, target):
        """
        前向传播，计算标签平滑后的损失
        
        Args:
            x: 模型预测的log概率分布，形状为 [batch_size, vocab_size]
            target: 目标标签，形状为 [batch_size]
            
        Returns:
            计算得到的KL散度损失
        """
        # 确保预测分布的大小与词表大小一致
        assert x.size(1) == self.size
        
        # 创建真实分布
        true_dist = x.data.clone()
        # 将平滑因子均匀分配给所有非目标类别
        # (self.size - 2) 是因为要排除目标类别和填充token
        true_dist.fill_(self.smoothing / (self.size - 2))
        
        # 将置信度分配给目标类别
        # unsqueeze(1) 将target从[batch_size]变为[batch_size, 1]
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        # 将填充位置的概率设为0
        true_dist[:, self.padding_idx] = 0
        
        # 处理填充位置
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            # 将填充位置对应的分布设为0
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        # 保存真实分布用于可视化或调试
        self.true_dist = true_dist
        
        # 计算KL散度损失
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    """一个简单的损失计算和训练函数类。
    这个类主要用于计算Transformer模型的损失，并返回用于反向传播的损失值。
    """

    def __init__(self, generator, criterion):
        """
        初始化损失计算器
        
        Args:
            generator: 生成器，用于将模型输出转换为预测分布
            criterion: 损失函数，用于计算预测分布和真实标签之间的差异
        """
        self.generator = generator  # 生成器实例
        self.criterion = criterion  # 损失函数实例

    def __call__(self, x, y, norm):
        """
        计算损失值，每次对象被当作函数调用时都会执行
        
        Args:
            x: 模型输出，形状为 [batch_size, seq_len, vocab_size]
            y: 目标标签，形状为 [batch_size, seq_len]
            norm: 归一化因子，通常是批次中的token数量
            
        Returns:
            tuple: (未归一化的损失值, 用于反向传播的损失节点)
        """
        # 使用生成器将模型输出转换为预测分布
        x = self.generator(x)
        
        # 计算损失
        # 1. 将输入张量重塑为二维：[batch_size * seq_len, vocab_size]
        # 2. 将目标标签重塑为一维：[batch_size * seq_len]
        # 3. 计算损失并除以归一化因子
        sloss = (
            #此处会调用 LabelSmoothing 的 forward 方法
            self.criterion(
                x.contiguous().view(-1, x.size(-1)),  # 重塑预测分布
                y.contiguous().view(-1)               # 重塑目标标签
            )
            / norm  # 归一化损失
        )
        
        # 返回未归一化的损失值和用于反向传播的损失节点
        return sloss.data * norm, sloss

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


def run_epoch(
    data_iter,        # 数据迭代器，用于遍历训练数据
    model,            # Transformer模型实例
    loss_compute,     # 损失计算函数
    optimizer,        # 优化器
    scheduler,        # 学习率调度器
    mode="train",     # 运行模式：train/train+log/eval
    accum_iter=1,     # 梯度累积的步数
    train_state=TrainState(),  # 训练状态跟踪器
):
    """训练单个epoch"""
    start = time.time()  # 记录开始时间
    total_tokens = 0     # 总token数
    total_loss = 0       # 总损失
    tokens = 0           # 当前批次的token数
    n_accum = 0          # 梯度累积计数器
    
    # 遍历数据迭代器中的每个批次
    for i, batch in enumerate(data_iter):
        # 前向传播：计算模型输出
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        # 计算损失
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        
        # 如果是训练模式，执行反向传播和参数更新
        if mode == "train" or mode == "train+log":
            loss_node.backward()  # 反向传播
            train_state.step += 1  # 更新步数
            train_state.samples += batch.src.shape[0]  # 更新样本数
            train_state.tokens += batch.ntokens  # 更新token数
            
            # 达到累积步数时更新参数
            if i % accum_iter == 0:
                optimizer.step()  # 更新参数
                optimizer.zero_grad(set_to_none=True)  # 清空梯度
                n_accum += 1  # 更新累积计数器
                train_state.accum_step += 1  # 更新累积步数
            scheduler.step()  # 更新学习率

        # 更新统计信息
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        
        # 每40个批次打印一次训练信息
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]  # 获取当前学习率
            elapsed = time.time() - start  # 计算耗时
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()  # 重置计时器
            tokens = 0  # 重置token计数
            
        # 清理内存
        del loss
        del loss_node
        
    # 返回平均损失和训练状态
    return total_loss / total_tokens, train_state

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
