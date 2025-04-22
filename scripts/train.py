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
    "Implement label smoothing. 最终计算损失函数"

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
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


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
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
