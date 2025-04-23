"""
We can begin by trying out a simple copy-task.
Given a random set of input symbols from a small vocabulary, the goal is to generate back those same symbols.
"""
import torch
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from scripts.train import Batch, LabelSmoothing, rate, run_epoch, DummyOptimizer, DummyScheduler, SimpleLossCompute
from src.models.transformer import subsequent_mask, make_model, greedy_decode


def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    """
        生成随机数据用于训练。它创建了一个大小为batch_size x 10的张量，其中每个元素都是从1到V-1的随机整数。
        第一列被设置为1，表示起始符号。
        然后，它将数据克隆并分离为源序列（src）和目标序列（tgt），并返回一个Batch对象。
    """
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)



# Train the simple copy task.
def example_simple_model():
    V = 11
    # 损失函数，用于计算损失。它使用LabelSmoothing类，其中size表示词汇表的大小，padding_idx表示填充符号的索引，smoothing表示平滑系数。
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    # 创建了一个Transformer模型，其中：
    # 输入和输出维度都是V N=2 表示使用2层Transformer编码器和解码器
    model = make_model(V, V, N=2)

    # 使用Adam优化器，学习率为0.5
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    # 学习率调度器，用于调整学习率。它使用LambdaLR类，其中step表示当前的训练步数，model_size表示模型的输入维度，factor表示学习率缩放因子，warmup表示预热步数。
    # 预热步数为400，表示在训练开始时，学习率会逐渐增加，直到达到0.5。
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    train_losses = []
    eval_losses = []
    
    print("开始训练...")
    print("Epoch\t训练损失\t验证损失\t差值")
    print("-" * 50)

    # 训练20个epoch
    for epoch in range(20):
        # 设置模型为训练模式
        # 在训练模式下，模型会启用dropout等训练时使用的功能
        model.train()
        # 使用data_gen生成数据，并使用run_epoch训练模型。
        train_loss, _ = run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        train_losses.append(train_loss)


        # 设置模型为评估模式
        # 在评估模式下，模型会关闭dropout等训练时使用的功能
        model.eval()
        eval_loss, _ = run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        eval_losses.append(eval_loss)

        loss_diff = train_loss - eval_loss
        print(f"{epoch+1}\t{train_loss:.4f}\t{eval_loss:.4f}\t{loss_diff:.4f}")
        
        if epoch > 0:
            if eval_loss > train_loss and abs(loss_diff) > 0.1:
                print(f"警告：可能出现过拟合！训练损失和验证损失差距较大：{loss_diff:.4f}")

        print("-" * 50)
    print("训练完成！")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(eval_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("损失曲线已保存为 loss_curve.png")

    model.eval()
    print("模型参数：", model.parameters())

    

    # src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    src = torch.LongTensor([[0, 1, 2, 3, 4, 4, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)

    print("测试输入：")
    print("输入序列:", src.tolist()[0])
    print("序列长度:", max_len)
    print("掩码形状:", src_mask.shape)
    print("输入张量值:", src)
    print("掩码张量值:", src_mask)

    print("测试模型输出：")
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


if __name__ == '__main__':
    example_simple_model()