# =====================================================================
# 手写数字识别 — 基于 PyTorch ResNet-18（卷积神经网络 CNN）
# =====================================================================
# 本脚本使用 torchvision 提供的 ResNet-18 预训练架构来完成手写数字识别，
# 是 03_pytorch_mlp_training.py 的升级版。
#
# 与 03 脚本（全连接网络 / MLP）的核心区别：
#   - 03 使用 nn.Sequential 搭建的 6 层全连接网络（MLP）
#   - 04 使用 ResNet-18（深度残差卷积网络），参数量更大、特征提取能力更强
#
# 关键新概念：
#   - CNN（卷积神经网络）：通过卷积核滑动扫描图像，提取局部空间特征
#     全连接层把像素当成独立的向量 → 丢失了"相邻像素有关联"的空间信息
#     CNN 保留了 2D 空间结构 → 能学习边缘、纹理、形状等视觉特征
#
#   - ResNet（残差网络）：通过"跳跃连接"（skip connection）解决深层网络
#     的梯度消失问题，让 18 层乃至 152 层的深度网络都能正常训练
#     核心公式：output = F(x) + x（把输入直接加到输出上，梯度不衰减）
#
#   - 迁移学习：利用在大数据集（ImageNet 1000类）上学到的视觉特征，
#     微调到自己的小任务上（MNIST 10类），加快收敛、提高准确率
#
# ResNet-18 结构概览（修改后）：
#   conv1(1→64, 7×7) → BN → ReLU → [4个残差块×2层] → AvgPool → fc(512→10)
# =====================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models              # torchvision 内置了常用预训练模型（ResNet、VGG 等）

from common.load_data import load_digtial_data, get_device

# ====================== 超参数配置 ======================
batch_size = 512   # 批量大小：ResNet 比 MLP 计算量大，适当增大 batch 利用 GPU 并行能力
                   #   512 是常用的较大 batch，适合 GPU 显存充足时
lr = 0.05          # 学习率：与 02 保持一致
epochs = 50        # 训练轮数：与 02 保持一致

# ====================== 设备检测 ======================
device = get_device()
print(f"当前使用的计算设备：{device}")

# ====================== 1. 加载数据 + 重塑为 CNN 格式 ======================
# load_digtial_data() 返回的数据形状是 (N, 784)，即每张图是 784 维向量
x_train, x_test, y_train, y_test = load_digtial_data()

# CNN 需要 4 维输入：(N, C, H, W)
#   N = 样本数（batch_size）
#   C = 通道数（Channel），灰度图 = 1，RGB 彩色图 = 3
#   H = 图像高度（Height），28 像素
#   W = 图像宽度（Width），28 像素
#
# reshape(-1, 1, 28, 28)：
#   -1     → 自动推导，保持样本数量不变
#   1      → 单通道灰度图
#   28, 28 → 将 784 个像素还原为 28×28 的二维网格
#
# 为什么 CNN 必须保留 2D 结构？
#   - 卷积核（如 3×3）在空间上滑动，需要知道哪些像素是相邻的
#   - 如果输入是 (784,) 的向量，模型只知道第 0 个和第 1 个像素相邻
#     但不知道第 0 个和第 28 个像素在图像上是上下相邻的
#   - 保留 (28, 28) 后，卷积核可以同时看到上下左右相邻的像素
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

# ====================== 2. 创建数据集和数据加载器 ======================
train_ds = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_ds = TensorDataset(x_test, y_test)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

# ====================== 3. 定义模型：ResNet-18 + 自定义修改 ======================
# models.resnet18() 是 torchvision 提供的 ResNet-18 实现
# ResNet-18 的含义：18 层深的残差网络
#   - 1 个初始卷积层 (conv1)
#   - 4 个"阶段"(layer1~4)，每个阶段包含 2 个残差块，每个残差块有 2 个卷积层
#     → 1 + 4×2×2 + 1(全连接) = 18 层
#
# pretrained=False：不加载 ImageNet 预训练权重
#   - True： 加载在 ImageNet（120万张图，1000类）上训练好的权重
#             好处是模型已经学会了通用视觉特征（边缘、纹理、形状），
#             我们只需微调最后几层，收敛极快（迁移学习）
#   - False：权重随机初始化，从头开始训练
#             本脚本设 False 是为了演示从零训练的过程
model = models.resnet18(pretrained=False)

# ====================== 3a. 修改 conv1：适配灰度图输入 ======================
# 原始 ResNet-18 的 conv1：
#   nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#   输入 3 通道（RGB 彩色图），输出 64 通道
#
# 为什么需要修改？
#   - MNIST 是灰度图，只有 1 个通道（不像 ImageNet 的 RGB 三通道）
#   - 如果直接用 3 通道的卷积层处理 1 通道数据，尺寸不匹配会报错
#   - 改造为：nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#
# 参数说明：
#   in_channels=1                           → 输入通道数改为 1（灰度图）
#   out_channels=model.conv1.out_channels   → 保持原始输出通道数（64），不改动
#   kernel_size=7                           → 7×7 大卷积核，适合捕捉整体结构
#                                             （28×28 的图上用 7×7 的感受野较大）
#   stride=2                                → 步长 2，每步跳过 1 个像素
#                                             输出尺寸 = 输入尺寸 / stride = 28/2 = 14
#                                             相当于一次"下采样"，降低计算量
#   padding=3                               → 边缘填充 3 个像素，保证边缘信息不丢失
#   bias=False                              → 不使用偏置（因为后面紧跟 BatchNorm，
#                                             BN 有自己的 β 参数，conv 的 bias 是冗余的）
model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

# ====================== 3b. 删除 MaxPool：适配小尺寸图像 ======================
# 原始 ResNet-18 在 conv1 之后有一个 MaxPool2d(kernel_size=3, stride=2, padding=1)：
#   输入尺寸 14×14（conv1 输出）→ 输出尺寸 7×7（再缩小一半）
#
# 为什么删除？
#   - MNIST 图片只有 28×28，经过 conv1(stride=2) 后变成 14×14
#   - 如果再 MaxPool(stride=2)，变成 7×7 → 再经过 4 个残差块 → 可能只剩 1×1
#   - 特征图太小会导致信息丢失严重，后续卷积层形同虚设
#   - 所以用 nn.Identity() 替换：输入 = 输出，不做任何操作，保留 14×14 的空间尺寸
#
# nn.Identity() 是一个"占位符"层：f(x) = x，什么也不做
#   用它替换 maxpool 后，数据流变成：conv1(28→14) → Identity(14→14) → layer1(14→14) → ...
model.maxpool = nn.Identity()

# ====================== 3c. 修改全连接层：适配 10 分类 ======================
# 原始 ResNet-18 的 fc（fully connected / 全连接层）：
#   nn.Linear(512, 1000)
#   输入 512 维（经过 AvgPool 后的特征向量），输出 1000 维（ImageNet 的 1000 个类别）
#
# 修改为：
#   nn.Linear(model.fc.in_features, 10)
#   输入 512 维（保持不变），输出 10 维（手写数字 0~9）
#
# model.fc.in_features = 512：
#   我们不硬编码 512，而是通过模型属性动态获取
#   这样即使换了 ResNet-34/ResNet-50 等其他版本，代码也不需要改
model.fc = nn.Linear(model.fc.in_features, 10)

# 将模型迁移到计算设备（GPU 或 CPU）
# ⚠️ 必须在定义优化器之前调用 .to(device)
#    因为优化器需要知道参数当前在哪个设备上
model.to(device)

# ====================== 4. 定义损失函数 ======================
# 与 02 脚本相同：CrossEntropyLoss = LogSoftmax + NLLLoss
# 多分类任务的标准选择
loss_fn = nn.CrossEntropyLoss()

# ====================== 5. 定义优化器 ======================
# SGD（随机梯度下降）：与 02 脚本相同
# model.parameters() 现在包含 ResNet-18 的全部参数
# ResNet-18 参数量约 11.7M（1170 万），远多于 02 脚本的 MLP（约 10 万）
# 更多参数意味着更强的表达能力，但也更容易过拟合
optimizer = optim.SGD(model.parameters(), lr=lr)

# ====================== 6. 训练循环 ======================
# 与 02 脚本结构完全相同的训练/验证循环
# 但由于使用了 ResNet-18，每轮训练时间会比 MLP 长很多
# 建议在 GPU 上运行以获得可接受的训练速度

for epoch in range(epochs):

    # ====== 训练阶段 ======
    train_loss_total = 0   # 累加本轮所有 batch 的损失值
    train_acc_num = 0      # 累加本轮所有 batch 预测正确的样本数

    model.train()  # 设置为训练模式
    #   虽然 ResNet-18 有 BatchNorm 层（与 02 的纯 MLP 不同），
    #   model.train() 会启用 BatchNorm 的统计量更新（running_mean/running_var）
    #   如果忘记设 train()，BatchNorm 在训练时会行为异常

    for input, target in train_loader:
        input, target = input.to(device), target.to(device)

        # ① 前向传播：数据流过 ResNet-18 的全部 18 层
        # input (512, 1, 28, 28) → conv1 → BN → ReLU → Identity →
        #   layer1(2个残差块) → layer2(2个残差块) → layer3(2个残差块) →
        #   layer4(2个残差块) → AvgPool → flatten → fc → y_pred (512, 10)
        y_pred = model(input)

        # ② 计算损失
        loss = loss_fn(y_pred, target)

        # ③ 反向传播：计算所有 11.7M 参数的梯度
        loss.backward()

        # ④ 更新参数：SGD 沿梯度方向更新所有权重
        optimizer.step()

        # ⑤ 清零梯度：PyTorch 默认累积梯度，必须手动清零
        optimizer.zero_grad()

        # 累加统计信息
        train_loss_total += loss.item()
        # 计算当前 batch 的预测准确数
        # .argmax(dim=-1)：沿最后一维（10 个类别）取最大值的索引 = 预测的数字
        train_acc_num += (model(input).argmax(dim=-1) == target).sum().item()

    # 训练集平均损失 = 总损失 / batch 数量
    train_loss = train_loss_total / len(train_loader)
    # 训练集准确率 = 正确数 / 总样本数
    train_acc = train_acc_num / len(train_ds)

    # ====== 验证阶段 ======
    val_loss_total = 0
    val_acc_num = 0

    model.eval()  # 设置为评估模式
    #   与 02 脚本的重要区别：ResNet-18 有 BatchNorm 层！
    #   model.eval() 会冻结 BatchNorm 的 running_mean 和 running_var
    #   使用训练期间积累的全局统计量，而非当前 batch 的统计量
    #   如果忘记设置 eval()，验证结果会不稳定（每个 batch 的 BN 统计量不同）

    for input, target in val_loader:
        input, target = input.to(device), target.to(device)

        # ① 前向传播（验证阶段不计算梯度）
        y_pred = model(input)
        # ② 计算损失
        loss = loss_fn(y_pred, target)

        val_loss_total += loss.item()
        val_acc_num += (model(input).argmax(dim=-1) == target).sum().item()

    # 验证集平均损失
    val_loss = val_loss_total / len(val_loader)
    # 验证集准确率
    val_acc = val_acc_num / len(val_ds)

    # ====== 打印训练进度 ======
    # 对比 02 脚本（MLP）和 03 脚本（ResNet-18）的输出：
    #   - ResNet-18 的 train_acc 和 val_acc 通常更高（CNN 更适合图像任务）
    #   - ResNet-18 的 train_loss 下降更快（参数量大、拟合能力强）
    #   - 但要小心过拟合：如果 train_acc ≈ 99% 而 val_acc 停滞不涨，
    #     说明模型在"背"训练集而非真正学习
    print(
        f'第{epoch + 1}轮, '
        f'训练损失: {train_loss:.4f}, '
        f'训练准确率: {train_acc:.4f}, '
        f'验证损失: {val_loss:.4f}, '
        f'验证准确率: {val_acc:.4f}'
    )

# ====================== 与 02 脚本（MLP）的关键对比总结 ======================
#
# | 对比维度         | 03_MLP 全连接网络          | 04_ResNet-18 CNN            |
# |-----------------|--------------------------|------------------------------|
# | 网络类型         | 全连接网络（MLP）          | 卷积神经网络（CNN）            |
# | 层数             | 6 层（5隐藏 + 1输出）      | 18 层（含残差连接）            |
# | 参数量           | ~90,000                   | ~11,700,000（约 130 倍）      |
# | 输入格式         | (N, 784) 二维向量          | (N, 1, 28, 28) 四维张量       |
# | 空间信息         | 丢失（像素当独立特征）       | 保留（2D 卷积保留空间关系）     |
# | 关键机制         | Linear + ReLU             | Conv2d + BN + ReLU + 残差连接 |
# | 梯度消失防护     | 无（靠 ReLU 缓解）          | 残差连接（skip connection）    |
# | 训练速度         | 快                        | 慢（需要 GPU）                |
# | 准确率上限       | ~92-95%                   | ~98-99%+                     |
# | 适用场景         | 简单模式识别、教学演示       | 真实图像分类任务               |
# =====================================================================