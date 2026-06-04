# =====================================================================
# 手写数字识别 — 基于 PyTorch 神经网络（完整训练流程）
# =====================================================================
# 本脚本演示了使用 PyTorch 从零开始训练一个多层全连接神经网络
# 来完成手写数字识别任务。包含完整的训练循环和验证循环。
#
# 与 02_pytorch_mlp_inference.py（只做推理）的区别：
#   - 02 脚本是加载别人已训练好的模型参数（.pt 文件），只做预测
#   - 本脚本是自己定义模型、配置优化器、从零训练、每轮评估验证
#
# 模型架构（5 层全连接网络）：
#   输入(784) → 100 → ReLU → 50 → ReLU → 25 → ReLU → 20 → ReLU → 60 → ReLU → 输出(10)
#
# 关键概念：
#   - DataLoader：批量加载数据，支持打乱和多线程
#   - CrossEntropyLoss：多分类交叉熵损失（内部集成 Softmax）
#   - SGD 优化器：随机梯度下降，沿梯度反方向更新参数
#   - 训练/验证循环：每个 epoch 完整遍历一次数据集
# =====================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from common.load_data import load_digtial_data, get_device

# ====================== 超参数配置 ======================
# 超参数（Hyper-parameter）是在训练开始前人工设定的参数，
# 区别于模型参数（权重 W 和偏置 b，由训练自动学习）
# 好的超参数选择对模型收敛和最终效果至关重要

batch_size = 256    # 批量大小：每次迭代（iteration）喂给模型的样本数
                    #   较大的 batch_size：训练快但内存占用大，梯度估计稳定但泛化可能略差
                    #   较小的 batch_size：训练慢但内存占用小，梯度噪声有助于跳出局部最优
                    #   常用值：32, 64, 128, 256（通常选 2 的幂次）

lr = 0.05           # 学习率（Learning Rate）：控制参数更新的步长
                    #   太大 → 训练不稳定，损失震荡或发散（无法收敛）
                    #   太小 → 收敛速度慢，训练时间过长
                    #   典型范围：0.1, 0.01, 0.001, 0.0001

epochs = 50         # 训练轮数：整个训练集被完整遍历的次数
                    #   太少 → 欠拟合（模型还没学会）
                    #   太多 → 过拟合（模型记住了训练数据但泛化能力差）
                    #   需要通过观察验证集准确率来确定最佳 epoch

# ====================== 设备检测 ======================
# get_device() 自动检测可用的计算设备，优先选择顺序：
#   1. CUDA（NVIDIA GPU）→ 训练速度最快
#   2. MPS（Apple Silicon GPU / Metal Performance Shaders）→ Mac M1/M2/M3 等芯片
#   3. CPU → 最通用但最慢
# 模型和数据都需要 .to(device) 才能在同一设备上计算
device = get_device()
print(f"当前使用的计算设备：{device}")

# ====================== 1. 加载和预处理数据 ======================
# 使用封装好的函数读取 MNIST 手写数字数据
# 返回的 x_train, x_test 已经是归一化后的 float32 Tensor
# 返回的 y_train, y_test 是 int64 Tensor（CrossEntropyLoss 要求标签为 long 类型）
x_train, x_test, y_train, y_test = load_digtial_data()

# ====================== 2. 创建数据集和数据加载器 ======================
# TensorDataset 将特征和标签打包成一个数据集对象
#   作用：将 X 和 y 按索引配对，dataset[i] 返回 (feature_i, label_i)
#   可以看作是 PyTorch 版的 zip(x, y)
train_ds = TensorDataset(x_train, y_train)

# DataLoader 将数据集包装成可迭代的批量加载器，核心参数：
#   dataset：要加载的数据集
#   batch_size：每次取多少样本（这里 256 个样本一批）
#   shuffle=True：每个 epoch 开始时随机打乱数据顺序
#     为什么需要 shuffle？
#     → 避免模型记住样本顺序而非学习真正的特征
#     → 打乱后每批数据更接近整体分布，梯度更新更稳定
#     → 注意：测试集的 DataLoader 一般不需要 shuffle（不影响结果，但也没必要）
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# 验证集的数据封装（测试集在这里作为验证集使用）
val_ds = TensorDataset(x_test, y_test)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

# ====================== 3. 定义神经网络模型 ======================
# 这是一个 6 层全连接网络（5 个隐藏层 + 1 个输出层）
# 设计思路：逐步压缩维度，让网络学习分层抽象的特征表示
#
# 各层详解：
#   nn.Linear(784, 100) → 输入层 → 隐藏层1
#     将 784 维原始像素压缩到 100 维特征空间
#     每个神经元学习一种像素组合模式（如边缘、笔画走向等低级特征）
#
#   nn.Linear(100, 50)  → 隐藏层1 → 隐藏层2
#     进一步压缩到 50 维，提取更抽象的中级特征
#
#   nn.Linear(50, 25)   → 隐藏层2 → 隐藏层3
#   nn.Linear(25, 20)   → 隐藏层3 → 隐藏层4（先压缩再略微扩展）
#   nn.Linear(20, 60)   → 隐藏层4 → 隐藏层5（扩展特征空间，增加表达能力）
#
#   nn.Linear(60, 10)   → 最后一个隐藏层 → 输出层
#     输出 10 个 logits（原始得分），对应数字 0~9
#
# 每层之间使用 ReLU 激活函数：
#   - 引入非线性，否则多层线性层可等价于单层线性层
#   - 将负值置零，正值保留，增加稀疏性
#   - 计算简单，梯度为 0 或 1，缓解梯度消失问题
#
# ⚠️ 注意：输出层后没有 Softmax
#   因为 CrossEntropyLoss 内部会先做 Softmax 再计算交叉熵
#   如果加了 Softmax，会出现"Softmax(Softmax(x))"的双重压缩问题
model = nn.Sequential(
    nn.Linear(784, 100),  # 第1层：784 → 100
    nn.ReLU(),

    nn.Linear(100, 50),   # 第2层：100 → 50
    nn.ReLU(),

    nn.Linear(50, 25),    # 第3层：50 → 25
    nn.ReLU(),

    nn.Linear(25, 20),    # 第4层：25 → 20
    nn.ReLU(),

    nn.Linear(20, 60),    # 第5层：20 → 60
    nn.ReLU(),

    nn.Linear(60, 10),    # 输出层：60 → 10（10个类别的logits）
    # nn.Softmax(dim=-1),  # 训练时不能加！CrossEntropyLoss 内部已包含
)

# .to(device) 将模型的所有参数和缓冲区移动到指定设备（GPU/CPU）
# 这一步必须在定义优化器之前完成
# 因为优化器需要知道参数在哪个设备上，才能正确更新
model.to(device)

# ====================== 4. 定义损失函数 ======================
# CrossEntropyLoss（交叉熵损失）是多分类任务最常用的损失函数
#
# 它内部做了两件事：
#   1. LogSoftmax：将模型的 logits 转为对数概率 → log(softmax(x))
#   2. NLLLoss（负对数似然损失）：取正确类别的负对数概率 → -log(p_correct)
#
# 数学公式：Loss = -log( exp(z_correct) / Σ exp(z_i) )
#                = -z_correct + log( Σ exp(z_i) )
#
# 直观理解：
#   - 当模型对正确类别的预测概率接近 1 时，损失接近 0
#   - 当模型对正确类别的预测概率接近 0 时，损失非常大
#   - 训练的目标是最小化这个损失
#
# ⚠️ 重要：传入的 y_pred 必须是 raw logits（未经过 Softmax）
#         传入的 y_true 必须是类别的整数索引（0~9），不能是 one-hot 编码
loss_fn = nn.CrossEntropyLoss()

# ====================== 5. 定义优化器 ======================
# SGD（Stochastic Gradient Descent，随机梯度下降）
# 每次取一个 batch 的数据计算梯度并更新参数
#
# 参数更新公式：θ_new = θ_old - lr × ∇L(θ)
#   其中 θ 是模型参数，lr 是学习率，∇L(θ) 是损失对参数的梯度
#
# model.parameters() 返回模型中所有需要训练的参数（权重和偏置）
# lr=0.05 是学习率（之前已定义）
#
# 其他常见优化器：
#   - Adam：自适应学习率，收敛快，是默认首选
#   - AdamW：Adam + 权重衰减解耦，目前最常用
#   - RMSprop：适合 RNN 和非平稳目标
optimizer = optim.SGD(model.parameters(), lr=lr)

# ====================== 6. 训练循环 ======================
# 外层循环：epoch（轮次），每轮完整遍历一次训练集
# 内层循环：iteration（迭代），每次处理一个 batch 的数据
#
# 训练集共约 33600 个样本，batch_size=256
# 每个 epoch 有 33600/256 ≈ 132 次迭代
# 总共 50 个 epoch，即 50 × 132 = 6600 次参数更新

for epoch in range(epochs):

    # ====== 训练阶段 ======
    # train_loss_total：累加每个 batch 的损失，用于计算整轮的平均训练损失
    train_loss_total = 0
    # train_acc_num：累加每个 batch 中预测正确的样本数，用于计算训练集准确率
    train_acc_num = 0

    model.train()  # 设置为训练模式
    #   - 启用 Dropout（如果有）
    #   - 启用 BatchNorm 的统计更新（如果有）
    #   - 启用梯度计算
    #   虽然本模型没有 Dropout/BatchNorm，但这是最佳实践

    for input, target in train_loader:
        # 将数据移动到与模型相同的设备（GPU 或 CPU）
        input, target = input.to(device), target.to(device)

        # ① 前向传播（Forward Pass）
        # 数据依次流过网络的所有层，得到预测结果
        # input (256, 784) → ... → y_pred (256, 10)
        # 10 个 logit 值对应 10 个数字类别
        y_pred = model(input)

        # ② 计算损失（Loss Computation）
        # 比较模型预测 y_pred 和真实标签 target
        # CrossEntropyLoss 内部自动做 Softmax 再计算交叉熵
        # loss 是一个标量 Tensor，用于反向传播
        loss = loss_fn(y_pred, target)

        # ③ 反向传播（Backward Pass / Backpropagation）
        # 自动计算 loss 对每个参数的梯度（∂L/∂W, ∂L/∂b）
        # 梯度存储在 parameter.grad 中
        # 这是 PyTorch 自动微分（autograd）的核心功能
        loss.backward()

        # ④ 更新参数（Parameter Update）
        # 优化器使用刚才计算的梯度来更新模型参数
        # SGD：θ = θ - lr × grad
        optimizer.step()

        # ⑤ 清零梯度（Zero Gradients）
        # ⚠️ 关键步骤！PyTorch 默认会累积梯度（相加而非覆盖）
        # 如果不清零，当前 batch 的梯度会叠加上之前所有 batch 的梯度
        # 导致梯度越来越大，参数更新方向错误
        # 必须在每次 optimizer.step() 之后调用 optimizer.zero_grad()
        optimizer.zero_grad()

        # 累加当前 batch 的损失值（.item() 将 Tensor 转为 Python float）
        train_loss_total += loss.item()

        # 累加当前 batch 中预测正确的样本数
        # model(input) 重新做一次前向传播来计算训练准确率
        #   （这里可以复用 y_pred，但为了代码清晰单独计算亦可）
        # .argmax(dim=-1) 取 logits 最大值的索引作为预测类别
        # == target 比较得到布尔值，.sum() 统计 True 的数量
        train_acc_num += (model(input).argmax(dim=-1) == target).sum().item()

    # 训练集平均损失 = 所有 batch 损失之和 / batch 数量
    train_loss = train_loss_total / len(train_loader)
    # 训练集准确率 = 预测正确的样本数 / 训练集总样本数
    train_acc = train_acc_num / len(train_ds)

    # ====== 验证阶段 ======
    # 验证集用于评估模型在未见过数据上的表现
    # 如果训练准确率很高但验证准确率很低 → 过拟合（overfitting）
    # 如果训练准确率和验证准确率都很低 → 欠拟合（underfitting）

    val_loss_total = 0
    val_acc_num = 0

    model.eval()  # 设置为评估/验证模式
    #   - 禁用 Dropout（所有神经元都参与计算）
    #   - 冻结 BatchNorm 的统计量（使用训练期间累积的均值/方差）
    #   - 虽然本模型没有这些层，但这是必须有的最佳实践

    # torch.no_grad() 上下文管理器：
    #   禁用梯度计算和自动微分图的构建
    #   好处：
    #     1. 大幅减少内存占用（不需要存储中间激活值用于反向传播）
    #     2. 加快计算速度（不需要追踪梯度计算图）
    # ⚠️ 本脚本中漏掉了 with torch.no_grad(): 包在验证循环外
    #    虽然不影响正确性（因为没调用 backward），但会浪费内存

    for input, target in val_loader:
        input, target = input.to(device), target.to(device)

        # ① 前向传播（验证阶段不做反向传播和参数更新）
        y_pred = model(input)

        # ② 计算损失
        loss = loss_fn(y_pred, target)

        # 累加验证损失
        val_loss_total += loss.item()

        # 累加验证集预测正确的样本数
        val_acc_num += (model(input).argmax(dim=-1) == target).sum().item()

    # 验证集平均损失
    val_loss = val_loss_total / len(val_loader)
    # 验证集准确率
    val_acc = val_acc_num / len(val_ds)

    # ====== 打印训练进度 ======
    # 每轮输出四个指标：
    #   train_loss：训练损失（越小越好，反映模型对训练数据的拟合程度）
    #   train_acc： 训练准确率（越高越好）
    #   val_loss：  验证损失（越小越好，反映模型的泛化能力）
    #   val_acc：   验证准确率（越高越好，是最终关心的指标）
    #
    # 观察技巧：
    #   - train_loss 和 val_loss 都下降 → 模型在学习，正常 ✅
    #   - train_loss 下降但 val_loss 上升 → 过拟合 ⚠️，需要正则化或早停
    #   - 两者都降不下去 → 欠拟合，可能需要更复杂的模型或更多训练
    print(
        f'第{epoch + 1}轮, '
        f'训练损失: {train_loss:.4f}, '
        f'训练准确率: {train_acc:.4f}, '
        f'验证损失: {val_loss:.4f}, '
        f'验证准确率: {val_acc:.4f}'
    )

# ====================== 8. 保存训练好的模型 ======================
# torch.save() 将模型的 state_dict（所有层的 W 和 b）保存到磁盘
# 保存后可以用 02_pytorch_mlp_inference.py 的方式加载复用
# 文件扩展名通常用 .pt 或 .pth（PyTorch 约定）
torch.save(model.state_dict(), "03_mlp_digital.pt")
print("模型已保存到 03_mlp_digital.pt")

# ====================== 9. 查看训练后各层权重的统计信息 ======================
# 通过检查权重的均值、标准差、最值，可以判断训练是否正常：
#   - 均值接近 0 且标准差适中 → 正常 ✅
#   - 权重全为 0 或全相同 → 梯度消失/未训练 ⚠️
#   - 权重数值极大（如 >100）→ 梯度爆炸，需要调小 lr 或加梯度裁剪 ⚠️
#   - 标准差极小 → 权重萎缩，网络退化，可能需要调整学习率
print("\n==================== 各层权重统计 ====================")
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name:15s} | 均值: {param.mean():+.6f} | "
              f"标准差: {param.std():.6f} | "
              f"最小值: {param.min():+.6f} | "
              f"最大值: {param.max():+.6f}")
print("========================================================\n")

# ====================== 训练完成 ======================
# 训练结束后，model 中保存的是最后一轮（第 50 轮）的参数
# 如果需要保存最佳模型（而非最后一轮的模型），应该在验证准确率最高时
# 用 torch.save(model.state_dict(), "03_mlp_best.pt") 保存检查点