# =====================================================================
# 手写数字识别 — 基于 PyTorch 神经网络（加载预训练模型进行推理）
# =====================================================================
# 本脚本演示如何使用 PyTorch 加载一个已经训练好的神经网络模型，
# 并在测试集上进行预测和评估。
#
# 与 01_sklearn_softmax.py（使用 Scikit-learn 逻辑回归）的区别：
#   - 逻辑回归是线性分类器，只能学习线性决策边界
#   - 本脚本使用多层全连接神经网络 + ReLU 激活函数，能够学习非线性特征
#   - 本脚本的模型是提前训练好并保存到 .pt 文件中的，这里只做加载和推理
#
# 模型架构（保存在 data/nn_example.pt 中）：
#   输入层(784) → 隐藏层1(50) → ReLU → 隐藏层2(100) → ReLU → 输出层(10)
# =====================================================================

import torch
import torch.nn as nn

from common.load_data import load_digtial_data

# ====================== 1. 读取数据 ======================
# load_digtial_data() 封装了完整的数据预处理流程（详见 common/load_data.py）：
#   1. 从 CSV 加载 42000 张手写数字图片
#   2. 分离特征（784 像素）和标签（数字 0~9）
#   3. 按 80/20 划分训练集和测试集
#   4. MinMaxScaler 归一化：像素值 0~255 → 0~1
#   5. 转换为 PyTorch Tensor（float32）
# 返回四个 Tensor：train_x, test_x, train_y, test_y
train_x, test_x, train_y, test_y = load_digtial_data()

# ====================== 2. 定义模型结构 ======================
# nn.Sequential 是一种"顺序容器"，将多个网络层按顺序串联起来
# 数据会依次流过每一层，前一层的输出自动成为下一层的输入
#
# 层详解：
#   nn.Linear(784, 50)  → 全连接层（线性变换层）
#     输入 784 维（28×28 像素展平后的向量）
#     输出 50 维（第一个隐藏层，50 个神经元）
#     数学：y = Wx + b，其中 W 的形状是 (50, 784)，b 的形状是 (50,)
#     参数数量：784 × 50 + 50 = 39,250
#
#   nn.ReLU()  → ReLU（Rectified Linear Unit）激活函数
#     公式：f(x) = max(0, x)
#     作用：引入非线性，让网络能够学习复杂特征
#     如果没有激活函数，多层线性层等价于单层线性层（矩阵乘法可合并）
#     ReLU 的优点：计算简单、梯度不饱和、缓解梯度消失
#
#   nn.Linear(50, 100) → 第二隐藏层
#     输入 50 维，输出 100 维（增加神经元数量，提取更丰富的特征）
#     参数数量：50 × 100 + 100 = 5,100
#
#   nn.Linear(100, 10) → 输出层
#     输入 100 维，输出 10 维（对应数字 0~9 共 10 个类别）
#     输出的 10 个数值称为 logits（原始得分，尚未归一化）
#     参数数量：100 × 10 + 10 = 1,010
#
# ⚠️ 为什么训练时不用 Softmax？
#   nn.CrossEntropyLoss 内部已经集成了 Softmax + 负对数似然（NLL）
#   如果模型输出已经经过 Softmax，再套一层 Softmax 会导致：
#     1. 概率被双重压缩，梯度变得极小
#     2. 模型无法正常训练
#   正确的做法：训练时模型输出 raw logits，由损失函数内部处理 Softmax
model = nn.Sequential(
    nn.Linear(28 * 28, 50),   # 输入层 → 隐藏层1：784 → 50
    nn.ReLU(),                 # 非线性激活

    nn.Linear(50, 100),        # 隐藏层1 → 隐藏层2：50 → 100
    nn.ReLU(),                 # 非线性激活

    nn.Linear(100, 10),        # 隐藏层2 → 输出层：100 → 10（10个类别的logits）
    # nn.Softmax(dim=-1)       # ⚠️ 训练时注释掉！因为 CrossEntropyLoss 内部已包含 Softmax
)

# ====================== 3. 加载预训练模型参数 ======================
# torch.load() 从磁盘加载之前保存的模型参数（state_dict）
# state_dict 是一个 Python 字典：
#   {
#       '0.weight': Tensor(50, 784),   # 第一层 Linear 的权重
#       '0.bias':   Tensor(50,),       # 第一层 Linear 的偏置
#       '2.weight': Tensor(100, 50),   # 第二层 Linear 的权重
#       '2.bias':   Tensor(100,),      # 第二层 Linear 的偏置
#       '4.weight': Tensor(10, 100),   # 第三层 Linear 的权重
#       '4.bias':   Tensor(10,),       # 第三层 Linear 的偏置
#   }
#
# map_location=torch.device('cpu')：
#   强制将模型参数加载到 CPU 上
#   如果模型是在 GPU 上训练的但在无 GPU 环境下加载，必须指定此参数
#   否则会因为 CUDA 不可用而报错
state_dict = torch.load("data/nn_example.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)  # 将加载的参数复制到模型各层中

# ====================== 4. 进行预测（推理 / Inference） ======================
# model(test_x) 对测试集中所有样本进行前向传播
# y_pred 的形状：(8400, 10)
#   8400 个测试样本，每个样本输出 10 个 logits（对应 0~9 每个数字的原始得分）
y_pred = model(test_x)

# torch.argmax(y_pred, dim=-1)：
#   沿着最后一个维度（dim=-1，即 10 个类别维度）找最大值的索引
#   例如 y_pred[0] = [0.1, 0.2, 8.5, 0.3, ...]
#        → argmax 返回 2（第 2 个位置的值 8.5 最大，即预测为数字 2）
# y_pred_class 的形状：(8400,)，每个元素是 0~9 的预测数字
y_pred_class = torch.argmax(y_pred, dim=-1)

# ====================== 5. 计算准确率（Accuracy） ======================
# 准确率 = 预测正确的样本数 / 总样本数
#
# 三种等价的写法来计算预测正确的样本数：
#
# 方法1：(y_pred_class == test_y).sum()
#   == 运算符逐元素比较，返回布尔型 Tensor
#   .sum() 将 True 计为 1，False 计为 0，求和得到正确预测的数量
#   acc_count = (y_pred_class == test_y).sum()
#
# 方法2：y_pred_class.eq(test_y).sum()
#   .eq() 是 PyTorch 的逐元素相等判断方法，等同于 ==
#   acc_count = (y_pred_class.eq(test_y)).sum()
#
# 方法3（当前使用）：torch.sum(y_pred_class == test_y)
#   torch.sum() 与 .sum() 功能相同，仅仅写法不同
acc_count = torch.sum(y_pred_class == test_y)

# 准确率 = 正确数 / 总数
# .item() 将标量 Tensor 转为 Python float（便于打印和后续计算）
# len(test_y) = 8400（测试集样本总数）
acc = acc_count.item() / len(test_y)
print(f"模型在测试集上的准确率：{acc:.4f} ({acc*100:.2f}%)")

# ====================== 6. 单张图片预测示例 ======================
# test_x[33:34] 取第 34 张图片（索引从 33 开始，到 34 结束但不包含 34）
# 切片写法 [33:34] 而非 [33]，是为了保持 batch 维度：
#   [33]    → 形状 (784,)，缺少 batch 维度，模型不接受
#   [33:34] → 形状 (1, 784)，有 batch 维度，模型可以处理
# 输出：模型对这 1 张图预测的 10 个 logits 值 + 真实标签
print("第34张图片的预测logits和真实标签：")
print(model(test_x[33:34]), test_y[33:34])

# ====================== 7. 找出所有预测错误的样本 ======================
# 遍历所有 8400 个测试样本，打印出模型预测错误的样本索引
# y_pred_class == test_y 返回布尔型 Tensor，True=预测正确，False=预测错误
# not i 将 False（预测错误）变为 True，进入 if 分支打印索引
c = 0
for i in y_pred_class == test_y:
    if not i:                    # 如果预测错误（i 为 False）
        print(f"第 {c} 个样本预测错误")
    c += 1