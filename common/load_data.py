# =====================================================================
# 数据加载与预处理模块（common/load_data.py）
# =====================================================================
# 本模块封装了多个数据集的数据加载、预处理和特征工程流程，
# 为不同的机器学习任务提供统一的数据接口。
#
# 包含以下函数：
#   load_digtial_data()  → 手写数字识别（MNIST，回归/分类任务）
#   get_device()         → 自动检测最佳可用计算设备
#   get_house_data()     → 房价预测（回归任务，混合数据类型）
#   get_fashion_data()   → 时尚物品分类（Fashion-MNIST，CNN 图像分类）
# =====================================================================

import pandas as pd                                   # 数据处理，提供 DataFrame
import torch                                          # PyTorch 深度学习框架
from sklearn.compose import ColumnTransformer          # 对不同列分别应用不同的预处理管道
from sklearn.impute import SimpleImputer               # 缺失值填充（均值、常数等策略）
from sklearn.model_selection import train_test_split   # 数据集划分
from sklearn.pipeline import Pipeline                  # 预处理步骤的管道化执行
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
# MinMaxScaler      → 归一化到 [0, 1] 区间（适合图像像素、神经网络输入）
# StandardScaler    → 标准化到均值0、标准差1（适合数值特征差异大时）
# OneHotEncoder     → 将类别文字转为独热编码向量（用于处理非数值特征）


# =====================================================================
# 函数：load_digtial_data()
# 功能：加载并预处理 MNIST 手写数字数据集（CSV 格式）
# =====================================================================
# 数据来源：data/train.csv（相对于 common/ 的上级目录）
# 数据格式：每行一张 28×28 的手写数字灰度图
#   第 0 列 label：数字标签（0~9）
#   第 1~784 列：像素灰度值（0~255）
# 样本数量：42000 行
#
# 处理流程：
#   1. CSV 加载 → Pandas DataFrame
#   2. 特征/标签分离
#   3. 训练集/测试集划分（80/20）
#   4. MinMaxScaler 归一化（0~255 → 0~1）
#   5. 转换为 PyTorch Tensor
#
# 返回值：
#   x_train (Tensor): 训练集特征，形状 (33600, 784)，float32
#   x_test  (Tensor): 测试集特征，形状 (8400, 784)，float32
#   y_train (Tensor): 训练集标签，形状 (33600,)，int64
#   y_test  (Tensor): 测试集标签，形状 (8400,)，int64
# =====================================================================
def load_digtial_data():
    # 1. 加载 CSV 数据
    # 路径 ../data/train.csv 是因为当前文件在 common/ 子目录下
    data = pd.read_csv("data/train.csv")

    # 2. 分离特征 X 和标签 y
    # drop("label", axis=1)：删除 label 列，保留所有像素列作为特征（axis=1 表示列）
    x = data.drop("label", axis=1)
    # 标签 y：仅取 label 列（数字 0~9）
    y = data["label"]

    # 3. 划分数据集
    # test_size=0.2    → 20% 测试，80% 训练
    # random_state=42  → 固定随机种子，保证每次运行划分一致
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 4. 数据归一化（Min-Max Scaling）
    # 将像素值从 [0, 255] 缩放到 [0, 1]
    # 公式：x_scaled = (x - min) / (max - min) = x / 255（因为 min=0, max=255）
    # 为什么归一化？
    #   - 神经网络对输入尺度敏感，大数值会导致梯度爆炸
    #   - 归一化后梯度更新更稳定，收敛更快
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)   # 训练集：学习缩放参数 + 应用缩放
    x_test = scaler.transform(x_test)         # 测试集：仅应用训练集学到的缩放规则
                                               # ⚠️ 不能对测试集 fit，否则数据泄露

    # 5. 转换为 PyTorch Tensor
    # 特征转为 float32（神经网络权重通常是 float32，保持一致）
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()

    # 标签转为 int64/long（CrossEntropyLoss 要求标签为整数类型而非浮点）
    # to_numpy() 先将 Pandas Series 转为 NumPy 数组
    y_train = torch.tensor(y_train.to_numpy())
    y_test = torch.tensor(y_test.to_numpy())
    return x_train, x_test, y_train, y_test


# =====================================================================
# 函数：get_device()
# 功能：自动检测并返回可用的最佳计算设备
# =====================================================================
# 检测优先级（从快到慢）：
#   1. "cuda"  → NVIDIA GPU（需安装 CUDA 工具包和对应 PyTorch 版本）
#                使用 NVIDIA 的 CUDA 核心进行大规模并行计算，速度最快
#   2. "mps"   → Apple Metal Performance Shaders
#                利用 Apple Silicon（M1/M2/M3 等芯片）的 GPU 加速
#                适用于 Mac 电脑，无需额外安装 CUDA
#   3. "cpu"   → 中央处理器（所有平台通用的后备方案）
#                速度最慢但始终可用，适合小规模数据和调试
#
# 使用方法：
#   device = get_device()
#   model.to(device)       # 模型迁移到设备
#   data = data.to(device) # 数据迁移到设备
# =====================================================================
def get_device():
    if torch.cuda.is_available():
        # CUDA 可用 → NVIDIA GPU
        return "cuda"
    elif torch.backends.mps.is_available():
        # MPS 可用 → Apple Silicon GPU（Mac M1/M2/M3 等）
        return "mps"
    # 以上都不行 → 使用 CPU
    return "cpu"


# =====================================================================
# 函数：get_house_data()
# 功能：加载并预处理房价预测数据集（回归任务）
# =====================================================================
# 数据来源：data/house_prices.csv
# 任务类型：回归（预测房价 SalePrice，一个连续数值）
#
# 与手写数字识别的关键区别：
#   - 回归任务而非分类任务（预测连续数值而非离散类别）
#   - 数据包含混合特征类型（数值型 + 类别型文字）
#   - 存在缺失值需要填充
#   - 类别特征需要编码（One-Hot Encoding）
#   - 数值特征使用 StandardScaler（标准化）而非 MinMaxScaler
#
# 处理流程：
#   1. 加载 CSV + 删除无关列（Id）
#   2. 分离特征和标签（SalePrice）
#   3. 训练/测试划分
#   4. 识别数值列和类别列
#   5. 构建数值处理管道 + 类别处理管道
#   6. ColumnTransformer 分别处理
#   7. 转换为 Tensor
#
# 返回值：
#   x_train, x_test (Tensor float32)：特征
#   y_train, y_test (Tensor float32)：目标房价
# =====================================================================
def get_house_data():
    # 1. 加载数据 + 删除无关特征
    data = pd.read_csv("../data/house_prices.csv")
    # Id 列只是行编号，不携带任何与房价相关的信息，直接删除
    # inplace=True：在原 DataFrame 上直接修改，不创建副本
    data.drop("Id", axis=1, inplace=True)

    # 2. 分离特征 X 和标签 y（回归任务）
    # SalePrice 是我们要预测的目标变量（房屋售价，单位：美元）
    x = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    # 3. 划分数据集（80% 训练 / 20% 测试）
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 4. 特征工程：识别不同类型的列
    # select_dtypes(exclude="object")：排除字符串/对象类型 → 得到数值列
    #   例如：面积、房间数、建造年份等连续数值
    num_features = x.select_dtypes(exclude="object").columns
    # select_dtypes(include="object")：只保留字符串/对象类型 → 得到类别列
    #   例如：房屋风格、材料类型、区域等离散类别
    cat_features = x.select_dtypes(include="object").columns

    # 5. 构建数值特征的处理管道（Pipeline）
    # Pipeline 将多个预处理步骤串联起来，前一步的输出自动成为后一步的输入
    # 好处：
    #   - 代码整洁，所有步骤封装在一起
    #   - 防止数据泄露（fit/transform 一致应用于所有步骤）
    num_pipeline = Pipeline([
        # 步骤1：缺失值填充 → 用该列的均值填充 NaN
        # strategy="mean"：取该特征列所有非空值的平均值进行填充
        # 为什么用均值？数值型数据的中心趋势，对异常值有一定容忍度
        ("impute", SimpleImputer(strategy="mean")),

        # 步骤2：标准化 → 将数值缩放到均值 0、标准差 1 的分布
        # 公式：x_scaled = (x - mean) / std
        # 为什么要标准化而非归一化（MinMaxScaler）？
        #   - StandardScaler 对异常值更鲁棒（z-score 方法）
        #   - 适合假设数据服从正态分布的模型（如线性回归）
        #   - 房价数据中可能有极端的豪宅，MinMaxScaler 会被极端值压缩
        ("scaler", StandardScaler())
    ])

    # 6. 构建类别特征的处理管道
    cat_pipeline = Pipeline([
        # 步骤1：缺失值填充 → 用字符串 "NaN" 填充缺失的类别值
        # strategy="constant", fill_value="NaN"：把缺失值当作一个独立的类别
        # 为什么不用众数（most_frequent）？
        #   - 缺失本身可能携带信息（如"未填写"可能意味着某些特定情况）
        #   - 用 "NaN" 标记让 OneHotEncoder 为缺失值单独创建一个列
        ("imputer", SimpleImputer(strategy="constant", fill_value="NaN")),

        # 步骤2：独热编码（One-Hot Encoding）
        # 将类别文字转换为数值向量，例如：
        #   颜色列：["红", "蓝", "绿"] → 红=[1,0,0], 蓝=[0,1,0], 绿=[0,0,1]
        #
        # 参数说明：
        #   handle_unknown="ignore"：遇到训练集中没见过的类别时，全部置 0
        #     （而非报错，增强了模型的鲁棒性）
        #   drop="first"：丢弃第一个类别列（如上例丢弃"红"列）
        #     目的：避免虚拟变量陷阱（dummy variable trap）
        #     如果保留所有列，列之间会线性相关（Σ=1），导致多重共线性
        #   sparse_output=False：返回稠密矩阵而非稀疏矩阵（配合 Tensor 使用）
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    # 7. ColumnTransformer：对不同列组合应用不同的处理管道
    # 这是 sklearn 中处理混合数据类型的核心工具
    #   - ("num", num_pipeline, num_features)：对数值列用数值管道处理
    #   - ("cat", cat_pipeline, cat_features)：对类别列用类别管道处理
    # 未被列出的列会被自动丢弃
    ct = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    # 应用转换（与普通 Scaler 一样的模式）
    x_train = ct.fit_transform(x_train)   # 训练集：学习所有预处理参数 + 转换
    x_test = ct.transform(x_test)         # 测试集：仅转换（使用训练集学到的参数）

    # 8. 转换为 PyTorch Tensor
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()

    # 标签也转为 float32（回归任务的标签是连续值）
    # .values 将 Pandas Series 转为 NumPy 数组
    y_train = torch.tensor(y_train.values).float()
    y_test = torch.tensor(y_test.values).float()

    return x_train, x_test, y_train, y_test


# =====================================================================
# 函数：get_fashion_data()
# 功能：加载并预处理 Fashion-MNIST 时尚物品数据集（图像分类任务）
# =====================================================================
# 数据来源：
#   - data/fashion-mnist_train.csv（训练集）
#   - data/fashion-mnist_test.csv（测试集）
#
# Fashion-MNIST 是 MNIST 的变体，每个样本是一张 28×28 的时尚物品灰度图
#   10 个类别：T恤、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴
#   数据格式与 MNIST 完全一致：784 像素 + 1 标签
#
# 与 MNIST 处理的关键区别：
#   - 数据已预先划分好训练/测试集（两个独立文件），不需要再 split
#   - 图像 reshape 为 (N, 1, 28, 28) 的四维张量
#     这是 CNN（卷积神经网络）要求的输入格式：
#       N  = 样本数（batch size）
#       1  = 通道数（灰度图只有 1 个通道，RGB 图则有 3 个）
#       28 = 图像高度（像素）
#       28 = 图像宽度（像素）
# =====================================================================
def get_fashion_data():
    # 1. 分别加载训练集和测试集（两个独立的 CSV 文件）
    train_data = pd.read_csv("../data/fashion-mnist_train.csv")
    test_data = pd.read_csv("../data/fashion-mnist_test.csv")

    # 2. 分离特征和标签
    # iloc[:, 1:]：训练集特征（第 1 列到最后一列，即 784 个像素）
    x_train = train_data.iloc[:, 1:]
    x_test = test_data.iloc[:, 1:]

    # iloc[:, 0]：训练集标签（第 0 列，0~9 的类别编号）
    # .values：转为 NumPy 数组
    y_train = train_data.iloc[:, 0].values
    y_test = test_data.iloc[:, 0].values

    # 3. 像素归一化（0~255 → 0~1）
    # 与 MNIST 相同，使用 MinMaxScaler 将像素值缩放到 [0, 1]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 4. 转换为 PyTorch Tensor + 重塑为 CNN 输入格式
    # reshape(-1, 1, 28, 28)：
    #   -1     → 自动推导样本数量（保持原有数量不变）
    #   1      → 通道数（灰度图 = 1，RGB 彩色图 = 3）
    #   28, 28 → 图像的高度和宽度
    #
    # 为什么要 reshape？
    #   CNN 的卷积层 nn.Conv2d 要求输入形状为 (N, C, H, W) 的四维张量
    #   如果不 reshape，输入是 (N, 784) 的二维向量，卷积层无法定位空间信息
    #   CNN 利用 2D 空间结构（相邻像素的关系），而全连接层将像素视为独立的
    x_train = torch.tensor(x_train).reshape(-1, 1, 28, 28).float()
    x_test = torch.tensor(x_test).reshape(-1, 1, 28, 28).float()

    # 标签转为 float32（虽然某些损失函数要求 long，这里先统一转 float）
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    return x_train, x_test, y_train, y_test


# =====================================================================
# 模块自测代码
# 仅在直接运行此文件时执行（python load_data.py），被 import 时不执行
# 用于快速验证数据加载功能是否正常
# =====================================================================
if __name__ == '__main__':
    # 测试 Fashion-MNIST 数据加载
    # 可以观察输出形状确认 reshape 是否正确
    x_train, x_test, y_train, y_test = get_fashion_data()
    print(f"Fashion-MNIST 训练集特征形状：{x_train.shape}")  # 应为 (N, 1, 28, 28)
    print(f"Fashion-MNIST 训练集标签形状：{y_train.shape}")  # 应为 (N,)
    print(f"Fashion-MNIST 测试集特征形状：{x_test.shape}")
    print(f"Fashion-MNIST 测试集标签形状：{y_test.shape}")