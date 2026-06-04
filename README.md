# 🧠 手写数字识别（Digit Recognition）— 从逻辑回归到 ResNet

> 一个循序渐进的手写数字识别教学项目，覆盖 Scikit-learn 经典机器学习到 PyTorch 深度学习，
> 从最简单的逻辑回归到 18 层残差卷积网络，适合课堂教学和自学。

🔗 **GitHub 仓库**：[https://github.com/EdmendZ/digitRecognition](https://github.com/EdmendZ/digitRecognition)

---

## 📁 项目结构

```
digitRecognition/
├── 01_sklearn_softmax.py          # ① Scikit-learn 逻辑回归（Softmax 多分类）
├── 02_pytorch_mlp_inference.py    # ② PyTorch MLP 推理（加载预训练模型）
├── 03_pytorch_mlp_training.py     # ③ PyTorch MLP 完整训练（从零训练 6 层网络）
├── 04_pytorch_resnet_training.py  # ④ PyTorch ResNet-18 CNN（卷积残差网络）
├── main.py                        # 空模板（PyCharm 生成的启动文件）
├── digital.model                  # ① 脚本训练出的 Scikit-learn 模型文件
│
├── common/
│   ├── __init__.py                # 包初始化
│   └── load_data.py               # 数据加载 & 预处理模块（4个数据函数 + 设备检测）
│
└── data/
    ├── train.csv                  # MNIST 数据集（42000 张 28×28 手写数字）
    ├── nn_example.pt              # 预训练 MLP 模型权重（供 ② 加载）
    └── nn_sample                  # 备用模型文件
```

---

## 🗺️ 学习路线图

```
难度 ★☆☆☆☆                  ★★☆☆☆                  ★★★☆☆                 ★★★★☆
                                                          
  ① sklearn           ② PyTorch MLP           ③ PyTorch MLP          ④ PyTorch ResNet
  逻辑回归             加载预训练模型             完整训练                  CNN 训练
     │                     │                      │                      │
     │ 只需 fit()          │ 只需 load+forward    │ 训练循环+SGD         │ CNN+残差连接
     │ 5 行代码训练         │ 5 行代码预测          │ 50 epoch 训练         │ 18 层深度网络
     │                     │                      │                      │
     ▼                     ▼                      ▼                      ▼
  ┌─────────┐         ┌─────────┐           ┌─────────┐           ┌─────────┐
  │ 准确率   │         │ 准确率   │           │ 准确率   │           │ 准确率   │
  │ ~92%    │         │ ~92%    │           │ ~95%    │           │ ~99%    │
  └─────────┘         └─────────┘           └─────────┘           └─────────┘
```

---

## 📊 四种方案全面对比

### 核心差异一览

| 维度 | ① sklearn 逻辑回归 | ② PyTorch MLP 推理 | ③ PyTorch MLP 训练 | ④ PyTorch ResNet-18 |
|:---|:---|:---|:---|:---|
| **框架** | Scikit-learn | PyTorch | PyTorch | PyTorch + torchvision |
| **模型类型** | 线性分类器 | 3层全连接网络 | 6层全连接网络 | 18层残差卷积网络 |
| **参数量** | ~7,850 | ~44,360 | ~89,835 | ~11,700,000 |
| **输入格式** | (N, 784) 向量 | (N, 784) 向量 | (N, 784) 向量 | (N, 1, 28, 28) 张量 |
| **空间信息** | ❌ 丢失 | ❌ 丢失 | ❌ 丢失 | ✅ 保留 2D 结构 |
| **非线性** | ❌ 线性决策边界 | ✅ ReLU 激活 | ✅ ReLU 激活 | ✅ ReLU + BN |
| **训练方式** | `model.fit()` 一行 | ❌ 不训练，直接加载 | 50轮 SGD 训练 | 50轮 SGD 训练 |
| **训练时间(CPU)** | ~30秒 | 无需训练 | ~5分钟 | ~2小时+ |
| **典型准确率** | ~92% | ~92% | ~95% | ~99%+ |
| **代码行数** | ~94行 | ~145行 | ~110行 | ~94行(不含注释) |
| **教学重点** | ML 基础流程 | 模型加载/推理 | 训练循环机制 | CNN/ResNet 原理 |

### 模型架构可视化

```
① sklearn 逻辑回归（线性模型）
   输入(784) ─────────── Linear(784→10) ─── Softmax ─── 输出(10)
                         ↑
                    一层线性变换，相当于 y = softmax(Wx + b)
                    只能画一条直线分开各类，表达能力有限


② PyTorch MLP 推理（3层，预训练权重）
   输入(784) → Linear(784→50) → ReLU → Linear(50→100) → ReLU → Linear(100→10)
              └── 39,250 参数 ──┘       └── 5,100 参数 ──┘     └── 1,010 参数 ──┘


③ PyTorch MLP 训练（6层，从零训练）
   输入(784) → 100 → ReLU → 50 → ReLU → 25 → ReLU → 20 → ReLU → 60 → ReLU → 10
              └─ 压缩阶段 ─┘└── 瓶颈层 ──┘└─ 扩展阶段 ──┘└──── 输出 ────┘
              逐步降低维度，提取越来越抽象的特征，再略微扩展增强表达能力


④ PyTorch ResNet-18（18层，残差卷积网络）
   (1,28,28)                    残差块 ×8                        (10,)
   ─────────→ conv1 ─→ ┌────────┬───────┬───────┬───────┐ ─→ AvgPool → fc ─→ 输出
    7×7, 64     │       │layer1  │layer2 │layer3 │layer4 │     512→10
    stride=2    │       │ 64通道  │128通道│256通道│512通道│
                │       │ 14×14  │ 7×7   │ 4×4   │ 2×2   │
                │       └────────┴───────┴───────┴───────┘
                │
                └→ 每个残差块内部：Conv→BN→ReLU→Conv→BN → + → ReLU
                                                      ↑
                                               输入直接加过来（跳跃连接）
```

---

## 🔄 数据流全景图

```
                        data/train.csv
                     (42000行 × 785列)
                            │
                            ▼
              ┌─────────────────────────┐
              │   common/load_data.py   │
              │                         │
              │  load_digtial_data()    │
              │   1. pd.read_csv()      │
              │   2. 分离 X(784列)/y(1列)│
              │   3. 80/20 划分         │
              │   4. MinMaxScaler 归一化 │
              │   5. 转 torch.Tensor    │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  train_x: (33600, 784)  │
              │  test_x:  (8400, 784)   │
              │  train_y: (33600,)      │
              │  test_y:  (8400,)       │
              └─────────────────────────┘
                            │
          ┌─────────────────┼──────────────────┬──────────────────┐
          ▼                 ▼                  ▼                  ▼
    ① sklearn       ② MLP 推理          ③ MLP 训练         ④ ResNet 训练
   fit()+predict()  load+forward()   50轮SGD训练         50轮SGD训练
          │                 │                  │                  │
          ▼                 ▼                  ▼                  ▼
     ~92% 准确率       ~92% 准确率        ~95% 准确率         ~99% 准确率
```

---

## 🧩 核心概念图谱

```
                           机器学习 / 深度学习 核心概念
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   【数据预处理】             【模型架构】              【训练机制】
        │                         │                         │
  ├─ 特征/标签分离            ├─ 线性层(Linear)          ├─ 前向传播
  ├─ 训练集/测试集划分         ├─ 卷积层(Conv2d)          ├─ 损失函数(Loss)
  │  · 防止过拟合              ├─ 激活函数(ReLU)          │  · CrossEntropyLoss
  │  · test_size 比例          │  · 引入非线性             │  · 交叉熵公式
  ├─ 归一化(Normalization)     │  · 梯度不饱和             ├─ 反向传播(Backward)
  │  · MinMaxScaler [0,1]     ├─ 池化层(Pooling)         │  · 链式法则
  │  · StandardScaler (μ=0,σ=1)│  · 降采样                │  · 梯度计算
  ├─ 独热编码(One-Hot)        ├─ 批归一化(BatchNorm)      ├─ 参数更新
  │  · 类别→向量               │  · 加速收敛               │  · SGD: W=W-lr×∇L
  └─ 缺失值填充                │  · 正则化效果             │  · Adam/AdamW
                               ├─ 残差连接(Skip)          ├─ 梯度清零
                               │  · F(x)+x               │  · zero_grad()
                               │  · 解决梯度消失           ├─ 学习率(lr)
                               └─ Softmax                 │  · 步长控制
                                  · 分数→概率              └─ Batch Size
                                  · Σp=1                     · 批量大小
```

---

## 🎯 各脚本详解

### ① `01_sklearn_softmax.py` — 经典 ML 入门

```
完整 ML 流程：加载数据 → 预处理 → 训练 → 保存 → 评估 → 可视化

关键代码量：~5行核心代码即可完成训练
   scaler = MinMaxScaler()
   model = LogisticRegression(max_iter=10000)
   model.fit(train_x, train_y)
   joblib.dump(model, "digital.model")
   print(model.score(test_x, test_y))

✅ 优点：代码简洁、训练快、适合入门
❌ 局限：线性分类器，无法学习复杂特征
```

### ② `02_pytorch_mlp_inference.py` — 模型加载 + 推理

```
只做推理不做训练，相当于"使用别人训练好的模型"

流程：定义结构 → torch.load() → load_state_dict() → 预测

⚠️ 关键点：
  - 模型结构必须和保存时完全一致（否则 load_state_dict 报错）
  - map_location='cpu' 允许在无 GPU 环境下加载
  - 去掉 load_state_dict，模型就是随机瞎猜（准确率 ~10%）
```

### ③ `03_pytorch_mlp_training.py` — 完整训练循环

```
从零开始训练，理解训练循环的每个步骤：

  for epoch in range(50):           # 外层：轮次
      for input, target in loader:  # 内层：批次迭代
          ① y_pred = model(input)      # 前向传播
          ② loss = loss_fn(y_pred, t)  # 计算损失
          ③ loss.backward()            # 反向传播（求梯度）
          ④ optimizer.step()           # 更新参数
          ⑤ optimizer.zero_grad()      # 清零梯度 ⚠️

🔑 核心概念：
  - 梯度累积机制：PyTorch 默认累加梯度，step() 后必须 zero_grad()
  - model.train() vs model.eval()：影响 Dropout 和 BatchNorm 行为
  - 过拟合判断：train_acc ↑ + val_acc → = 过拟合
```

### ④ `04_pytorch_resnet_training.py` — CNN 深度网络

```
使用 torchvision 的 ResNet-18，做三处关键修改：

修改1：conv1 输入通道 3→1（适配灰度图）
  nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

修改2：删除 MaxPool（图片太小，28×28 不能过度池化）
  model.maxpool = nn.Identity()  # 原样输出，不做下采样

修改3：fc 输出 1000→10（适配手写数字的10个类别）
  model.fc = nn.Linear(512, 10)

ResNet 核心创新 — 残差连接（Skip Connection）：
  ┌──────────────┐
  │   Conv+BN+ReLU │
  │      ↓         │
  │   Conv+BN      │
  │      ↓         │
  │   F(x) + x─────┼── 输入 x 直接加到输出（跳跃连接）
  │      ↓         │   梯度可以通过这条"高速公路"直达浅层
  │    ReLU        │   解决了深层网络的梯度消失问题
  └──────────────┘
```

---

## 🏗️ 架构演进与关键公式

### 从线性到非线性的跨越

```
逻辑回归 (①)：
  y = softmax(Wx + b)                              ← 线性决策边界
  每个像素独立投票，无法捕捉像素间的空间关系

MLP 全连接 (②③)：
  y = W₃·ReLU(W₂·ReLU(W₁x + b₁) + b₂) + b₃        ← 非线性，但丢失空间结构
  784 个像素被当作 784 个独立特征，不知道谁在谁旁边

CNN 卷积 (④)：
  y = fc(AvgPool(ResBlock(Conv(x))))               ← 非线性 + 保留空间结构
  3×3 卷积核在 28×28 的网格上滑动，每次只看一个局部区域
  浅层学边缘/纹理 → 深层学形状/部件 → 最终理解整张图
```

### 关键公式速查

| 概念 | 公式 | 说明 |
|:---|:---|:---|
| **Softmax** | `p_i = exp(z_i) / Σexp(z_j)` | 将 logits 转为概率，Σp = 1 |
| **交叉熵损失** | `L = -log(p_correct)` | 正确类别的负对数概率 |
| **ReLU** | `f(x) = max(0, x)` | 负值置零，引入非线性 |
| **SGD 更新** | `θ = θ - η·∇L(θ)` | 沿梯度反方向更新 |
| **MinMax 归一化** | `x' = (x-min)/(max-min)` | 缩放到 [0, 1] |
| **Standard 标准化** | `x' = (x-μ)/σ` | 缩放到均值0、标准差1 |
| **残差连接** | `y = F(x) + x` | 输入直连输出，梯度不衰减 |
| **卷积输出尺寸** | `H_out = (H-K+2P)/S + 1` | K=核大小, P=填充, S=步长 |

### 训练循环的五个步骤

```
        ┌──────────────────────────────────────────────────────────┐
        │                     一个 Batch 的训练流程                  │
        │                                                          │
        │   input ──────────────────────────────────────┐          │
        │     │                                         │          │
        │     ▼                                         │          │
        │  ┌──────────┐                                 │          │
        │  │ ① forward │  y_pred = model(input)          │          │
        │  └────┬─────┘                                 │          │
        │       │ y_pred                                │          │
        │       ▼                                       │          │
        │  ┌──────────┐    target                       │          │
        │  │ ②  loss  │─────────────┐                   │          │
        │  └────┬─────┘  loss = loss_fn(y_pred, target) │          │
        │       │ loss                                   │          │
        │       ▼                                       │          │
        │  ┌──────────┐                                 │          │
        │  │③ backward│  loss.backward()  计算梯度       │          │
        │  └────┬─────┘  ∂L/∂W 存入 W.grad              │          │
        │       │ grad                                   │          │
        │       ▼                                       │          │
        │  ┌──────────┐                                 │          │
        │  │ ④  step  │  optimizer.step()  更新参数      │          │
        │  └────┬─────┘  W = W - lr × W.grad            │          │
        │       │ 参数已更新                              │          │
        │       ▼                                       │          │
        │  ┌──────────┐                                 │          │
        │  │⑤ zero_grad│  optimizer.zero_grad() 清梯度   │          │
        │  └──────────┘  W.grad = 0     ⚠️ 不能忘记！   │          │
        │                                                │          │
        │  ──────────────────────────────────────────────┘          │
        │  回到 ① 处理下一个 batch（如果不清零，梯度会累加）        │
        └──────────────────────────────────────────────────────────┘
```

---

## 💡 常见问题 FAQ

<details>
<summary><b>Q1: 为什么 01 的准确率只有 92%，而 04 能达到 99%？</b></summary>

- ① 逻辑回归是**线性分类器**，只能画"直线"分类。手写数字的像素分布不是线性可分的
- ④ ResNet-18 有 **1170 万参数 + 非线性激活 + 卷积空间特征**，能捕捉极其复杂的像素模式
- 这是"模型容量"（model capacity）的直接体现：参数越多 → 能学的模式越复杂 → 上限越高
</details>

<details>
<summary><b>Q2: 02 和 03 代码几乎一样，区别在哪？</b></summary>

| | 02（推理） | 03（训练） |
|:---|:---|:---|
| 权重来源 | `torch.load()` 从文件读 | 训练循环自动学到 |
| 有训练循环吗 | ❌ 没有 | ✅ 50 轮 |
| 运行时间 | 秒级 | 分钟级 |
| 作用 | 演示如何复用模型 | 演示如何训练模型 |

02 = 拿着别人写好的答案直接抄，03 = 自己从头做一遍题
</details>

<details>
<summary><b>Q3: 为什么 03 的 MLP 不把输入 reshape 成 (28,28)，04 的 CNN 却需要？</b></summary>

- **全连接层** `nn.Linear` 对输入位置**无感知**：`x[0]` 和 `x[27]` 只是两个不同的数字，它不知道它们在图像上是左边缘和右边缘
- **卷积层** `nn.Conv2d` 的卷积核在 **2D 空间上滑动**：它明确知道 `(i,j)` 和 `(i,j+1)` 是水平相邻的像素
- 因此 CNN **必须**保留 2D 结构才能发挥空间感知的优势，MLP 保留了也没用
</details>

<details>
<summary><b>Q4: loss_fn 不参与训练是什么意思？</b></summary>

- `loss_fn` 是一个**固定的数学公式**（CrossEntropyLoss），没有可训练参数
- 它参与两件事：
  1. **前向**：计算 loss 值（衡量预测和真实值的差距）
  2. **反向**：把梯度传给 y_pred（让模型知道"往哪个方向改"）
- 但它自己**不会被修改**——就像一个打分裁判，规则不变，只告诉学生哪里扣了分
</details>

<details>
<summary><b>Q5: 为什么要 optimizer.zero_grad()？</b></summary>

PyTorch 默认**累加梯度**（不覆盖），设计目的是支持 RNN 等需要梯度累积的场景。

```
忘记清零的后果：
  batch1 梯度 = [0.1, 0.2]
  batch2 梯度 = 累加 → [0.1+0.3, 0.2-0.1] = [0.4, 0.1]  ← 混入了 batch1 的梯度！
  batch3 梯度 = 继续累加 → 梯度越来越大 → 参数更新量失控 → 训练崩溃
```
</details>

---

## 🚀 运行指南

### 环境要求

```bash
# 核心依赖
pip install torch torchvision    # PyTorch 深度学习框架
pip install pandas scikit-learn  # 数据处理 + 经典机器学习
pip install matplotlib joblib    # 可视化 + 模型保存

# 可选：GPU 加速（NVIDIA）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 按顺序学习运行

```bash
# Step 1: 经典机器学习入门（约 30 秒）
python 01_sklearn_softmax.py

# Step 2: 加载预训练模型推理（秒级完成）
python 02_pytorch_mlp_inference.py

# Step 3: 从零训练 MLP（CPU ~5 分钟）
python 03_pytorch_mlp_training.py

# Step 4: 训练 ResNet-18（建议 GPU，CPU ~2 小时）
python 04_pytorch_resnet_training.py
```

### 模块单独测试

```bash
# 测试数据加载模块
cd common && python load_data.py
```

---

## 📚 知识点索引

| 你想学什么 | 去看哪个文件 | 关键行/概念 |
|:---|:---|:---|
| 完整的 ML 流程长什么样 | `01_sklearn_softmax.py` | 数据→预处理→训练→保存→评估→可视化 |
| 归一化的作用和两种方式 | `01_sklearn_softmax.py` + `load_data.py` | MinMaxScaler vs StandardScaler |
| Softmax 是什么 | `01_sklearn_softmax.py` | 第11b段：10个概率之和=1 |
| 过拟合是什么、怎么判断 | `01_sklearn_softmax.py` | train_test_split 的作用 |
| 如何加载别人训练好的模型 | `02_pytorch_mlp_inference.py` | torch.load() + load_state_dict() |
| state_dict 里有什么 | `02_pytorch_mlp_inference.py` | 第3段：W 和 b 的形状 |
| 三层 MLP 结构 | `02_pytorch_mlp_inference.py` | 784→50→100→10 |
| 训练循环五个步骤 | `03_pytorch_mlp_training.py` | forward→loss→backward→step→zero_grad |
| 为什么要梯度清零 | `03_pytorch_mlp_training.py` | 第⑤步：梯度累积机制 |
| model.train() vs eval() | `03_pytorch_mlp_training.py` | BatchNorm/Dropout 行为差异 |
| DataLoader 怎么用 | `03_pytorch_mlp_training.py` | batch_size、shuffle 参数 |
| 超参数怎么调 | `03_pytorch_mlp_training.py` | lr、batch_size、epochs |
| CNN 为什么比 MLP 强 | `04_pytorch_resnet_training.py` | 空间信息保留 |
| ResNet 残差连接原理 | `04_pytorch_resnet_training.py` | F(x)+x 跳跃连接 |
| 如何修改预训练模型适配新任务 | `04_pytorch_resnet_training.py` | conv1/fc 修改、nn.Identity() |
| 混合数据类型怎么处理 | `load_data.py` | ColumnTransformer + Pipeline |
| 独热编码是什么 | `load_data.py` | OneHotEncoder, drop="first" |
| 设备检测 (GPU/MPS/CPU) | `load_data.py` | get_device() |

---

## 🔧 自定义练习建议

1. **调参实验**：修改 `03_pytorch_mlp_training.py` 中的 `lr`（试试 0.001, 0.01, 0.1, 1.0），观察训练曲线的变化
2. **架构实验**：在 ③ 的 `nn.Sequential` 中增删隐藏层，看准确率如何变化
3. **迁移学习**：把 ④ 的 `pretrained=False` 改为 `True`，观察收敛速度的巨大差异
4. **换数据集**：用 `load_data.py` 中的 `get_fashion_data()` 替换 MNIST，挑战 10 类时尚物品识别
5. **混合数据处理**：研究 `get_house_data()` 中的特征工程管道，理解数值/类别特征的不同处理方式
