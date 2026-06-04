# ====================== 1. 导入需要的工具库 ======================
import matplotlib
import pandas as pd                 # 读取CSV表格数据
from sklearn.linear_model import LogisticRegression  # 逻辑回归（分类模型）
from sklearn.model_selection import train_test_split # 划分训练集/测试集
from sklearn.preprocessing import MinMaxScaler      # 数据归一化（0~1缩放）
import matplotlib.pyplot as plt     # 画图显示手写数字
import joblib                      # 保存/加载训练好的模型

# ====== 添加这段代码解决中文显示问题 ======
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
# ======================================

# ====================== 2. 读取数据集 ======================
# （42000行，每行=一张28×28 = 784像素的图像）
dataset = pd.read_csv("../data/train.csv")

# ====================== 3. 划分特征X 和 标签y ======================
# 特征X：所有行，从第1列之后的所有列 → 784个像素值
x = dataset.iloc[:, 1:]
# 标签y：所有行，第0列 → 数字0~9（正确答案）
y = dataset.iloc[:, 0]

print("x.shape表示42000张图，每张图784个像素： ",x.shape) # 输出：(42000, 784) 表示42000张图，每张图784个像素
print("y.shape表示42000个标签： ", y.shape) # (42000,) 表示42000个标签

# ====================== 4. 划分训练集 和 测试集 ======================
# train_test_split：把数据分成80%训练，20%测试
# test_size=0.2 → 20%用作测试
# random_state=42 → 固定随机种子，保证每次运行结果一样
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# ====================== 5. 数据归一化（非常重要） ======================
# MinMaxScaler：把像素值 0~255 缩放到 0~1 之间，让模型训练更快更稳
scaler = MinMaxScaler()

# 训练集：fit + transform
train_x = scaler.fit_transform(train_x)
# 测试集：只transform（不能重新fit，必须用训练集的规则）
test_x = scaler.transform(test_x)

# ====================== 6. 创建逻辑回归模型 ======================
# multi_class='multinomial' 是sklearn中默认行为（当n_classes>2时）自动使用softmax函数进行多分类
model = LogisticRegression(max_iter=10000)# max_iter=10000：最大迭代次数（数据量42000，必须加大）

# ====================== 7. 训练模型 ======================
# 让模型学习像素 → 数字的映射关系
model.fit(train_x, train_y)

# ====================== 8. 保存训练好的模型 ======================
joblib.dump(model, "digital.model")

# ====================== 9. 加载模型 ======================
model = joblib.load("digital.model")

# ====================== 10. 模型评估（测试集准确率） ======================
# 输出模型在测试集上的正确率
print("输出模型在测试集上的正确率: ",model.score(test_x, test_y))

# ====================== 11. 单张图片预测 ======================
# 取测试集第20张图，reshape(1,-1) → 变成模型能识别的形状
pred_y = model.predict(test_x[20].reshape(1, -1))

# 获取softmax输出概率
# 输出每个数字的预测概率（0~9的置信度）
# 输出10个概率值（分别对应数字0-9），所有概率之和 = 1，这就是softmax函数的输出结果
probabilities = model.predict_proba(test_x[20].reshape(1, -1))
###############平时关闭此段注释，上课再打开给同学们看画图###########################
# 手动验证：概率求和=1
# print(f"概率总和：{probabilities.sum():.6f}")  # 输出 1.000000
# # 各数字概率可视化
# plt.bar(range(10), probabilities[0])
# plt.xlabel('数字')
# plt.ylabel('预测概率')
# plt.title('Softmax输出的10个类别概率')
# plt.show()
##########################################
print("输出每个数字的预测概率（0~9的置信度）: \n",probabilities)

# 输出最终预测结果（概率最大的数字）
print("输出最终预测结果（概率最大的数字）: ",pred_y)


# ====================== 12. 画出这张手写数字图 ======================
# test_x[20]是784维一维数组 → reshape成28×28图像
plt.imshow(test_x[20].reshape(28, -1), cmap="gray")
plt.show()
print()

# ====================== 13. 最终的教学提问和思考 ======================
print("给同学们讲解，为什么预测是数字8，出来是9 ？")
print("这张图真实标签：", test_y.iloc[20])
