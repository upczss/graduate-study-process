# 张量计算
## 初始化张量
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,device=device,requires_grad=True)


print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)//会输出【2，3】代表两行三列的矩阵
print(my_tensor.requires_grad)


# 其他常用的张量初始化方法

x = torch.empty(size=(3, 3))                       # 创建一个 3×3 的未初始化张量，其中的值取决于内存中的原有数据
x = torch.zeros((3, 3))                            # 创建一个 3×3 的全零张量
x = torch.rand((3, 3))                             # 创建一个 3×3 的张量，元素从 [0, 1) 的均匀分布中随机采样
x = torch.ones((3, 3))                             # 创建一个 3×3 的全一张量
x = torch.eye(5, 5)                                # 创建一个 5×5 的单位矩阵，即主对角线为 1，其余位置为 0
x = torch.arange(start=0, end=5, step=1)           # 创建从 0 开始、到 5 之前结束、步长为 1 的一维张量
x = torch.linspace(start=0.1, end=1, steps=10)     # 在 0.1 到 1 之间生成 10 个等间距的数，包含起点和终点
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)  # 创建 1×5 的张量，并用均值为 0、标准差为 1 的正态分布原地填充
x = torch.empty(size=(1, 5)).uniform_(0, 1)        # 创建 1×5 的张量，并用 [0, 1) 均匀分布中的随机数原地填充
x = torch.diag(torch.ones(3))                      # 将长度为 3 的全一张量放在主对角线上，生成 3×3 的单位矩阵



# 如何初始化张量并将其转换为其他数据类型（整数、浮点数、双精度浮点数）

tensor = torch.arange(4)  # 创建包含 0、1、2、3 的一维张量
print(tensor.bool())  # 转换为布尔类型，0 为 False，非零值为 True
print(tensor.short())  # 转换为 16 位整数类型 int16
print(tensor.long())  # 转换为 64 位整数类型 int64（常用且重要）
print(tensor.half())  # 转换为 16 位浮点数类型 float16
print(tensor.float())  # 转换为 32 位浮点数类型 float32（常用且重要）
print(tensor.double())  # 转换为 64 位双精度浮点数类型 float64

# NumPy 数组与 PyTorch 张量之间的相互转换

import numpy as np  # 导入 NumPy，并将其简写为 np
np_array = np.zeros((5, 5))  # 创建一个 5×5 的 NumPy 全零数组
tensor = torch.from_numpy(np_array)  # 将 NumPy 数组转换为 PyTorch 张量
np_array_back = tensor.numpy()  # 将 PyTorch 张量转换回 NumPy 数组

```

## 数学运算
```python
x = torch.tensor([1, 2, 3])  # 创建张量 x
y = torch.tensor([9, 8, 7])  # 创建张量 y

# 加法
z1 = torch.empty(3)  # 创建一个长度为 3 的未初始化张量，用于保存运算结果
torch.add(x, y, out=z1)  # 将 x 和 y 相加，并把结果写入 z1

z2 = torch.add(x, y)  # 将 x 和 y 相加，并返回一个新张量
z = x + y  # 使用加法运算符执行逐元素加法

# 减法
z = x - y  # 对 x 和 y 执行逐元素减法

# 除法
z = torch.true_divide(x, y)  # 对 x 和 y 执行逐元素除法，返回浮点数结果
# 原地操作
t = torch.zeros(3)  # 创建一个长度为 3 的全零张量
t.add_(x)  # 将 x 加到 t 上，并直接修改 t 本身，这里的_代表的是原地操作
t += x  # 将 x 加到 t 上，相当于执行 t = t + x

# 幂运算
z = x.pow(2)  # 计算 x 中每个元素的平方
z = x ** 2  # 使用幂运算符计算 x 中每个元素的平方

# 简单比较
z = x > 0  # 判断 x 中的每个元素是否大于 0，返回布尔张量
z = x < 0  # 判断 x 中的每个元素是否小于 0，返回布尔张量
#返回的类型是张量


# 矩阵乘法
x1 = torch.rand((2, 5))  # 创建一个形状为 2×5 的随机张量
x2 = torch.rand((5, 3))  # 创建一个形状为 5×3 的随机张量
x3 = torch.mm(x1, x2)  # 对 x1 和 x2 进行矩阵乘法，结果形状为 2×3
x3 = x1.mm(x2)  # 使用张量的 mm 方法进行矩阵乘法，效果与上一行相同

# 矩阵幂运算
matrix_exp = torch.rand(5, 5)  # 创建一个形状为 5×5 的随机方阵
print(matrix_exp.matrix_power(3))  # 计算并输出该矩阵的三次幂，即矩阵自身连续相乘三次

# element wise mult.  # 对应元素相乘（哈达玛积）
z = x * y   # 张量逐元素相乘操作
print(z)    # 打印逐元素相乘结果

# dot product  # 向量点积运算
z = torch.dot(x, y)   # 使用torch.dot计算一维向量内积，输出的是一个数字
print(z)              # 打印向量点积计算结果


# 批量矩阵乘法（Batch Matrix Multiplication）
batch = 32  # 一个批次中有 32 组矩阵
n = 10  # 第一个矩阵的行数
m = 20  # 第一个矩阵的列数，同时也是第二个矩阵的行数
p = 30  # 第二个矩阵的列数

tensor1 = torch.rand((batch, n, m))  # 创建 32 个形状为 10×20 的随机矩阵，整体形状为 (32, 10, 20)
tensor2 = torch.rand((batch, m, p))  # 创建 32 个形状为 20×30 的随机矩阵，整体形状为 (32, 20, 30)
out_bmm = torch.bmm(tensor1, tensor2)  # 对同一批次位置的矩阵分别相乘，输出形状为 (32, 10, 30)

# 广播机制（Broadcasting）示例
x1 = torch.rand((5, 5))  # 创建一个形状为 (5, 5) 的随机张量
x2 = torch.rand((1, 5))  # 创建一个形状为 (1, 5) 的随机张量

z = x1 - x2  # x2 的这一行会被广播到 x1 的每一行，再进行逐元素减法，结果形状为 (5, 5)
z = x1 ** x2  # x2 的这一行会被广播到 x1 的每一行，再进行逐元素幂运算，结果形状为 (5, 5)，广播时，可以把 x2 理解为被自动扩展

# 其他常用的张量操作

sum_x = torch.sum(x, dim=0)  # 沿第 0 维对 x 求和；如果 x 是二维张量，相当于按列求和

values, indices = torch.max(x, dim=0)  # 沿第 0 维寻找最大值，同时返回最大值及其索引
# values 保存每一列的最大值，indices 保存这些最大值在第 0 维中的位置

values, indices = torch.min(x, dim=0)  # 沿第 0 维寻找最小值，同时返回最小值及其索引
# values 保存每一列的最小值，indices 保存这些最小值在第 0 维中的位置

abs_x = torch.abs(x)  # 对 x 中的每个元素取绝对值，张量形状保持不变

z = torch.argmax(x, dim=0)  # 返回第 0 维中最大值所在的索引，不返回最大值本身
z = torch.argmin(x, dim=0)  # 返回第 0 维中最小值所在的索引，不返回最小值本身

mean_x = torch.mean(x.float(), dim=0)  # 先把 x 转换为 float32，再沿第 0 维计算平均值
# torch.mean 通常要求输入为浮点数或复数，因此整数张量需要先使用 float() 转换

z = torch.eq(x, y)  # 逐元素比较 x 和 y 是否相等，返回由 True 和 False 组成的布尔张量

sorted_y, indices = torch.sort(y, dim=0, descending=False)  # 沿第 0 维对 y 进行升序排列
# sorted_y 是排序后的值，indices 是这些值在原张量 y 中对应的索引
# descending=False 表示升序；改为 True 则表示降序

z = torch.clamp(x, min=0)  # 将 x 中所有小于 0 的元素限制为 0，其余元素保持不变
# 这相当于只保留非负值，其效果与 ReLU 激活函数类似

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)  # 创建布尔张量；1 转换为 True，0 转换为 False
z = torch.any(x)  # 只要 x 中至少有一个元素为 True，就返回 True
z = torch.all(x)  # 只有 x 中所有元素都为 True 才返回 True；这里包含一个 False，因此结果为 False
```


## 张量索引

```python
batch_size = 10  # 设置批次大小，表示一共有 10 个样本
features = 25  # 设置每个样本的特征数量，共有 25 个特征
x = torch.rand((batch_size, features))  # 创建形状为 (10, 25) 的随机张量，每一行代表一个样本
# x[样本索引, 特征索引]
print(x[0].shape)  # 取出第 1 个样本的所有特征，等价于 x[0, :]，结果形状为 (25,)
print(x[:, 0].shape)  # 取出所有样本的第 1 个特征，结果形状为 (10,)

print(x[2, 0:10])  # 取出第 3 个样本中索引为 0～9 的特征，不包含索引 10

x[0, 0] = 100  # 将第 1 个样本的第 1 个特征修改为 100


# 高级索引（Fancy Indexing）
x = torch.arange(10)  # 创建包含 0～9 的一维张量
indices = [2, 5, 8]  # 指定需要选取的元素索引
print(x[indices])  # 取出索引为 2、5、8 的元素，结果为 tensor([2, 5, 8])

x = torch.rand((3, 5))  # 创建一个形状为 (3, 5) 的随机张量
rows = torch.tensor([1, 0])  # 指定行索引，分别选择第 2 行和第 1 行
cols = torch.tensor([4, 0])  # 指定列索引，分别选择第 5 列和第 1 列
print(x[rows, cols].shape)  # 配对选取 x[1, 4] 和 x[0, 0]，结果形状为 (2,)

# 更高级的索引操作
x = torch.arange(10)  # 创建包含 0～9 的一维张量
print(x[(x < 2) & (x > 8)])  # 筛选同时满足“小于 2”和“大于 8”的元素，因此结果为空张量
print(x[x.remainder(2) == 0])  # 筛选除以 2 余数为 0 的元素，结果为 tensor([0, 2, 4, 6, 8])

# 常用操作
print(torch.where(x > 5, x, x * 2))  # 若元素大于 5，则保留原值，否则将该元素乘以 2。torch.where(条件, 条件为 True 时使用的值, 条件为 False 时使用的值)
print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())  # 去除重复元素，返回 tensor([0, 1, 2, 3, 4])
print(x.ndimension())  # 返回张量的维度数量；x 是一维张量，因此结果为 1
print(x.numel())  # 返回张量中元素的总数量；x 包含 10 个元素，因此结果为 10
```

## 张量重塑
```python
# 张量形状变换：view、reshape 和 contiguous

x = torch.arange(9)  # 创建包含 0～8 的一维张量，形状为 (9,)

x_3x3 = x.view(3, 3)  # 将 x 变为 3×3 的张量，不改变元素的数量和排列顺序
print(x_3x3)  # 输出变形后的 3×3 张量
x_3x3 = x.reshape(3, 3)  # 同样将 x 变为 3×3；reshape 可以在必要时自动复制数据

y = x_3x3.t()  # 对二维张量进行转置，交换行和列；元素顺序变为 [0, 3, 6, 1, 4, 7, 2, 5, 8]
print(y.contiguous().view(9))  # 先让数据在内存中连续排列，再将其展开为形状为 (9,) 的一维张量
```

`view()` 和 `reshape()` 都能改变张量的形状，但有一点区别：

- `view()` 要求张量的数据在内存中连续。
- `reshape()` 更灵活。如果数据不连续，它会在必要时创建一份副本。
- 转置后的 `y` 通常不连续，因此要先调用 `contiguous()`，再使用 `view()`。

最终展开的结果是：

```
tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])
```

```python
# 拼接张量以及自动推断维度

x1 = torch.rand((2, 5))  # 创建形状为 (2, 5) 的随机张量
x2 = torch.rand((2, 5))  # 再创建一个形状为 (2, 5) 的随机张量

print(torch.cat((x1, x2), dim=0).shape)  # 沿第 0 维纵向拼接，结果形状为 (4, 5)
print(torch.cat((x1, x2), dim=1).shape)  # 沿第 1 维横向拼接，结果形状为 (2, 10)

z = x1.view(-1)  # 将 x1 展开为一维张量，-1 表示让 PyTorch 自动计算这一维的大小
print(z.shape)  # x1 共有 2×5=10 个元素，因此结果形状为 (10,)
```

拼接时，`dim` 表示要在哪个维度上增加：

```
dim=0：按行拼接    (2, 5) + (2, 5) → (4, 5)
dim=1：按列拼接    (2, 5) + (2, 5) → (2, 10)
```

```python
# 保留批次维度进行变形，以及交换张量的维度

batch = 64  # 设置批次大小，一共有 64 个样本
x = torch.rand((batch, 2, 5))  # 创建形状为 (64, 2, 5) 的随机张量
z = x.view(batch, -1)  # 保留批次维度，将每个样本的 2×5 展开成长度为 10 的一维数据
print(z.shape)  # 输出形状为 (64, 10)

z = x.permute(0, 2, 1)  # 按照指定顺序重新排列维度，将 (64, 2, 5) 变为 (64, 5, 2)
print(z.shape)  # 输出形状为 (64, 5, 2)

x = torch.arange(10)  # 创建包含 0～9 的一维张量，形状为 (10,)
print(x.unsqueeze(0).shape)  # 在第 0 维添加一个维度，形状变为 (1, 10)
print(x.unsqueeze(1).shape)  # 在第 1 维添加一个维度，形状变为 (10, 1)
```

`permute(0, 2, 1)` 中的数字表示原维度的新排列顺序：

```
原形状：(64, 2, 5)
原维度：  0  1  2
新顺序：  0  2  1
新形状：(64, 5, 2)
```

`unsqueeze()` 用于添加一个大小为 `1` 的维度：

```
原形状：(10,)

unsqueeze(0) → (1, 10)  # 变成一行
unsqueeze(1) → (10, 1)  # 变成一列
```

```python
# 添加和移除大小为 1 的维度

x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 连续添加两个维度，形状从 (10,) 变为 (1, 1, 10)

z = x.squeeze(1)  # 删除第 1 维；因为这一维的大小为 1，所以形状变为 (1, 10)
print(z.shape)  # 输出 torch.Size([1, 10])
```

形状变化过程如下：

```
torch.arange(10)     → (10,)
unsqueeze(0)         → (1, 10)
unsqueeze(1)         → (1, 1, 10)
squeeze(1)           → (1, 10)
```

`unsqueeze(dim)` 会增加一个大小为 `1` 的维度，而 `squeeze(dim)` 只能删除大小为 `1` 的维度。如果指定维度的大小不是 `1`，张量形状不会发生变化。


# PyTorch神经网络
![504](../图片/Pasted%20image%2020260807003944.png)
## 本节目标

使用一个简单的全连接神经网络完成 MNIST 手写数字分类。整体流程如下：

1. 导入需要的库
2. 定义全连接神经网络
3. 选择 CPU 或 GPU
4. 设置超参数
5. 加载 MNIST 数据集
6. 创建损失函数与优化器
7. 训练神经网络
8. 检查模型在训练集和测试集上的准确率

MNIST 图片的尺寸为 `28×28`，每张图片属于 `0～9` 中的一个数字，因此：

```
输入特征数：28 × 28 = 784
输出类别数：10
```

---

## 2. 导入相关库

```
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.nn.functional as F  # 导入常用的神经网络函数
from torch.utils.data import DataLoader  # 用于分批加载数据
import torchvision.datasets as datasets  # 提供 MNIST 等常用数据集
import torchvision.transforms as transforms  # 用于对图像进行预处理
```

---

## 3. 创建全连接神经网络

```
class NN(nn.Module):  # 定义神经网络类，并继承 nn.Module
    def __init__(self, input_size, num_classes):  # 定义并初始化网络中的层
        super(NN, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.fc1 = nn.Linear(input_size, 50)  # 第一层：将 784 个输入特征映射为 50 个隐藏特征
        self.fc2 = nn.Linear(50, num_classes)  # 第二层：将 50 个隐藏特征映射为 10 个类别

    def forward(self, x):  # 定义数据在网络中的前向传播过程
        x = F.relu(self.fc1(x))  # 先通过第一层，再使用 ReLU 激活函数
        x = self.fc2(x)  # 通过输出层得到每个类别的分数
        return x  # 返回模型输出
```

网络结构如下：

```
输入层                  隐藏层               输出层
784 个特征  ───────→  50 个神经元  ───────→  10 个类别
                         ReLU
```

### `nn.Linear` 的作用

```
nn.Linear(in_features, out_features)
```

全连接层会执行如下计算：

```
输出 = 输入 × 权重 + 偏置
```

因此：

```
self.fc1 = nn.Linear(784, 50)
```

表示输入中每个样本有 `784` 个特征，经过第一层后得到 `50` 个特征。

### 为什么使用 ReLU？

ReLU 的计算方式为：

```
ReLU(x) = max(0, x)
```

它将负数变成 `0`，正数保持不变。激活函数可以为神经网络引入非线性，使网络能够学习更复杂的关系。

输出层后面不需要手动添加 `Softmax`，因为后面使用的 `CrossEntropyLoss` 已经包含了相关计算。

---

## 4. 选择运行设备

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果 CUDA 可用就使用 GPU，否则使用 CPU
```

这里使用了 Python 的条件表达式：

```
如果 CUDA 可用 → device = "cuda"
否则          → device = "cpu"
```

后面需要将模型和数据都移动到同一个设备，否则会产生设备不一致的错误。

---

## 5. 设置超参数

```
input_size = 784  # 每张 MNIST 图片包含 28×28=784 个像素
num_classes = 10  # MNIST 一共有 0～9，共 10 个类别
learning_rate = 0.001  # 设置学习率，控制每次更新模型参数的幅度
batch_size = 64  # 每次使用 64 张图片训练模型
num_epochs = 1  # 将整个训练集完整训练 1 遍
```

---

## 6. 加载 MNIST 数据集

```
train_dataset = datasets.MNIST(
    root="dataset/",  # 数据集保存的位置
    train=True,  # 加载训练集
    transform=transforms.ToTensor(),  # 将图片转换为 PyTorch 张量
    download=True,  # 如果本地不存在数据集，则自动下载
)

train_loader = DataLoader(
    dataset=train_dataset,  # 指定需要加载的训练数据集
    batch_size=batch_size,  # 每个批次包含 64 个样本
    shuffle=True,  # 每轮训练前打乱训练数据
)

test_dataset = datasets.MNIST(
    root="dataset/",  # 数据集保存的位置
    train=False,  # 加载测试集
    transform=transforms.ToTensor(),  # 将图片转换为 PyTorch 张量
    download=True,  # 如果本地不存在数据集，则自动下载
)

test_loader = DataLoader(
    dataset=test_dataset,  # 指定需要加载的测试数据集
    batch_size=batch_size,  # 每个批次包含 64 个样本
    shuffle=False,  # 测试时通常不需要打乱数据
)
```

`transforms.ToTensor()` 主要完成两件事：

1. 将图片转换成 PyTorch 张量。
2. 将像素值从 `0～255` 缩放到 `0～1`。

从 `DataLoader` 中取出的数据形状通常为：

```
图片 data：  (batch_size, 1, 28, 28)
标签 targets：(batch_size,)
```

其中：

- `batch_size`：当前批次的图片数量。
- `1`：图片的通道数，MNIST 是灰度图。
- `28, 28`：图片的高度和宽度。

---

## 7. 初始化网络

```
model = NN(
    input_size=input_size,  # 输入特征数量为 784
    num_classes=num_classes,  # 输出类别数量为 10
).to(device)  # 将模型移动到 CPU 或 GPU
```

`.to(device)` 会将模型的参数移动到指定设备。

如果模型位于 GPU，那么训练数据也必须移动到 GPU：

```
data = data.to(device)
targets = targets.to(device)
```

---

## 8. 定义损失函数和优化器

```
criterion = nn.CrossEntropyLoss()  # 创建交叉熵损失函数，用于多分类任务
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用 Adam 更新模型中的参数
```

### 交叉熵损失函数

`CrossEntropyLoss` 常用于多分类任务，用来衡量模型预测结果与正确标签之间的差距。

```
loss = criterion(scores, targets)
```

其中：

```
scores 的形状：(batch_size, 10)
targets 的形状：(batch_size,)
```

`scores` 中每一行包含一个样本对 `10` 个类别的预测分数。

### Adam 优化器

```
optim.Adam(model.parameters(), lr=learning_rate)
```

- `model.parameters()`：告诉优化器需要更新哪些参数。
- `lr`：指定学习率。
- `optimizer.step()`：根据计算出的梯度更新参数。

---

## 9. 训练神经网络

```
for epoch in range(num_epochs):  # 循环训练指定的轮数
    for batch_idx, (data, targets) in enumerate(train_loader):  # 分批读取图片和正确标签
        data = data.to(device=device)  # 将图片移动到模型所在的设备
        targets = targets.to(device=device)  # 将标签移动到模型所在的设备

        data = data.reshape(data.shape[0], -1)  # 将每张 28×28 的图片展开为长度为 784 的一维向量

        scores = model(data)  # 前向传播，得到每张图片对 10 个类别的预测分数
        loss = criterion(scores, targets)  # 根据预测分数和正确标签计算损失

        optimizer.zero_grad()  # 清除上一个批次留下的梯度
        loss.backward()  # 反向传播，计算每个参数对应的梯度
        optimizer.step()  # 使用 Adam 优化器更新模型参数
```

### 图片为什么需要展开？

模型中的第一层是：

```
nn.Linear(784, 50)
```

它要求每个样本的输入形状为 `(784,)`，但原始图片的形状为：

```
(batch_size, 1, 28, 28)
```

因此需要执行：

```
data = data.reshape(data.shape[0], -1)
```

形状变化如下：

```
(batch_size, 1, 28, 28) → (batch_size, 784)
```

其中：

- `data.shape[0]` 保留当前批次的样本数量。
- `-1` 让 PyTorch 自动计算剩余维度的大小，即 `1×28×28=784`。

### 每个批次的训练步骤

```
读取数据
   ↓
将数据移动到设备
   ↓
展开图片
   ↓
前向传播
   ↓
计算损失
   ↓
梯度清零
   ↓
反向传播
   ↓
更新参数
```

### 为什么要调用 `optimizer.zero_grad()`？

PyTorch 默认会累加梯度。如果不清零，当前批次的梯度就会与之前批次的梯度相加。

因此，每次反向传播前都需要执行：

```
optimizer.zero_grad()
```

---

## 10. 检查模型准确率

下面对课程代码做了一个小修正：截图中的函数最后写了 `return acc`，但前面没有创建 `acc`，直接运行会出现 `NameError`。这里补充了准确率的计算。

```
def check_accuracy(loader, model):  # 定义检查模型准确率的函数
    if loader.dataset.train:  # 判断当前加载的是训练集还是测试集
        print("Checking accuracy on training data")  # 提示正在检查训练集
    else:
        print("Checking accuracy on test data")  # 提示正在检查测试集

    num_correct = 0  # 用于累计预测正确的样本数量
    num_samples = 0  # 用于累计检查过的样本总数
    model.eval()  # 将模型切换到评估模式

    with torch.no_grad():  # 关闭梯度计算，减少内存占用并提高评估速度
        for x, y in loader:  # 分批读取图片 x 和正确标签 y
            x = x.to(device=device)  # 将图片移动到模型所在设备
            y = y.to(device=device)  # 将标签移动到模型所在设备
            x = x.reshape(x.shape[0], -1)  # 将图片从 28×28 展开成长度为 784 的向量

            scores = model(x)  # 使用模型计算每个类别的预测分数
            _, predictions = scores.max(dim=1)  # 取每个样本分数最高的类别作为预测结果
            num_correct += (predictions == y).sum()  # 累加预测正确的样本数量
            num_samples += predictions.size(0)  # 累加当前批次中的样本数量

    accuracy = float(num_correct) / float(num_samples) * 100  # 计算百分比形式的准确率
    print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}")  # 输出正确数量、总数和准确率

    model.train()  # 检查结束后，将模型重新切换到训练模式
    return accuracy  # 返回计算出的准确率
```

调用函数：

```
check_accuracy(train_loader, model)  # 检查模型在训练集上的准确率
check_accuracy(test_loader, model)  # 检查模型在测试集上的准确率
```

### `model.eval()` 和 `model.train()`

```
model.eval()
```

将模型切换到评估模式。它会影响 Dropout 和 Batch Normalization 等层的行为。

```
model.train()
```

将模型切换回训练模式。

虽然这个简单网络没有使用 Dropout 和 Batch Normalization，但养成正确切换模式的习惯很重要。

### `torch.no_grad()`

测试模型时不需要进行反向传播，因此可以关闭梯度计算：

```
with torch.no_grad():
```

这样可以：

- 减少内存占用
- 加快计算速度
- 避免保存不需要的计算图

### 获取预测类别

```
_, predictions = scores.max(dim=1)
```

`scores` 的形状为：

```
(batch_size, 10)
```

`dim=1` 表示在每个样本的 `10` 个类别分数中寻找最大值：

- 第一个返回值是最大分数，这里使用 `_` 表示不需要它。
- 第二个返回值是最大分数所在的索引，也就是模型预测的类别。

---

## 11. 完整代码

```
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.nn.functional as F  # 导入常用神经网络函数
from torch.utils.data import DataLoader  # 导入数据加载器
import torchvision.datasets as datasets  # 导入常用数据集
import torchvision.transforms as transforms  # 导入数据预处理工具


class NN(nn.Module):  # 创建全连接神经网络
    def __init__(self, input_size, num_classes):  # 初始化网络结构
        super(NN, self).__init__()  # 初始化 nn.Module 父类
        self.fc1 = nn.Linear(input_size, 50)  # 将 784 个输入特征映射为 50 个隐藏特征
        self.fc2 = nn.Linear(50, num_classes)  # 将 50 个隐藏特征映射为 10 个类别

    def forward(self, x):  # 定义前向传播
        x = F.relu(self.fc1(x))  # 经过第一层和 ReLU 激活函数
        x = self.fc2(x)  # 经过输出层，得到类别分数
        return x  # 返回类别分数


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择 GPU 或 CPU

input_size = 784  # MNIST 图片展开后的特征数量
num_classes = 10  # 分类数量
learning_rate = 0.001  # 学习率
batch_size = 64  # 每批样本数量
num_epochs = 1  # 训练轮数

train_dataset = datasets.MNIST(
    root="dataset/",  # 数据集保存目录
    train=True,  # 加载训练集
    transform=transforms.ToTensor(),  # 将图片转换为张量
    download=True,  # 必要时自动下载
)

train_loader = DataLoader(
    dataset=train_dataset,  # 指定训练数据集
    batch_size=batch_size,  # 每批加载 64 个样本
    shuffle=True,  # 打乱训练数据
)

test_dataset = datasets.MNIST(
    root="dataset/",  # 数据集保存目录
    train=False,  # 加载测试集
    transform=transforms.ToTensor(),  # 将图片转换为张量
    download=True,  # 必要时自动下载
)

test_loader = DataLoader(
    dataset=test_dataset,  # 指定测试数据集
    batch_size=batch_size,  # 每批加载 64 个样本
    shuffle=False,  # 测试数据不需要打乱
)

model = NN(input_size=input_size, num_classes=num_classes).to(device)  # 创建模型并移动到指定设备

criterion = nn.CrossEntropyLoss()  # 创建交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 创建 Adam 优化器

for epoch in range(num_epochs):  # 按照指定轮数进行训练
    for batch_idx, (data, targets) in enumerate(train_loader):  # 分批读取数据
        data = data.to(device=device)  # 将图片移动到指定设备
        targets = targets.to(device=device)  # 将标签移动到指定设备

        data = data.reshape(data.shape[0], -1)  # 将图片展开为长度为 784 的向量

        scores = model(data)  # 前向传播
        loss = criterion(scores, targets)  # 计算损失

        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播并计算梯度
        optimizer.step()  # 更新模型参数


def check_accuracy(loader, model):  # 定义准确率检查函数
    if loader.dataset.train:  # 判断是否为训练集
        print("Checking accuracy on training data")  # 输出训练集提示
    else:
        print("Checking accuracy on test data")  # 输出测试集提示

    num_correct = 0  # 记录预测正确的数量
    num_samples = 0  # 记录样本总数
    model.eval()  # 切换到评估模式

    with torch.no_grad():  # 关闭梯度计算
        for x, y in loader:  # 分批读取图片和标签
            x = x.to(device=device)  # 将图片移动到指定设备
            y = y.to(device=device)  # 将标签移动到指定设备
            x = x.reshape(x.shape[0], -1)  # 将图片展开为一维向量

            scores = model(x)  # 获取预测分数
            _, predictions = scores.max(dim=1)  # 获取分数最高的类别
            num_correct += (predictions == y).sum()  # 累加预测正确的数量
            num_samples += predictions.size(0)  # 累加样本数量

    accuracy = float(num_correct) / float(num_samples) * 100  # 计算准确率
    print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}")  # 输出准确率

    model.train()  # 恢复训练模式
    return accuracy  # 返回准确率


check_accuracy(train_loader, model)  # 检查训练集准确率
check_accuracy(test_loader, model)  # 检查测试集准确率
```

---

## 12. 本节重点总结

### 网络结构

```
MNIST 图片
(1, 28, 28)
     ↓ reshape
(784,)
     ↓ Linear
(50,)
     ↓ ReLU
(50,)
     ↓ Linear
(10,)
     ↓ 取最大分数的索引
预测类别 0～9
```

### 一次训练的核心代码

```
scores = model(data)  # 前向传播
loss = criterion(scores, targets)  # 计算损失

optimizer.zero_grad()  # 梯度清零
loss.backward()  # 反向传播
optimizer.step()  # 更新参数
```

可以将其记忆为：

```
预测 → 计算损失 → 梯度清零 → 反向传播 → 更新参数
```

### 需要特别注意的地方

1. 模型和数据必须位于同一个设备。
2. MNIST 图片需要从 `(1, 28, 28)` 展开为 `(784,)`。
3. 使用 `CrossEntropyLoss` 时，输出层后不需要手动添加 Softmax。
4. 每次反向传播前需要使用 `optimizer.zero_grad()` 清除旧梯度。
5. 测试时应使用 `model.eval()` 和 `torch.no_grad()`。
6. 训练结束后，测试集准确率比训练集准确率更能反映模型对新数据的分类能力。


# 使用 PyTorch 实现卷积神经网络（CNN）

## 1. 本节目标

使用卷积神经网络对 MNIST 手写数字进行分类。

上一节使用全连接网络时，需要直接把图片展开为长度为 `784` 的向量。这样会破坏图片原本的空间结构。

卷积神经网络可以直接接收形状为 `(通道数, 高度, 宽度)` 的图片，并提取边缘、纹理和形状等局部特征。

本节重点：

- `nn.Conv2d` 二维卷积层
- 卷积核、步长和填充
- 特征图尺寸的计算
- `nn.MaxPool2d` 最大池化
- CNN 中的数据形状变化
- 卷积层与全连接层的连接
- 使用 CNN 完成 MNIST 分类

---

## 2. CNN 的基本结构

一个简单的 CNN 通常由以下部分组成：

```
输入图片
   ↓
卷积层
   ↓
激活函数
   ↓
池化层
   ↓
卷积层
   ↓
激活函数
   ↓
池化层
   ↓
展平
   ↓
全连接层
   ↓
分类结果
```

本节使用的网络结构为：

```
输入：(1, 28, 28)
   ↓
Conv2d：1 → 16
   ↓
ReLU
   ↓
MaxPool2d
   ↓
Conv2d：16 → 32
   ↓
ReLU
   ↓
MaxPool2d
   ↓
展平：32×7×7 = 1568
   ↓
Linear：1568 → 128
   ↓
ReLU
   ↓
Linear：128 → 10
```

---

## 3. 二维卷积层

PyTorch 中可以使用 `nn.Conv2d` 创建二维卷积层：

```
nn.Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
)
```

主要参数：

- `in_channels`：输入图片或特征图的通道数。
- `out_channels`：卷积核数量，也是输出特征图的通道数。
- `kernel_size`：卷积核大小。
- `stride`：卷积核每次移动的距离。
- `padding`：在图片边缘填充的像素数量。

对于 MNIST 灰度图片：

```
输入通道数 = 1
```

如果输入的是普通 RGB 彩色图片：

```
输入通道数 = 3
```

### 输出通道数

```
nn.Conv2d(1, 16, kernel_size=3)
```

这里有 `16` 个卷积核，每个卷积核会生成一张特征图，所以输出通道数为 `16`。

不同卷积核可以学习不同特征，例如：

- 水平边缘
- 垂直边缘
- 曲线
- 纹理
- 更复杂的局部形状

---

## 4. 卷积输出尺寸

卷积层输出特征图的高度或宽度可以通过下面的公式计算：

```
输出尺寸 = floor((输入尺寸 + 2×padding - kernel_size) / stride) + 1
```

例如：

```
输入尺寸 = 28
kernel_size = 3
stride = 1
padding = 1
```

代入公式：

```
输出尺寸 = (28 + 2×1 - 3) / 1 + 1
         = 28
```

因此，当卷积核为 `3×3`、步长为 `1`、填充为 `1` 时，卷积前后的高度和宽度不变：

```
(1, 28, 28) → (16, 28, 28)
```

---

## 5. 最大池化

使用 `nn.MaxPool2d` 创建二维最大池化层：

```
nn.MaxPool2d(kernel_size=2, stride=2)
```

它会在每个 `2×2` 区域中保留最大值：

```
1  5
2  3
```

经过最大池化后得到：

```
5
```

当 `kernel_size=2`、`stride=2` 时，特征图的高度和宽度都会减半：

```
(16, 28, 28) → (16, 14, 14)
```

最大池化的作用包括：

- 减小特征图尺寸
- 减少后续计算量
- 保留较明显的特征
- 提高模型对轻微位置变化的适应能力

池化只改变高度和宽度，不改变通道数。

---

## 6. 定义卷积神经网络

```
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入常用神经网络函数


class CNN(nn.Module):  # 定义卷积神经网络
    def __init__(self, num_classes=10):  # 初始化网络中的各个层
        super(CNN, self).__init__()  # 初始化 nn.Module 父类

        self.conv1 = nn.Conv2d(
            in_channels=1,  # MNIST 是灰度图，因此输入通道数为 1
            out_channels=16,  # 使用 16 个卷积核，输出 16 张特征图
            kernel_size=3,  # 使用 3×3 的卷积核
            stride=1,  # 卷积核每次移动 1 个像素
            padding=1,  # 在边缘填充 1 圈，使高度和宽度保持不变
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,  # 接收上一层输出的 16 个通道
            out_channels=32,  # 使用 32 个卷积核，输出 32 张特征图
            kernel_size=3,  # 使用 3×3 的卷积核
            stride=1,  # 卷积核每次移动 1 个像素
            padding=1,  # 保持特征图的高度和宽度不变
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,  # 在每个 2×2 区域中选取最大值
            stride=2,  # 每次移动 2 个像素，使高度和宽度减半
        )

        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 将卷积特征映射为 128 个隐藏特征
        self.fc2 = nn.Linear(128, num_classes)  # 输出 10 个类别的预测分数

    def forward(self, x):  # 定义前向传播过程
        x = self.pool(F.relu(self.conv1(x)))  # 第一次卷积、激活和池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二次卷积、激活和池化

        x = torch.flatten(x, start_dim=1)  # 保留批次维度，将其余维度全部展开

        x = F.relu(self.fc1(x))  # 经过第一个全连接层和 ReLU
        x = self.fc2(x)  # 输出每个类别的预测分数
        return x  # 返回预测结果
```

---

## 7. 网络中的形状变化

假设一个批次中有 `64` 张图片，那么输入形状为：

```
(64, 1, 28, 28)
```

PyTorch 图像张量的维度顺序为：

```
(batch_size, channels, height, width)
```

经过网络后，形状变化如下：

```
输入
(64, 1, 28, 28)

↓ conv1：通道数从 1 变成 16

(64, 16, 28, 28)

↓ ReLU：只修改数值，不改变形状

(64, 16, 28, 28)

↓ pool：高度和宽度减半

(64, 16, 14, 14)

↓ conv2：通道数从 16 变成 32

(64, 32, 14, 14)

↓ ReLU：不改变形状

(64, 32, 14, 14)

↓ pool：高度和宽度再次减半

(64, 32, 7, 7)

↓ flatten：保留批次维度，展开其余维度

(64, 1568)

↓ fc1

(64, 128)

↓ fc2

(64, 10)
```

这里的 `1568` 来自：

```
32 × 7 × 7 = 1568
```

因此，全连接层必须写成：

```
self.fc1 = nn.Linear(32 * 7 * 7, 128)
```

如果这里计算错误，运行时会出现矩阵形状不匹配的错误。

---

## 8. `torch.flatten` 的作用

```
x = torch.flatten(x, start_dim=1)
```

假设卷积部分的输出形状为：

```
(64, 32, 7, 7)
```

`start_dim=1` 表示：

- 保留第 `0` 维，也就是批次维度。
- 从第 `1` 维开始，将后面的维度全部展开。

结果为：

```
(64, 32, 7, 7) → (64, 1568)
```

也可以使用：

```
x = x.reshape(x.shape[0], -1)  # 保留批次维度，将其他维度自动展开
```

两种写法在这里具有相同效果。

---

## 9. 完整训练代码

```
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.nn.functional as F  # 导入常用神经网络函数
from torch.utils.data import DataLoader  # 导入数据加载器
import torchvision.datasets as datasets  # 导入常用数据集
import torchvision.transforms as transforms  # 导入图像预处理工具


class CNN(nn.Module):  # 定义卷积神经网络
    def __init__(self, num_classes=10):  # 初始化网络结构
        super(CNN, self).__init__()  # 初始化父类

        self.conv1 = nn.Conv2d(
            in_channels=1,  # 输入为单通道灰度图
            out_channels=16,  # 输出 16 个通道
            kernel_size=3,  # 卷积核大小为 3×3
            stride=1,  # 步长为 1
            padding=1,  # 填充为 1，保持图片大小不变
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,  # 输入通道数为 16
            out_channels=32,  # 输出通道数为 32
            kernel_size=3,  # 卷积核大小为 3×3
            stride=1,  # 步长为 1
            padding=1,  # 保持特征图尺寸不变
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,  # 池化窗口大小为 2×2
            stride=2,  # 步长为 2，使高度和宽度减半
        )

        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 将卷积特征映射到 128 个隐藏特征
        self.fc2 = nn.Linear(128, num_classes)  # 将隐藏特征映射到 10 个类别

    def forward(self, x):  # 定义前向传播
        x = self.conv1(x)  # 第一次卷积，形状变为 (batch, 16, 28, 28)
        x = F.relu(x)  # 使用 ReLU 激活函数
        x = self.pool(x)  # 第一次池化，形状变为 (batch, 16, 14, 14)

        x = self.conv2(x)  # 第二次卷积，形状变为 (batch, 32, 14, 14)
        x = F.relu(x)  # 使用 ReLU 激活函数
        x = self.pool(x)  # 第二次池化，形状变为 (batch, 32, 7, 7)

        x = torch.flatten(x, start_dim=1)  # 展开为 (batch, 1568)
        x = F.relu(self.fc1(x))  # 经过第一个全连接层，形状变为 (batch, 128)
        x = self.fc2(x)  # 生成 10 个类别分数，形状变为 (batch, 10)
        return x  # 返回类别分数


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择运行设备

learning_rate = 0.001  # 设置学习率
batch_size = 64  # 设置批次大小
num_epochs = 3  # 设置训练轮数
num_classes = 10  # MNIST 一共有 10 个类别

transform = transforms.ToTensor()  # 将 MNIST 图片转换为张量

train_dataset = datasets.MNIST(
    root="dataset/",  # 指定数据集保存目录
    train=True,  # 加载训练集
    transform=transform,  # 应用图片转换
    download=True,  # 必要时自动下载数据集
)

test_dataset = datasets.MNIST(
    root="dataset/",  # 指定数据集保存目录
    train=False,  # 加载测试集
    transform=transform,  # 应用图片转换
    download=True,  # 必要时自动下载数据集
)

train_loader = DataLoader(
    dataset=train_dataset,  # 指定训练集
    batch_size=batch_size,  # 每批加载 64 张图片
    shuffle=True,  # 打乱训练数据
)

test_loader = DataLoader(
    dataset=test_dataset,  # 指定测试集
    batch_size=batch_size,  # 每批加载 64 张图片
    shuffle=False,  # 测试数据不需要打乱
)

model = CNN(num_classes=num_classes).to(device)  # 创建 CNN 并移动到指定设备
criterion = nn.CrossEntropyLoss()  # 创建多分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 创建 Adam 优化器


for epoch in range(num_epochs):  # 循环训练多个轮次
    model.train()  # 将模型设置为训练模式
    total_loss = 0  # 记录当前轮次的总损失

    for batch_idx, (data, targets) in enumerate(train_loader):  # 分批读取训练数据
        data = data.to(device)  # 将图片移动到指定设备
        targets = targets.to(device)  # 将标签移动到指定设备

        scores = model(data)  # 前向传播；CNN 可以直接接收四维图片张量
        loss = criterion(scores, targets)  # 计算预测结果与正确标签之间的损失

        optimizer.zero_grad()  # 清除上一批数据产生的梯度
        loss.backward()  # 反向传播并计算梯度
        optimizer.step()  # 根据梯度更新模型参数

        total_loss += loss.item()  # 将当前批次的损失累加到总损失中

    average_loss = total_loss / len(train_loader)  # 计算当前轮次的平均损失
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}"
    )  # 输出当前轮次和平均损失


def check_accuracy(loader, model):  # 定义准确率检查函数
    num_correct = 0  # 记录预测正确的样本数量
    num_samples = 0  # 记录样本总数
    model.eval()  # 将模型切换到评估模式

    with torch.no_grad():  # 关闭梯度计算
        for data, targets in loader:  # 分批读取图片和标签
            data = data.to(device)  # 将图片移动到指定设备
            targets = targets.to(device)  # 将标签移动到指定设备

            scores = model(data)  # 使用 CNN 计算每个类别的分数
            predictions = scores.argmax(dim=1)  # 取分数最高的类别作为预测结果

            num_correct += (predictions == targets).sum().item()  # 累加预测正确的数量
            num_samples += targets.size(0)  # 累加样本总数

    accuracy = num_correct / num_samples * 100  # 计算准确率
    model.train()  # 将模型恢复为训练模式
    return accuracy  # 返回准确率


train_accuracy = check_accuracy(train_loader, model)  # 计算训练集准确率
test_accuracy = check_accuracy(test_loader, model)  # 计算测试集准确率

print(f"Accuracy on training data: {train_accuracy:.2f}%")  # 输出训练集准确率
print(f"Accuracy on test data: {test_accuracy:.2f}%")  # 输出测试集准确率
```

---

## 10. CNN 与上一节全连接网络的区别

### 全连接网络

全连接网络需要先展开图片：

```
data = data.reshape(data.shape[0], -1)  # (batch, 1, 28, 28) → (batch, 784)
scores = model(data)  # 将展开后的数据送入全连接网络
```

它直接处理所有像素，无法显式保留像素之间的位置关系。

### 卷积神经网络

CNN 直接接收原始图片形状：

```
scores = model(data)  # 输入形状为 (batch, 1, 28, 28)
```

不要在送入卷积层之前将图片展开，因为 `nn.Conv2d` 需要四维输入：

```
(batch_size, channels, height, width)
```

只有完成卷积和池化后，才需要将特征图展开并送入全连接层。

---

## 11. 卷积层在学习什么？

卷积核的初始参数通常是随机的。在训练过程中，反向传播会自动更新卷积核参数。

较浅的卷积层通常会学习：

- 边缘
- 方向
- 简单线条
- 局部纹理

较深的卷积层会组合前面提取的特征，从而学习：

- 数字的弯曲部分
- 笔画组合
- 更完整的数字结构

因此，第二个卷积层的输入不是原始图片，而是第一个卷积层提取出的特征图：

```
原始图片
   ↓
简单特征：边缘、线条
   ↓
复杂特征：笔画组合、数字形状
   ↓
分类结果
```

---

## 12. 卷积层的参数是怎样组织的？

第一层卷积为：

```
self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
```

其权重形状为：

```
(16, 1, 3, 3)
```

含义是：

```
16 个卷积核
每个卷积核接收 1 个输入通道
每个卷积核的大小为 3×3
```

第二层卷积为：

```
self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
```

其权重形状为：

```
(32, 16, 3, 3)
```

含义是：

```
32 个输出卷积核
每个卷积核都需要处理前一层的 16 个输入通道
每个卷积核的大小为 3×3
```

可以查看实际权重形状：

```
print(model.conv1.weight.shape)  # torch.Size([16, 1, 3, 3])
print(model.conv2.weight.shape)  # torch.Size([32, 16, 3, 3])
```

---

## 13. 一种更简洁的模型写法

连续的网络层可以使用 `nn.Sequential` 组合起来：

```
class SimpleCNN(nn.Module):  # 定义更简洁的卷积神经网络
    def __init__(self, num_classes=10):  # 初始化网络结构
        super(SimpleCNN, self).__init__()  # 初始化父类

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (batch, 1, 28, 28) → (batch, 16, 28, 28)
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # → (batch, 16, 14, 14)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → (batch, 32, 14, 14)
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # → (batch, 32, 7, 7)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # → (batch, 1568)
            nn.Linear(32 * 7 * 7, 128),  # → (batch, 128)
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.Linear(128, num_classes),  # → (batch, 10)
        )

    def forward(self, x):  # 定义前向传播
        x = self.features(x)  # 使用卷积部分提取图片特征
        x = self.classifier(x)  # 使用全连接部分完成分类
        return x  # 返回类别分数
```

`nn.Sequential` 会按照定义顺序依次执行其中的层，适合结构比较简单、数据流只有一条路径的网络。

---

## 14. 常见错误

### 错误一：提前展开图片

错误写法：

```
data = data.reshape(data.shape[0], -1)  # 将图片变成了二维张量
scores = model(data)  # Conv2d 无法按预期处理这种输入
```

正确写法：

```
scores = model(data)  # 直接传入形状为 (batch, 1, 28, 28) 的图片
```

### 错误二：输入通道数不正确

MNIST 是灰度图片，因此第一层应该写成：

```
nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
```

对于 RGB 图片，第一层通常应该写成：

```
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
```

### 错误三：全连接层输入数量计算错误

如果卷积部分最终输出：

```
(batch, 32, 7, 7)
```

全连接层输入数量必须是：

```
self.fc1 = nn.Linear(32 * 7 * 7, 128)
```

不能只填写通道数 `32`。

### 错误四：忘记池化会改变尺寸

卷积层可能保持高度和宽度不变，但池化层通常会减小尺寸：

```
28 → 14 → 7
```

定义全连接层之前，需要把所有卷积和池化产生的尺寸变化计算清楚。

---

## 15. 本节重点总结

### CNN 的核心组成

```
nn.Conv2d(...)  # 提取局部特征
F.relu(...)  # 引入非线性
nn.MaxPool2d(...)  # 缩小特征图
torch.flatten(...)  # 将特征图展开
nn.Linear(...)  # 根据提取出的特征完成分类
```

### 本节网络中的形状变化

```
(1, 28, 28)
    ↓ Conv2d
(16, 28, 28)
    ↓ MaxPool2d
(16, 14, 14)
    ↓ Conv2d
(32, 14, 14)
    ↓ MaxPool2d
(32, 7, 7)
    ↓ Flatten
(1568,)
    ↓ Linear
(128,)
    ↓ Linear
(10,)
```

### CNN 与全连接网络最关键的区别

```
全连接网络：先展开图片，再进行特征学习
卷积神经网络：先保留图片结构提取特征，最后再展开分类
```

### 最重要的形状规则

```
Conv2d 输入：(batch_size, channels, height, width)
Linear 输入：(batch_size, features)
```

所以 CNN 中的数据处理顺序是：

```
四维图片张量
    ↓ 卷积和池化
四维特征图
    ↓ flatten
二维特征张量
    ↓ 全连接层
分类结果
```


# 使用 PyTorch 实现循环神经网络（RNN）

## 1. 本节目标

使用循环神经网络对 MNIST 手写数字进行分类。

虽然 RNN 更常用于文本、语音和时间序列，但也可以把一张 MNIST 图片看成一个序列：

```
一张图片的形状：(1, 28, 28)

序列长度：28
每个时间步的输入特征数：28
```

也就是把图片的每一行看作一个时间步：

```
第 1 个时间步  → 图片第 1 行的 28 个像素
第 2 个时间步  → 图片第 2 行的 28 个像素
...
第 28 个时间步 → 图片第 28 行的 28 个像素
```

本节重点：

- 序列数据的表示方式
- RNN 的基本原理
- 隐藏状态 `hidden state`
- `nn.RNN` 的输入与输出
- 多层 RNN
- 使用最后一个时间步完成分类
- RNN 中的数据形状变化

---

## 2. RNN 的基本思想

普通全连接网络的每次输入相互独立，而 RNN 在处理当前输入时，还会使用上一个时间步产生的隐藏状态。

可以简单表示为：

```
当前隐藏状态 = RNN(当前输入, 上一个隐藏状态)
```

序列处理过程如下：

```
x₁ → RNN → h₁
           ↓
x₂ → RNN → h₂
           ↓
x₃ → RNN → h₃
           ↓
          ...
           ↓
x₂₈ → RNN → h₂₈ → 全连接层 → 分类结果
```

其中：

- `xₜ`：第 `t` 个时间步的输入。
- `hₜ`：处理到第 `t` 个时间步时的隐藏状态。
- 隐藏状态可以理解为 RNN 对之前序列信息的记忆。

对于 MNIST 分类，我们只需要为整张图片输出一个类别，因此这是一个“多对一”的任务：

```
多个时间步的输入 → 一个分类结果
```

---

## 3. RNN 中的重要参数

PyTorch 中可以使用 `nn.RNN` 创建循环神经网络：

```
nn.RNN(
    input_size=28,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
)
```

参数含义：

- `input_size`：每个时间步输入的特征数量。
- `hidden_size`：隐藏状态包含的特征数量。
- `num_layers`：堆叠的 RNN 层数。
- `batch_first=True`：规定输入张量的第一个维度是批次维度。

在 MNIST 中：

```
sequence_length = 28
input_size = 28
```

它们虽然都是 `28`，但含义不同：

```
sequence_length：一共有多少个时间步，也就是图片有多少行
input_size：每个时间步包含多少个特征，也就是每行有多少个像素
```

---

## 4. RNN 的输入形状

当设置：

```
batch_first=True
```

RNN 的输入形状为：

```
(batch_size, sequence_length, input_size)
```

对于 MNIST：

```
(batch_size, 28, 28)
```

但从 `DataLoader` 读取的 MNIST 图片形状为：

```
(batch_size, 1, 28, 28)
```

其中的 `1` 是灰度图的通道数。因为 RNN 需要三维输入，所以要移除这个大小为 `1` 的通道维度：

```
data = data.squeeze(1)
```

形状变化如下：

```
(batch_size, 1, 28, 28)
              ↓ squeeze(1)
(batch_size, 28, 28)
```

---

## 5. 隐藏状态

RNN 在开始处理序列之前，需要一个初始隐藏状态：

```
h0 = torch.zeros(num_layers, batch_size, hidden_size)
```

隐藏状态的形状固定为：

```
(num_layers, batch_size, hidden_size)
```

需要注意，即使设置了 `batch_first=True`，隐藏状态的形状也不会变成以批次维度开头。

例如：

```
num_layers = 2
batch_size = 64
hidden_size = 256

h0 的形状：(2, 64, 256)
```

一般将初始隐藏状态设置为全零张量。

---

## 6. 定义 RNN 模型

```
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块


class RNN(nn.Module):  # 定义循环神经网络
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
    ):
        super(RNN, self).__init__()  # 初始化 nn.Module 父类

        self.hidden_size = hidden_size  # 保存隐藏状态的特征数量
        self.num_layers = num_layers  # 保存 RNN 的层数

        self.rnn = nn.RNN(
            input_size=input_size,  # 每个时间步输入 28 个像素
            hidden_size=hidden_size,  # 每个时间步输出 256 个隐藏特征
            num_layers=num_layers,  # 堆叠 2 层 RNN
            batch_first=True,  # 输入形状使用 (batch, sequence, feature)
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes,
        )  # 将最后一个时间步的隐藏特征映射为 10 个类别

    def forward(self, x):  # 定义前向传播
        batch_size = x.size(0)  # 获取当前批次中的样本数量

        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=x.device,
        )  # 创建与输入位于同一设备的初始隐藏状态

        out, hidden = self.rnn(x, h0)  # 处理完整序列，返回所有时间步的输出和最终隐藏状态

        out = out[:, -1, :]  # 取每个样本最后一个时间步的输出
        out = self.fc(out)  # 将最后一个时间步的特征映射为 10 个类别分数
        return out  # 返回分类结果
```

---

## 7. `nn.RNN` 返回的两个结果

执行：

```
out, hidden = self.rnn(x, h0)
```

会得到 `out` 和 `hidden`。

### `out`

当 `batch_first=True` 时：

```
out 的形状：
(batch_size, sequence_length, hidden_size)
```

例如：

```
(64, 28, 256)
```

它保存最后一层 RNN 在所有时间步产生的输出：

```
out[:, 0, :]   → 第 1 个时间步的输出
out[:, 1, :]   → 第 2 个时间步的输出
...
out[:, -1, :]  → 最后一个时间步的输出
```

在图片分类中，我们需要综合整张图片的信息，因此使用最后一个时间步：

```
out = out[:, -1, :]
```

形状变化：

```
(64, 28, 256) → (64, 256)
```

随后送入全连接层：

```
(64, 256) → (64, 10)
```

### `hidden`

```
hidden 的形状：
(num_layers, batch_size, hidden_size)
```

例如：

```
(2, 64, 256)
```

`hidden` 保存每一层 RNN 最后一个时间步的隐藏状态：

```
hidden[0]  → 第 1 层的最终隐藏状态
hidden[1]  → 第 2 层的最终隐藏状态
```

如果只需要最后一层的最终隐藏状态，也可以写成：

```
out = hidden[-1]  # 取最后一层的最终隐藏状态，形状为 (batch_size, hidden_size)
out = self.fc(out)  # 将其送入全连接层进行分类
```

在普通的单向 RNN 中，下面两种写法通常可以达到相同目的：

```
out[:, -1, :]  # 最后一层在最后一个时间步的输出
hidden[-1]  # 最后一层最终的隐藏状态
```

---

## 8. 多层 RNN

当设置：

```
num_layers = 2
```

表示堆叠两个 RNN 层：

```
输入序列
   ↓
第 1 层 RNN
   ↓
第 2 层 RNN
   ↓
输出序列
```

第一层会把每个时间步的输出交给第二层：

```
第 1 层的输出序列 → 第 2 层的输入序列
```

增加层数可以提高模型的表达能力，但同时也会：

- 增加计算量
- 增加参数数量
- 提高过拟合风险
- 让训练变得更加困难

---

## 9. 完整训练代码

```
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
from torch.utils.data import DataLoader  # 导入数据加载器
import torchvision.datasets as datasets  # 导入常用数据集
import torchvision.transforms as transforms  # 导入图像预处理工具


class RNN(nn.Module):  # 定义循环神经网络
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
    ):
        super(RNN, self).__init__()  # 初始化父类

        self.hidden_size = hidden_size  # 保存隐藏状态大小
        self.num_layers = num_layers  # 保存 RNN 层数

        self.rnn = nn.RNN(
            input_size=input_size,  # 每个时间步输入 28 个像素
            hidden_size=hidden_size,  # 隐藏状态包含 256 个特征
            num_layers=num_layers,  # 使用两层 RNN
            batch_first=True,  # 输入采用 (batch, sequence, feature) 顺序
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes,
        )  # 将最后的隐藏特征映射为 10 个类别

    def forward(self, x):  # 定义前向传播
        batch_size = x.size(0)  # 获取当前批次大小

        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=x.device,
        )  # 创建初始隐藏状态，形状为 (num_layers, batch, hidden_size)

        out, hidden = self.rnn(x, h0)  # 让 RNN 依次处理 28 个时间步
        out = out[:, -1, :]  # 取最后一个时间步的输出，形状为 (batch, hidden_size)
        out = self.fc(out)  # 得到 10 个类别的预测分数
        return out  # 返回分类结果


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择运行设备

sequence_length = 28  # 每张图片包含 28 个时间步
input_size = 28  # 每个时间步输入 28 个像素
hidden_size = 256  # 隐藏状态包含 256 个特征
num_layers = 2  # 使用两层 RNN
num_classes = 10  # MNIST 包含 10 个类别
learning_rate = 0.001  # 设置学习率
batch_size = 64  # 设置批次大小
num_epochs = 3  # 设置训练轮数

transform = transforms.ToTensor()  # 将图片转换为张量

train_dataset = datasets.MNIST(
    root="dataset/",  # 设置数据集保存目录
    train=True,  # 加载训练集
    transform=transform,  # 将图片转换为张量
    download=True,  # 必要时自动下载
)

test_dataset = datasets.MNIST(
    root="dataset/",  # 设置数据集保存目录
    train=False,  # 加载测试集
    transform=transform,  # 将图片转换为张量
    download=True,  # 必要时自动下载
)

train_loader = DataLoader(
    dataset=train_dataset,  # 指定训练数据集
    batch_size=batch_size,  # 每个批次包含 64 个样本
    shuffle=True,  # 打乱训练数据
)

test_loader = DataLoader(
    dataset=test_dataset,  # 指定测试数据集
    batch_size=batch_size,  # 每个批次包含 64 个样本
    shuffle=False,  # 测试时不需要打乱数据
)

model = RNN(
    input_size=input_size,  # 每个时间步的输入特征数
    hidden_size=hidden_size,  # 隐藏状态大小
    num_layers=num_layers,  # RNN 层数
    num_classes=num_classes,  # 分类数量
).to(device)  # 创建模型并移动到指定设备

criterion = nn.CrossEntropyLoss()  # 创建交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 创建 Adam 优化器


for epoch in range(num_epochs):  # 循环训练多个轮次
    model.train()  # 将模型设置为训练模式
    total_loss = 0  # 记录当前轮次的总损失

    for batch_idx, (data, targets) in enumerate(train_loader):  # 分批读取数据
        data = data.to(device)  # 将图片移动到指定设备
        targets = targets.to(device)  # 将标签移动到指定设备

        data = data.squeeze(1)  # 删除灰度通道维度：(batch, 1, 28, 28) → (batch, 28, 28)

        scores = model(data)  # 前向传播，得到形状为 (batch, 10) 的类别分数
        loss = criterion(scores, targets)  # 计算分类损失

        optimizer.zero_grad()  # 清除上一次反向传播产生的梯度
        loss.backward()  # 通过时间反向传播并计算梯度
        optimizer.step()  # 更新 RNN 和全连接层的参数

        total_loss += loss.item()  # 累加当前批次的损失

    average_loss = total_loss / len(train_loader)  # 计算当前轮次的平均损失
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}"
    )  # 输出当前轮次和损失


def check_accuracy(loader, model):  # 定义准确率检查函数
    num_correct = 0  # 记录预测正确的样本数量
    num_samples = 0  # 记录样本总数
    model.eval()  # 将模型切换到评估模式

    with torch.no_grad():  # 评估时关闭梯度计算
        for data, targets in loader:  # 分批读取图片和标签
            data = data.to(device)  # 将图片移动到指定设备
            targets = targets.to(device)  # 将标签移动到指定设备

            data = data.squeeze(1)  # 将图片变为 (batch, sequence_length, input_size)

            scores = model(data)  # 使用 RNN 计算类别分数
            predictions = scores.argmax(dim=1)  # 选择分数最高的类别

            num_correct += (predictions == targets).sum().item()  # 累加预测正确的数量
            num_samples += targets.size(0)  # 累加样本数量

    accuracy = num_correct / num_samples * 100  # 计算百分比准确率
    model.train()  # 将模型恢复为训练模式
    return accuracy  # 返回准确率


train_accuracy = check_accuracy(train_loader, model)  # 计算训练集准确率
test_accuracy = check_accuracy(test_loader, model)  # 计算测试集准确率

print(f"Accuracy on training data: {train_accuracy:.2f}%")  # 输出训练集准确率
print(f"Accuracy on test data: {test_accuracy:.2f}%")  # 输出测试集准确率
```

---

## 10. RNN 中的完整形状变化

假设：

```
batch_size = 64
sequence_length = 28
input_size = 28
hidden_size = 256
num_layers = 2
num_classes = 10
```

形状变化为：

```
DataLoader 输出图片
(64, 1, 28, 28)

↓ squeeze(1)

RNN 输入
(64, 28, 28)

↓ nn.RNN

out
(64, 28, 256)

hidden
(2, 64, 256)

↓ out[:, -1, :]

最后一个时间步
(64, 256)

↓ Linear(256, 10)

分类结果
(64, 10)
```

最重要的三个形状是：

```
RNN 输入：    (batch_size, sequence_length, input_size)
RNN 输出：    (batch_size, sequence_length, hidden_size)
最终隐藏状态： (num_layers, batch_size, hidden_size)
```

---

## 11. `batch_first` 的作用

如果使用：

```
self.rnn = nn.RNN(
    input_size=28,
    hidden_size=256,
    batch_first=True,
)
```

输入形状为：

```
(batch_size, sequence_length, input_size)
```

如果不设置 `batch_first=True`，默认输入形状为：

```
(sequence_length, batch_size, input_size)
```

这时需要先调整维度：

```
data = data.permute(1, 0, 2)  # (batch, sequence, feature) → (sequence, batch, feature)
```

使用 `batch_first=True` 通常更容易与 `DataLoader` 的输出形式配合。

但是要再次注意：

```
batch_first 只影响输入 x 和输出 out
不会改变隐藏状态 hidden 的维度顺序
```

隐藏状态仍然是：

```
(num_layers, batch_size, hidden_size)
```

---

## 12. RNN 如何进行反向传播？

RNN 在不同时间步共享同一组参数。

训练时，模型不仅需要计算输出层的梯度，还要让梯度沿时间步反向传播：

```
损失
 ↓
h₂₈
 ↓
h₂₇
 ↓
h₂₆
 ↓
...
 ↓
h₁
```

这种过程称为：

```
Backpropagation Through Time
通过时间反向传播，简称 BPTT
```

代码仍然只需要调用：

```
loss.backward()  # PyTorch 会自动完成通过时间反向传播
```

---

## 13. RNN 的梯度问题

普通 RNN 在处理较长序列时容易出现两个问题：

### 梯度消失

梯度经过许多时间步后越来越小，模型难以学习相隔较远的信息。

例如：

```
序列开头的重要信息 ─────────→ 序列结尾
                         信息可能逐渐丢失
```

### 梯度爆炸

梯度经过多个时间步后变得非常大，导致训练不稳定，甚至使损失变为 `NaN`。

可以使用梯度裁剪限制梯度大小：

```
optimizer.zero_grad()  # 清除旧梯度
loss.backward()  # 计算梯度

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)  # 将模型梯度的整体范数限制在 1.0 以内

optimizer.step()  # 更新模型参数
```

对于较长序列，通常会使用 LSTM 或 GRU 来缓解普通 RNN 的长期依赖问题。

---

## 14. RNN 与 CNN 的区别

### CNN

CNN 主要关注图片中的局部空间特征：

```
边缘 → 纹理 → 局部形状 → 完整结构
```

图片各行可以同时参与卷积计算，并不需要严格按顺序处理。

### RNN

RNN 按时间步依次读取数据：

```
第 1 行 → 第 2 行 → 第 3 行 → ... → 第 28 行
```

每个时间步都会使用之前的隐藏状态，因此能够处理序列关系。

对于图像分类，CNN 通常比普通 RNN 更合适；但将 MNIST 用作 RNN 示例，可以帮助理解序列输入、隐藏状态和时间步的概念。

---

## 15. 不同类型的序列任务

### 多对一

输入多个时间步，最终只产生一个结果：

```
一段文字 → 情感分类
一段时间序列 → 类别预测
28 行图片 → 数字类别
```

本节 MNIST 分类属于多对一任务。

### 一对多

输入一个信息，生成多个时间步的输出：

```
一张图片 → 一段文字描述
```

### 多对多

多个输入时间步对应多个输出时间步：

```
英文句子 → 中文句子
每个时间点的数据 → 每个时间点的标签
```

不同任务需要选择不同的 RNN 输出，而不一定总是使用：

```
out[:, -1, :]
```

---

## 16. 更简洁的模型写法

下面使用最终隐藏状态进行分类：

```
class SimpleRNN(nn.Module):  # 定义简化版 RNN
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()  # 初始化父类

        self.hidden_size = hidden_size  # 保存隐藏状态大小
        self.num_layers = num_layers  # 保存 RNN 层数

        self.rnn = nn.RNN(
            input_size=input_size,  # 每个时间步的输入特征数量
            hidden_size=hidden_size,  # 隐藏状态大小
            num_layers=num_layers,  # RNN 层数
            batch_first=True,  # 批次维度放在最前面
        )

        self.fc = nn.Linear(hidden_size, num_classes)  # 创建分类层

    def forward(self, x):  # 定义前向传播
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device,
        )  # 创建初始隐藏状态

        _, hidden = self.rnn(x, h0)  # 只保留最终隐藏状态，不使用所有时间步的输出
        last_hidden = hidden[-1]  # 取最后一层的最终隐藏状态
        scores = self.fc(last_hidden)  # 将最终隐藏状态映射为类别分数
        return scores  # 返回分类结果
```

这里使用 `_` 忽略 `out`：

```
_, hidden = self.rnn(x, h0)
```

因为模型只需要最后一层的最终隐藏状态：

```
last_hidden = hidden[-1]
```

---

## 17. 常见错误

### 错误一：将图片完全展开

错误写法：

```
data = data.reshape(data.shape[0], -1)  # 结果为 (batch, 784)
scores = model(data)  # RNN 需要三维序列输入
```

正确写法：

```
data = data.squeeze(1)  # 结果为 (batch, 28, 28)
scores = model(data)
```

### 错误二：初始隐藏状态形状错误

错误写法：

```
h0 = torch.zeros(batch_size, num_layers, hidden_size)
```

正确写法：

```
h0 = torch.zeros(num_layers, batch_size, hidden_size)
```

### 错误三：隐藏状态和输入不在同一设备

错误写法：

```
h0 = torch.zeros(num_layers, batch_size, hidden_size)  # 默认创建在 CPU
```

如果输入位于 GPU，就会出现设备不一致错误。

正确写法：

```
h0 = torch.zeros(
    num_layers,
    batch_size,
    hidden_size,
    device=x.device,
)
```

### 错误四：取错了维度

当 `batch_first=True` 时：

```
out[:, -1, :]  # 所有样本、最后一个时间步、所有隐藏特征
```

不能写成：

```
out[-1, :, :]  # 这会取最后一个样本，而不是最后一个时间步
```

### 错误五：把 `hidden` 当作普通二维张量

对于多层 RNN：

```
hidden 的形状为 (num_layers, batch_size, hidden_size)
```

应该使用：

```
hidden[-1]  # 取最后一层的隐藏状态
```

---

## 18. 本节重点总结

### MNIST 如何转换成序列？

```
原始图片：(1, 28, 28)

去掉通道维度：

序列：(28, 28)
       ↑   ↑
   28 个时间步
   每步 28 个特征
```

### RNN 的核心代码

```
h0 = torch.zeros(
    num_layers,
    x.size(0),
    hidden_size,
    device=x.device,
)  # 创建初始隐藏状态

out, hidden = rnn(x, h0)  # 处理整个序列
last_output = out[:, -1, :]  # 取最后一个时间步
scores = fc(last_output)  # 完成分类
```

### 需要记住的形状

```
输入 x：
(batch_size, sequence_length, input_size)

初始隐藏状态 h0：
(num_layers, batch_size, hidden_size)

所有时间步的输出 out：
(batch_size, sequence_length, hidden_size)

最终隐藏状态 hidden：
(num_layers, batch_size, hidden_size)

分类结果 scores：
(batch_size, num_classes)
```

### RNN 分类流程

```
MNIST 图片
(1, 28, 28)
    ↓ squeeze
序列
(28, 28)
    ↓ RNN
所有时间步的隐藏特征
(28, hidden_size)
    ↓ 取最后一个时间步
(hidden_size,)
    ↓ Linear
(10,)
    ↓ argmax
预测数字
```


# 使用 PyTorch 实现 LSTM

## 1. LSTM 是什么？

LSTM（Long Short-Term Memory，长短期记忆网络）是 RNN 的一种改进版本。

普通 RNN 在处理长序列时容易出现：

- 梯度消失
- 梯度爆炸
- 难以记住较早时间步的信息

LSTM 通过额外的记忆单元和门控机制控制信息的保留与遗忘，因此更适合学习长期依赖关系。

```
普通 RNN：只有隐藏状态 h
LSTM：隐藏状态 h + 细胞状态 c
```

---

## 2. LSTM 的核心结构

LSTM 主要包含三个门：

- 遗忘门：决定丢弃多少旧信息。
- 输入门：决定保存多少新信息。
- 输出门：决定当前时间步输出什么信息。

可以简单理解为：

```
遗忘门：哪些旧信息不需要了？
输入门：哪些新信息需要记住？
输出门：当前应该输出哪些信息？
```

其中，细胞状态 `c` 负责在不同时间步之间传递长期信息。

```
上一个细胞状态 cₜ₋₁
          ↓
    遗忘旧信息
          ↓
    加入新信息
          ↓
当前细胞状态 cₜ
```

---

## 3. 创建 LSTM 层

```
self.lstm = nn.LSTM(
    input_size=input_size,  # 每个时间步的输入特征数量
    hidden_size=hidden_size,  # 隐藏状态和细胞状态的特征数量
    num_layers=num_layers,  # 堆叠的 LSTM 层数
    batch_first=True,  # 输入形状为 (batch, sequence, feature)
)
```

LSTM 输入形状与 RNN 相同：

```
(batch_size, sequence_length, input_size)
```

对于 MNIST：

```
(batch_size, 28, 28)
```

---

## 4. LSTM 与 RNN 的主要区别

普通 RNN 只需要一个初始隐藏状态：

```
out, hidden = self.rnn(x, h0)
```

LSTM 需要同时提供：

```
out, (hidden, cell) = self.lstm(x, (h0, c0))
```

其中：

- `h0`：初始隐藏状态，表示短期信息。
- `c0`：初始细胞状态，表示长期记忆。
- `hidden`：所有 LSTM 层最终的隐藏状态。
- `cell`：所有 LSTM 层最终的细胞状态。

`h0` 和 `c0` 的形状相同：

```
(num_layers, batch_size, hidden_size)
```

---

## 5. 定义 LSTM 模型

```
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块


class LSTM(nn.Module):  # 定义 LSTM 分类模型
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
    ):
        super(LSTM, self).__init__()  # 初始化 nn.Module 父类

        self.hidden_size = hidden_size  # 保存隐藏状态的大小
        self.num_layers = num_layers  # 保存 LSTM 的层数

        self.lstm = nn.LSTM(
            input_size=input_size,  # 每个时间步输入 28 个特征
            hidden_size=hidden_size,  # 隐藏状态包含 256 个特征
            num_layers=num_layers,  # 使用两层 LSTM
            batch_first=True,  # 输入形状为 (batch, sequence, feature)
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes,
        )  # 将最后一个时间步的输出映射为 10 个类别

    def forward(self, x):  # 定义前向传播
        batch_size = x.size(0)  # 获取当前批次大小

        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=x.device,
        )  # 创建初始隐藏状态

        c0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=x.device,
        )  # 创建初始细胞状态

        out, (hidden, cell) = self.lstm(x, (h0, c0))  # 使用 LSTM 处理整个输入序列
        out = out[:, -1, :]  # 取每个样本最后一个时间步的输出
        out = self.fc(out)  # 将最后的序列特征映射为类别分数
        return out  # 返回分类结果
```

---

## 6. LSTM 的输出形状

假设：

```
batch_size = 64
sequence_length = 28
input_size = 28
hidden_size = 256
num_layers = 2
```

执行：

```
out, (hidden, cell) = self.lstm(x, (h0, c0))
```

各个张量的形状为：

```
输入 x：
(64, 28, 28)

所有时间步的输出 out：
(64, 28, 256)

最终隐藏状态 hidden：
(2, 64, 256)

最终细胞状态 cell：
(2, 64, 256)
```

进行分类时，取最后一个时间步：

```
out = out[:, -1, :]  # (64, 28, 256) → (64, 256)
out = self.fc(out)  # (64, 256) → (64, 10)
```

---

## 7. 完整训练代码

数据加载、训练和准确率检查与 RNN 基本相同，主要修改模型部分。

```
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
from torch.utils.data import DataLoader  # 导入数据加载器
import torchvision.datasets as datasets  # 导入 MNIST 数据集
import torchvision.transforms as transforms  # 导入图像转换工具


class LSTM(nn.Module):  # 定义 LSTM 模型
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()  # 初始化父类

        self.hidden_size = hidden_size  # 保存隐藏状态大小
        self.num_layers = num_layers  # 保存 LSTM 层数

        self.lstm = nn.LSTM(
            input_size=input_size,  # 每个时间步输入 28 个像素
            hidden_size=hidden_size,  # 隐藏状态大小为 256
            num_layers=num_layers,  # 使用两层 LSTM
            batch_first=True,  # 批次维度放在第一维
        )

        self.fc = nn.Linear(hidden_size, num_classes)  # 创建分类层

    def forward(self, x):  # 定义前向传播
        batch_size = x.size(0)  # 获取当前批次大小

        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=x.device,
        )  # 初始化隐藏状态

        c0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=x.device,
        )  # 初始化细胞状态

        out, _ = self.lstm(x, (h0, c0))  # 处理完整序列，这里不使用最终状态
        out = out[:, -1, :]  # 取最后一个时间步的输出
        scores = self.fc(out)  # 得到 10 个类别分数
        return scores  # 返回分类结果


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择运行设备

sequence_length = 28  # 图片被看作包含 28 个时间步
input_size = 28  # 每个时间步包含 28 个像素
hidden_size = 256  # 隐藏状态大小
num_layers = 2  # LSTM 层数
num_classes = 10  # 分类数量
learning_rate = 0.001  # 设置学习率
batch_size = 64  # 设置批次大小
num_epochs = 3  # 设置训练轮数

train_dataset = datasets.MNIST(
    root="dataset/",  # 设置数据集保存目录
    train=True,  # 加载训练集
    transform=transforms.ToTensor(),  # 将图片转换为张量
    download=True,  # 必要时自动下载
)

test_dataset = datasets.MNIST(
    root="dataset/",  # 设置数据集保存目录
    train=False,  # 加载测试集
    transform=transforms.ToTensor(),  # 将图片转换为张量
    download=True,  # 必要时自动下载
)

train_loader = DataLoader(
    dataset=train_dataset,  # 指定训练数据集
    batch_size=batch_size,  # 设置批次大小
    shuffle=True,  # 打乱训练数据
)

test_loader = DataLoader(
    dataset=test_dataset,  # 指定测试数据集
    batch_size=batch_size,  # 设置批次大小
    shuffle=False,  # 测试数据不需要打乱
)

model = LSTM(
    input_size=input_size,  # 每个时间步的输入大小
    hidden_size=hidden_size,  # 隐藏状态大小
    num_layers=num_layers,  # LSTM 层数
    num_classes=num_classes,  # 输出类别数量
).to(device)  # 创建模型并移动到指定设备

criterion = nn.CrossEntropyLoss()  # 创建交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 创建 Adam 优化器


for epoch in range(num_epochs):  # 循环训练多个轮次
    model.train()  # 切换到训练模式
    total_loss = 0  # 记录当前轮次的总损失

    for data, targets in train_loader:  # 分批读取训练数据
        data = data.to(device)  # 将图片移动到指定设备
        targets = targets.to(device)  # 将标签移动到指定设备

        data = data.squeeze(1)  # (batch, 1, 28, 28) → (batch, 28, 28)

        scores = model(data)  # 使用 LSTM 进行前向传播
        loss = criterion(scores, targets)  # 计算损失

        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        total_loss += loss.item()  # 累加当前批次的损失

    average_loss = total_loss / len(train_loader)  # 计算平均损失
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}"
    )  # 输出当前训练情况


def check_accuracy(loader, model):  # 定义准确率检查函数
    num_correct = 0  # 记录预测正确的样本数量
    num_samples = 0  # 记录样本总数
    model.eval()  # 切换到评估模式

    with torch.no_grad():  # 关闭梯度计算
        for data, targets in loader:  # 分批读取数据
            data = data.to(device)  # 将图片移动到指定设备
            targets = targets.to(device)  # 将标签移动到指定设备
            data = data.squeeze(1)  # 将图片转换为序列形式

            scores = model(data)  # 获取类别分数
            predictions = scores.argmax(dim=1)  # 获取预测类别

            num_correct += (predictions == targets).sum().item()  # 累加正确数量
            num_samples += targets.size(0)  # 累加样本数量

    accuracy = num_correct / num_samples * 100  # 计算准确率
    model.train()  # 恢复训练模式
    return accuracy  # 返回准确率


train_accuracy = check_accuracy(train_loader, model)  # 计算训练集准确率
test_accuracy = check_accuracy(test_loader, model)  # 计算测试集准确率

print(f"Accuracy on training data: {train_accuracy:.2f}%")  # 输出训练集准确率
print(f"Accuracy on test data: {test_accuracy:.2f}%")  # 输出测试集准确率
```

---

## 8. 使用最终隐藏状态分类

除了使用最后一个时间步的 `out`：

```
out, (hidden, cell) = self.lstm(x, (h0, c0))
scores = self.fc(out[:, -1, :])
```

也可以直接使用最后一层的最终隐藏状态：

```
out, (hidden, cell) = self.lstm(x, (h0, c0))  # 获取 LSTM 的输出和最终状态
last_hidden = hidden[-1]  # 取最后一层的最终隐藏状态
scores = self.fc(last_hidden)  # 使用最终隐藏状态进行分类
```

对于这里使用的单向 LSTM，两种方法通常可以达到相同目的。

---

## 9. RNN 与 LSTM 的代码差异

### 普通 RNN

```
self.rnn = nn.RNN(
    input_size,
    hidden_size,
    num_layers,
    batch_first=True,
)

h0 = torch.zeros(num_layers, batch_size, hidden_size, device=x.device)
out, hidden = self.rnn(x, h0)
```

### LSTM

```
self.lstm = nn.LSTM(
    input_size,
    hidden_size,
    num_layers,
    batch_first=True,
)

h0 = torch.zeros(num_layers, batch_size, hidden_size, device=x.device)
c0 = torch.zeros(num_layers, batch_size, hidden_size, device=x.device)
out, (hidden, cell) = self.lstm(x, (h0, c0))
```

最核心的区别：

```
RNN：  输入 h0，返回 hidden
LSTM：输入 (h0, c0)，返回 (hidden, cell)
```

---

## 10. 本节重点总结

### LSTM 的两个状态

```
hidden state（h）：短期信息和当前输出
cell state（c）：用于传递长期记忆
```

### LSTM 的核心代码

```
h0 = torch.zeros(
    num_layers,
    x.size(0),
    hidden_size,
    device=x.device,
)  # 初始化隐藏状态

c0 = torch.zeros(
    num_layers,
    x.size(0),
    hidden_size,
    device=x.device,
)  # 初始化细胞状态

out, (hidden, cell) = lstm(x, (h0, c0))  # 处理输入序列
last_output = out[:, -1, :]  # 取最后一个时间步
scores = fc(last_output)  # 得到分类结果
```

### 需要记住的形状

```
输入 x：
(batch_size, sequence_length, input_size)

h0 和 c0：
(num_layers, batch_size, hidden_size)

out：
(batch_size, sequence_length, hidden_size)

hidden 和 cell：
(num_layers, batch_size, hidden_size)
```

### 总结

```
LSTM 在普通 RNN 的基础上增加了细胞状态和门控机制，
使模型能够更好地保留长期信息，并缓解梯度消失问题。
```



# PyTorch 模型的保存与加载

## 1. 两种保存方式

PyTorch 中常见的模型保存方式有两种：

```
方式一：只保存模型参数 state_dict（推荐）
方式二：保存整个模型
```

通常推荐第一种，因为文件更小，也更加灵活。

---

## 2. 使用 `state_dict` 保存模型

`state_dict` 是一个 Python 字典，其中保存了模型各层的参数，例如权重和偏置。

```
torch.save(model.state_dict(), "model.pth")  # 将模型参数保存到 model.pth 文件
```

可以查看模型参数：

```
print(model.state_dict())  # 查看模型中每一层的参数名称及对应数值
```

保存后的文件只包含参数，不包含模型的类和网络结构。

---

## 3. 加载模型参数

加载模型参数时，必须先创建一个结构完全相同的模型：

```
model = CNN(num_classes=10)  # 先创建与保存时结构相同的模型
model.load_state_dict(torch.load("model.pth"))  # 加载保存的模型参数
model.eval()  # 将模型切换到评估模式
```

完整写法：

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择运行设备

model = CNN(num_classes=10).to(device)  # 创建模型并移动到指定设备

state_dict = torch.load(
    "model.pth",
    map_location=device,
)  # 加载参数，并将参数映射到当前设备

model.load_state_dict(state_dict)  # 将参数放入模型
model.eval()  # 切换到评估模式
```

`map_location=device` 可以避免设备不一致问题。例如，模型原本在 GPU 上保存，但现在需要在 CPU 上加载。

---

## 4. 使用加载后的模型预测

```
model.eval()  # 将模型切换到评估模式

with torch.no_grad():  # 预测时关闭梯度计算
    data = data.to(device)  # 将输入数据移动到模型所在设备
    scores = model(data)  # 使用加载后的模型进行预测
    predictions = scores.argmax(dim=1)  # 获取预测类别
```

加载模型后进行测试或预测时，应调用：

```
model.eval()
```

---

## 5. 保存整个模型

也可以直接保存整个模型：

```
torch.save(model, "complete_model.pth")  # 保存模型结构和模型参数
```

加载整个模型：

```
model = torch.load(
    "complete_model.pth",
    map_location=device,
    weights_only=False,
)  # 加载整个模型

model.eval()  # 切换到评估模式
```

这种方法虽然代码简单，但依赖原来的类定义和代码结构。如果模型类的位置发生改变，加载时可能失败。

因此，通常更推荐保存：

```
model.state_dict()
```

---

## 6. 保存训练检查点

如果之后还想继续训练，除了模型参数，还需要保存优化器状态、当前轮次和损失等信息。

```
checkpoint = {
    "epoch": epoch,  # 保存当前训练轮次
    "model_state_dict": model.state_dict(),  # 保存模型参数
    "optimizer_state_dict": optimizer.state_dict(),  # 保存优化器状态
    "loss": loss.item(),  # 保存当前损失
}

torch.save(checkpoint, "checkpoint.pth")  # 保存训练检查点
```

优化器状态中包含 Adam 的动量等信息。如果只保存模型参数，虽然也可以继续训练，但优化器之前积累的状态会丢失。

---

## 7. 加载训练检查点

```
model = CNN(num_classes=10).to(device)  # 创建相同结构的模型
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)  # 创建与训练时相同的优化器

checkpoint = torch.load(
    "checkpoint.pth",
    map_location=device,
)  # 加载检查点

model.load_state_dict(
    checkpoint["model_state_dict"]
)  # 恢复模型参数

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)  # 恢复优化器状态

start_epoch = checkpoint["epoch"] + 1  # 从下一个轮次继续训练
previous_loss = checkpoint["loss"]  # 获取保存时的损失

model.train()  # 继续训练前切换到训练模式
```

之后可以从保存的轮次继续训练：

```
for epoch in range(start_epoch, num_epochs):  # 从上次结束的位置继续训练
    for data, targets in train_loader:  # 分批读取训练数据
        data = data.to(device)  # 移动输入数据
        targets = targets.to(device)  # 移动标签数据

        scores = model(data)  # 前向传播
        loss = criterion(scores, targets)  # 计算损失

        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
```

---

## 8. 文件后缀

PyTorch 模型文件常用的后缀包括：

```
.pth
.pt
```

例如：

```
torch.save(model.state_dict(), "mnist_cnn.pth")
torch.save(checkpoint, "mnist_checkpoint.pt")
```

两种后缀没有本质区别，只是命名习惯不同。

---

## 9. 常见使用场景

### 只保存最终模型，用于预测

```
torch.save(model.state_dict(), "model.pth")  # 只保存模型参数
```

加载时：

```
model = CNN(num_classes=10).to(device)  # 创建相同结构的模型
model.load_state_dict(
    torch.load("model.pth", map_location=device)
)  # 加载模型参数
model.eval()  # 切换到评估模式
```

### 保存训练进度，之后继续训练

```
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
    },
    "checkpoint.pth",
)  # 保存完整训练检查点
```

---

## 10. 重点总结

### 推荐的模型保存方式

```
torch.save(model.state_dict(), "model.pth")  # 保存模型参数
```

### 推荐的模型加载方式

```
model = CNN(num_classes=10).to(device)  # 先创建相同结构的模型

model.load_state_dict(
    torch.load("model.pth", map_location=device)
)  # 加载模型参数

model.eval()  # 设置为评估模式
```

### 如果需要继续训练

同时保存：

```
模型参数
优化器状态
当前训练轮次
当前损失
```

### `train()` 与 `eval()`

```
model.train()  # 继续训练时使用
model.eval()  # 测试或预测时使用
```

### 总结

```
只用于预测：保存 model.state_dict()
需要继续训练：保存 checkpoint
加载后预测：调用 model.eval()
加载后训练：调用 model.train()
```


# PyTorch 迁移学习与微调

## 1. 什么是迁移学习？

迁移学习（Transfer Learning）是指使用一个已经在大型数据集上训练好的模型，再将它应用到自己的任务中。

例如，ResNet-18 已经在 ImageNet 数据集上学习了大量通用图像特征：

```
浅层网络：学习边缘、颜色、纹理
深层网络：学习形状和更复杂的结构
分类层：根据特征完成具体分类
```

当我们只有少量数据时，通常不需要从头训练整个模型，而是：

```
加载预训练模型
      ↓
替换最后的分类层
      ↓
在自己的数据集上训练
```

这种方法通常训练更快，也更容易取得较好的效果。PyTorch 官方主要介绍了两种方式：固定特征提取和完整微调。[PyTorch 官方迁移学习教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial)

---

## 2. 两种常见方式

### 固定特征提取

冻结预训练模型的主体，只训练新添加的分类层：

```
预训练特征提取部分：冻结
新的分类层：训练
```

适合：

- 自己的数据量较少
- 新数据与 ImageNet 图片比较相似
- 希望减少训练时间

### Fine-tuning 微调

使用预训练权重初始化模型，然后继续训练模型的部分或全部参数：

```
预训练特征提取部分：继续训练
新的分类层：训练
```

适合：

- 自己的数据量相对充足
- 新数据与 ImageNet 存在一定差异
- 希望模型进一步适应新任务

通常可以先训练分类层，再用较小的学习率微调整个网络。

---

## 3. 数据集目录结构

可以使用 `ImageFolder` 加载自己的分类数据集：

```
dataset/
├── train/
│   ├── cats/
│   │   ├── cat1.jpg
│   │   └── cat2.jpg
│   └── dogs/
│       ├── dog1.jpg
│       └── dog2.jpg
└── val/
    ├── cats/
    │   └── cat3.jpg
    └── dogs/
        └── dog3.jpg
```

每个子文件夹的名称会自动成为类别名称。

---

## 4. 加载并预处理数据

预训练 ResNet 使用 ImageNet 图片进行训练，因此输入图片也应该使用相同的标准化参数：

```
import torch  # 导入 PyTorch
from torch.utils.data import DataLoader  # 导入数据加载器
from torchvision import datasets, transforms  # 导入数据集和图像转换工具


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪并缩放为 224×224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转，进行数据增强
    transforms.ToTensor(),  # 将图片转换为张量
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # 使用 ImageNet 的均值
        std=[0.229, 0.224, 0.225],  # 使用 ImageNet 的标准差
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),  # 将图片较短的一边缩放为 256
    transforms.CenterCrop(224),  # 从图片中心裁剪出 224×224
    transforms.ToTensor(),  # 将图片转换为张量
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # 使用 ImageNet 的均值
        std=[0.229, 0.224, 0.225],  # 使用 ImageNet 的标准差
    ),
])

train_dataset = datasets.ImageFolder(
    root="dataset/train",  # 指定训练集目录
    transform=train_transform,  # 使用包含数据增强的转换
)

val_dataset = datasets.ImageFolder(
    root="dataset/val",  # 指定验证集目录
    transform=val_transform,  # 使用验证集转换
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)

class_names = train_dataset.classes  # 获取所有类别名称
num_classes = len(class_names)  # 获取类别数量

print(class_names)  # 例如输出 ['cats', 'dogs']
```

训练集使用随机裁剪和翻转，是为了增加数据的多样性；验证集不应该使用随机增强。

---

## 5. 加载预训练模型

下面使用预训练的 ResNet-18：

```
import torch.nn as nn  # 导入神经网络模块
from torchvision.models import resnet18, ResNet18_Weights  # 导入 ResNet-18 和预训练权重


weights = ResNet18_Weights.DEFAULT  # 选择推荐的 ImageNet 预训练权重
model = resnet18(weights=weights)  # 创建 ResNet-18 并加载预训练参数
```

当前 `torchvision` 推荐通过 `weights` 参数加载预训练权重。[Torchvision ResNet-18 文档](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18)

旧版本中可能看到：

```
model = resnet18(pretrained=True)  # 旧版本写法
```

新代码推荐使用：

```
model = resnet18(weights=ResNet18_Weights.DEFAULT)  # 推荐写法
```

---

## 6. 替换分类层

原始 ResNet-18 用于识别 ImageNet 的 `1000` 个类别，因此最后一层不能直接用于自己的任务。

先获取原分类层的输入特征数：

```
num_features = model.fc.in_features  # 获取最后一层接收的特征数量
```

再替换成适合自己类别数量的新分类层：

```
model.fc = nn.Linear(
    num_features,
    num_classes,
)  # 将原来的 1000 类输出替换为自己的类别数量
```

例如二分类任务的变化为：

```
原分类层：Linear(512, 1000)
新分类层：Linear(512, 2)
```

新创建的分类层使用随机参数，需要在自己的数据集上进行训练。

---

## 7. 方法一：固定特征提取

首先冻结预训练模型的全部参数：

```
model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)  # 加载预训练的 ResNet-18

for param in model.parameters():
    param.requires_grad = False  # 冻结已有参数，不再计算它们的梯度
```

然后替换分类层：

```
num_features = model.fc.in_features  # 获取原分类层的输入特征数
model.fc = nn.Linear(
    num_features,
    num_classes,
)  # 创建新的分类层
```

新创建的 `model.fc` 默认满足：

```
requires_grad = True
```

因此只有新分类层会被训练。

优化器也只需要接收分类层参数：

```
import torch.optim as optim  # 导入优化器模块


optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001,
)  # 只更新最后的分类层
```

完整的核心代码：

```
model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)  # 加载预训练模型

for param in model.parameters():
    param.requires_grad = False  # 冻结整个预训练模型

num_features = model.fc.in_features  # 获取分类层的输入特征数
model.fc = nn.Linear(num_features, num_classes)  # 替换分类层

model = model.to(device)  # 将模型移动到指定设备

criterion = nn.CrossEntropyLoss()  # 创建交叉熵损失函数
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001,
)  # 只训练新分类层
```

---

## 8. 方法二：微调整个模型

如果希望所有参数都适应自己的任务，就不需要冻结参数：

```
model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)  # 加载预训练模型

num_features = model.fc.in_features  # 获取原分类层的输入特征数
model.fc = nn.Linear(num_features, num_classes)  # 替换分类层

model = model.to(device)  # 将模型移动到指定设备

criterion = nn.CrossEntropyLoss()  # 创建损失函数
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,
)  # 使用较小学习率微调整个模型
```

微调时通常使用较小的学习率，因为预训练参数已经比较好。如果学习率过大，可能会快速破坏模型已经学到的有用特征。

---

## 9. 推荐策略：先冻结，再解冻

实际使用时，可以分成两个阶段。

### 第一阶段：只训练分类层

```
for param in model.parameters():
    param.requires_grad = False  # 冻结所有预训练参数

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes,
)  # 创建新的分类层

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001,
)  # 使用相对较大的学习率训练分类层
```

先训练几个 epoch，让新的分类层能够基本完成分类。

### 第二阶段：解冻整个模型

```
for param in model.parameters():
    param.requires_grad = True  # 解冻所有参数

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,
)  # 使用较小学习率微调整个模型
```

整体过程如下：

```
加载预训练模型
      ↓
冻结模型主体
      ↓
训练新的分类层
      ↓
解冻模型
      ↓
使用较小学习率进行整体微调
```

---

## 10. 训练函数

训练过程与之前基本相同：

```
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # 将模型设置为训练模式
    total_loss = 0  # 记录总损失
    num_correct = 0  # 记录预测正确的样本数量
    num_samples = 0  # 记录样本总数

    for images, targets in loader:
        images = images.to(device)  # 将图片移动到指定设备
        targets = targets.to(device)  # 将标签移动到指定设备

        scores = model(images)  # 前向传播
        loss = criterion(scores, targets)  # 计算损失

        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新允许训练的参数

        total_loss += loss.item() * images.size(0)  # 累加当前批次的总损失

        predictions = scores.argmax(dim=1)  # 获取预测类别
        num_correct += (predictions == targets).sum().item()  # 累加正确数量
        num_samples += targets.size(0)  # 累加样本数量

    average_loss = total_loss / num_samples  # 计算平均损失
    accuracy = num_correct / num_samples * 100  # 计算准确率
    return average_loss, accuracy
```

---

## 11. 验证函数

```
def evaluate(model, loader, criterion, device):
    model.eval()  # 将模型切换到评估模式
    total_loss = 0  # 记录总损失
    num_correct = 0  # 记录预测正确的样本数量
    num_samples = 0  # 记录样本总数

    with torch.no_grad():  # 关闭梯度计算
        for images, targets in loader:
            images = images.to(device)  # 将图片移动到指定设备
            targets = targets.to(device)  # 将标签移动到指定设备

            scores = model(images)  # 获取类别分数
            loss = criterion(scores, targets)  # 计算验证损失

            total_loss += loss.item() * images.size(0)  # 累加损失

            predictions = scores.argmax(dim=1)  # 获取预测类别
            num_correct += (predictions == targets).sum().item()  # 累加正确数量
            num_samples += targets.size(0)  # 累加样本数量

    average_loss = total_loss / num_samples  # 计算平均损失
    accuracy = num_correct / num_samples * 100  # 计算准确率
    return average_loss, accuracy
```

调用训练和验证函数：

```
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )  # 训练一个 epoch

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )  # 在验证集上评估模型

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_accuracy:.2f}%, "
        f"Val Loss: {val_loss:.4f}, "
        f"Val Acc: {val_accuracy:.2f}%"
    )  # 输出训练结果
```

---

## 12. 只微调部分网络

除了冻结整个模型或训练整个模型，还可以只解冻最后一部分。

例如，只训练 ResNet 的最后一个残差阶段和分类层：

```
for param in model.parameters():
    param.requires_grad = False  # 先冻结所有参数

for param in model.layer4.parameters():
    param.requires_grad = True  # 解冻最后一个残差阶段

for param in model.fc.parameters():
    param.requires_grad = True  # 解冻分类层
```

然后只把允许训练的参数交给优化器：

```
trainable_parameters = filter(
    lambda param: param.requires_grad,
    model.parameters(),
)  # 筛选需要更新的参数

optimizer = optim.Adam(
    trainable_parameters,
    lr=0.0001,
)  # 只更新解冻的参数
```

这种方式介于“固定特征提取”和“完整微调”之间。

---

## 13. 保存最好的模型

通常根据验证集准确率保存效果最好的模型：

```
best_accuracy = 0.0  # 记录目前最好的验证集准确率

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy  # 更新最好准确率
        torch.save(
            model.state_dict(),
            "best_model.pth",
        )  # 保存当前最好的模型参数
```

加载最好的模型：

```
model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device,
    )
)  # 加载验证集上效果最好的参数

model.eval()  # 切换到评估模式
```

---

## 14. 完整核心示例

下面省略重复的数据加载和验证代码，只展示迁移学习最核心的部分：

```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # 选择运行设备

num_classes = 2  # 设置自己任务中的类别数量

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)  # 加载 ImageNet 预训练权重

for param in model.parameters():
    param.requires_grad = False  # 冻结预训练参数

num_features = model.fc.in_features  # 获取原分类层的输入特征数
model.fc = nn.Linear(
    num_features,
    num_classes,
)  # 替换成适合当前任务的分类层

model = model.to(device)  # 将模型移动到指定设备

criterion = nn.CrossEntropyLoss()  # 创建损失函数
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001,
)  # 第一阶段只训练分类层


for epoch in range(3):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )  # 训练新的分类层

    print(
        f"Head Training [{epoch + 1}/3], "
        f"Loss: {train_loss:.4f}, "
        f"Accuracy: {train_accuracy:.2f}%"
    )


for param in model.parameters():
    param.requires_grad = True  # 解冻整个模型

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,
)  # 第二阶段使用较小学习率微调整个模型


for epoch in range(3):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )  # 微调整个模型

    print(
        f"Fine-tuning [{epoch + 1}/3], "
        f"Loss: {train_loss:.4f}, "
        f"Accuracy: {train_accuracy:.2f}%"
    )
```

---

## 15. 总结

### 迁移学习的核心步骤

```
model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)  # 加载预训练模型

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes,
)  # 替换分类层
```

### 固定特征提取

```
for param in model.parameters():
    param.requires_grad = False  # 冻结预训练参数

model.fc = nn.Linear(model.fc.in_features, num_classes)  # 创建新分类层

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001,
)  # 只训练分类层
```

### 完整微调

```
for param in model.parameters():
    param.requires_grad = True  # 解冻所有参数

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,
)  # 使用较小学习率训练整个模型
```

### 如何选择？

```
数据较少、与 ImageNet 相似：
冻结主体，只训练分类层

数据较多、与 ImageNet 有一定差异：
解冻部分或全部网络进行微调

常用策略：
先冻结训练分类层，再解冻并使用小学习率微调
```

### 总结

```
迁移学习利用预训练模型已经学到的通用特征；
Fine-tuning 则使用自己的数据进一步调整这些特征。
```


# 在 PyTorch 中创建自定义图片数据集

## 1. `Dataset` 和 `DataLoader`

PyTorch 中，图片数据通常由两个部分负责：

```
Dataset：定义如何读取一个样本
DataLoader：将样本组合成批次，并负责打乱、多进程加载等
```

自定义数据集需要继承 `torch.utils.data.Dataset`，并实现三个方法：

```
__init__()  # 初始化数据集，读取图片路径和标签
__len__()  # 返回数据集中的样本数量
__getitem__()  # 根据索引读取并返回一个样本
```

这是 PyTorch 官方推荐的自定义数据集基本结构。[PyTorch 自定义 Dataset 教程](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)

---

## 2. 数据集结构

假设图片保存在：

```
dataset/
├── images/
│   ├── cat_01.jpg
│   ├── cat_02.jpg
│   ├── dog_01.jpg
│   └── dog_02.jpg
└── labels.csv
```

`labels.csv` 中保存图片名称和对应标签：

```
filename,label
cat_01.jpg,0
cat_02.jpg,0
dog_01.jpg,1
dog_02.jpg,1
```

这里规定：

```
0 → cat
1 → dog
```

---

## 3. 创建自定义图片数据集

```
import csv  # 用于读取 CSV 文件
from pathlib import Path  # 用于处理文件路径
from PIL import Image  # 用于读取图片

from torch.utils.data import Dataset  # 导入 Dataset 基类


class CustomImageDataset(Dataset):  # 创建自定义图片数据集
    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = Path(image_dir)  # 保存图片文件夹路径
        self.transform = transform  # 保存图片预处理方法
        self.samples = []  # 用于保存每张图片的文件名和标签

        with open(csv_file, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)  # 按照列名读取 CSV 文件

            for row in reader:
                filename = row["filename"]  # 获取图片文件名
                label = int(row["label"])  # 获取标签并转换为整数
                self.samples.append((filename, label))  # 保存文件名和标签

    def __len__(self):
        return len(self.samples)  # 返回数据集中的样本数量

    def __getitem__(self, index):
        filename, label = self.samples[index]  # 获取指定样本的文件名和标签
        image_path = self.image_dir / filename  # 拼接完整图片路径

        with Image.open(image_path) as image:
            image = image.convert("RGB")  # 将图片统一转换为 RGB 格式

        if self.transform is not None:
            image = self.transform(image)  # 对图片执行缩放、张量转换等操作

        return image, label  # 返回图片和对应标签
```

### 三个方法的作用

```
__init__()
```

在创建数据集对象时执行一次，适合读取图片路径和标签。通常不在这里一次性读取所有图片，否则数据量较大时会占用大量内存。

```
__len__()
```

让下面的代码能够正常工作：

```
len(dataset)  # 获取数据集中的样本数量
```

```
__getitem__(index)
```

读取第 `index` 个样本：

```
image, label = dataset[0]  # 获取第一个样本
```

图片只会在需要时从硬盘读取。

---

## 4. 设置图片预处理

```
from torchvision import transforms  # 导入图像转换工具


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将所有图片统一缩放为 224×224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图片
    transforms.ToTensor(),  # 将 PIL 图片转换为张量
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # 使用 ImageNet 的均值
        std=[0.229, 0.224, 0.225],  # 使用 ImageNet 的标准差
    ),
])
```

如果不使用 ImageNet 预训练模型，也可以根据自己的数据集计算均值和标准差，或者暂时只使用：

```
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一图片大小
    transforms.ToTensor(),  # 将图片转换为张量
])
```

---

## 5. 创建 `Dataset` 和 `DataLoader`

```
from torch.utils.data import DataLoader  # 导入数据加载器


dataset = CustomImageDataset(
    csv_file="dataset/labels.csv",  # 指定标签文件
    image_dir="dataset/images",  # 指定图片目录
    transform=transform,  # 指定图片预处理方法
)

data_loader = DataLoader(
    dataset=dataset,  # 指定自定义数据集
    batch_size=32,  # 每个批次包含 32 个样本
    shuffle=True,  # 每轮训练前打乱数据
    num_workers=0,  # 使用主进程加载数据，适合初学和 Windows 环境
)

print(f"样本数量：{len(dataset)}")  # 输出数据集大小
```

读取一个批次：

```
images, labels = next(iter(data_loader))  # 获取第一个批次

print(images.shape)  # 例如 torch.Size([32, 3, 224, 224])
print(labels.shape)  # 例如 torch.Size([32])
```

训练时直接遍历：

```
for images, labels in data_loader:
    images = images.to(device)  # 将图片移动到指定设备
    labels = labels.to(device)  # 将标签移动到指定设备

    scores = model(images)  # 前向传播
    loss = criterion(scores, labels)  # 计算损失

    optimizer.zero_grad()  # 清除旧梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新模型参数
```

---

## 6. 更简单的方式：`ImageFolder`

如果数据已经按照类别文件夹排列，就不需要自己编写 `Dataset`。

目录结构：

```
dataset/
├── cats/
│   ├── cat_01.jpg
│   └── cat_02.jpg
└── dogs/
    ├── dog_01.jpg
    └── dog_02.jpg
```

直接使用：

```
from torchvision.datasets import ImageFolder  # 导入 ImageFolder
from torch.utils.data import DataLoader  # 导入 DataLoader


dataset = ImageFolder(
    root="dataset",  # 每个子文件夹会被自动识别为一个类别
    transform=transform,  # 设置图片预处理
)

data_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)

print(dataset.classes)  # 输出 ['cats', 'dogs']
print(dataset.class_to_idx)  # 输出 {'cats': 0, 'dogs': 1}
```

`ImageFolder` 会根据类别文件夹自动生成标签，具体目录规范可以参考 [Torchvision ImageFolder 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)。

---

## 7. 如何选择？

```
图片已经按类别放在不同文件夹中
→ 直接使用 ImageFolder

标签保存在 CSV、JSON 或数据库中
→ 编写自定义 Dataset

每个样本需要特殊读取或处理逻辑
→ 编写自定义 Dataset
```

---

## 8. 重点总结

自定义图片数据集的核心结构：

```
class CustomImageDataset(Dataset):
    def __init__(self, ...):
        # 保存图片路径、标签和 transform
        pass

    def __len__(self):
        # 返回样本数量
        pass

    def __getitem__(self, index):
        # 读取一张图片
        # 获取对应标签
        # 执行 transform
        # 返回 image, label
        pass
```

完整数据流：

```
图片文件和标签
      ↓
自定义 Dataset
      ↓
__getitem__ 读取单个样本
      ↓
transform 处理图片
      ↓
DataLoader 组成批次
      ↓
送入模型训练
```

总结：

```
Dataset 决定一个样本怎么读取，
DataLoader 决定这些样本怎么组成批次。
```


# 在 PyTorch 中创建自定义文本数据集

## 1. 文本数据与图片数据的区别

图片可以直接转换为数值张量，但模型不能直接处理字符串。

文本送入模型前通常需要经过：

```
原始文本
   ↓
分词 Tokenization
   ↓
单词转换为编号
   ↓
补齐序列长度 Padding
   ↓
组成批次
   ↓
送入模型
```

例如：

```
原始文本：I love PyTorch
分词结果：["i", "love", "pytorch"]
编号结果：[5, 18, 42]
```

自定义文本数据集同样需要继承 `Dataset`，实现：

```
__init__()  # 读取文本和标签
__len__()  # 返回样本数量
__getitem__()  # 返回一个文本样本及其标签
```

---

## 2. 准备文本数据

假设要完成一个简单的情感二分类任务：

```
dataset/
└── reviews.csv
```

`reviews.csv` 内容如下：

```
text,label
I really like this movie,1
This movie is very interesting,1
I do not like this film,0
The story is boring,0
```

标签规定：

```
0 → 负面评价
1 → 正面评价
```

---

## 3. 文本分词

这里使用最简单的空格分词：

```
def tokenize(text):
    return text.lower().split()  # 转换为小写，然后按照空格切分
```

示例：

```
text = "I Love PyTorch"
tokens = tokenize(text)
print(tokens)  # ['i', 'love', 'pytorch']
```

这种方法适合简单英文示例，但无法很好地处理标点，也不适合没有自然空格的中文。

实际中文任务通常使用：

- 单字切分
- jieba
- 预训练模型自带的 tokenizer

---

## 4. 创建词表

词表负责把单词映射为整数：

```
"<pad>"   → 0
"<unk>"   → 1
"i"       → 2
"like"    → 3
"movie"   → 4
```

其中：

- `<pad>`：用于把不同长度的文本补齐到相同长度。
- `<unk>`：表示词表中没有出现过的未知单词。

```
from collections import Counter  # 用于统计单词出现次数


def build_vocabulary(texts, tokenizer, min_frequency=1):
    counter = Counter()  # 创建词频统计器

    for text in texts:
        tokens = tokenizer(text)  # 对文本进行分词
        counter.update(tokens)  # 统计每个单词出现的次数

    vocabulary = {
        "<pad>": 0,  # 填充符编号
        "<unk>": 1,  # 未知单词编号
    }

    for word, frequency in counter.items():
        if frequency >= min_frequency:
            vocabulary[word] = len(vocabulary)  # 为满足词频要求的单词分配编号

    return vocabulary  # 返回单词到编号的映射
```

`min_frequency` 用于过滤出现次数过少的单词：

```
min_frequency=2  # 只把至少出现两次的单词加入词表
```

低频单词会统一转换为 `<unk>`。

需要注意：正式任务中应该只使用训练集创建词表，不能使用验证集或测试集创建词表，否则可能造成数据泄漏。

---

## 5. 创建自定义文本数据集

```
import csv  # 用于读取 CSV 文件
import torch  # 导入 PyTorch
from torch.utils.data import Dataset  # 导入 Dataset 基类


class TextClassificationDataset(Dataset):
    def __init__(self, csv_file, vocabulary=None, tokenizer=None):
        self.texts = []  # 保存所有文本
        self.labels = []  # 保存所有标签
        self.tokenizer = tokenizer or tokenize  # 使用传入的分词函数，否则使用默认函数

        with open(csv_file, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)  # 根据 CSV 列名读取数据

            for row in reader:
                self.texts.append(row["text"])  # 保存文本
                self.labels.append(int(row["label"]))  # 保存整数标签

        if vocabulary is None:
            self.vocabulary = build_vocabulary(
                self.texts,
                self.tokenizer,
            )  # 没有传入词表时，使用当前数据创建词表
        else:
            self.vocabulary = vocabulary  # 验证集和测试集使用训练集的词表

    def __len__(self):
        return len(self.texts)  # 返回文本样本的数量

    def __getitem__(self, index):
        text = self.texts[index]  # 获取指定文本
        label = self.labels[index]  # 获取对应标签

        tokens = self.tokenizer(text)  # 将文本切分成单词

        token_ids = [
            self.vocabulary.get(
                token,
                self.vocabulary["<unk>"],
            )
            for token in tokens
        ]  # 将每个单词转换为编号，未知单词使用 <unk>

        token_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        )  # 将单词编号转换为一维整数张量

        label = torch.tensor(
            label,
            dtype=torch.long,
        )  # 将标签转换为整数张量

        return token_ids, label  # 返回单词编号和标签
```

创建训练数据集：

```
train_dataset = TextClassificationDataset(
    csv_file="dataset/train.csv",
)  # 使用训练数据创建词表

print(len(train_dataset))  # 输出训练样本数量
print(train_dataset.vocabulary)  # 查看词表
```

创建验证集时，应该传入训练集的词表：

```
val_dataset = TextClassificationDataset(
    csv_file="dataset/val.csv",
    vocabulary=train_dataset.vocabulary,
)  # 验证集复用训练集词表
```

---

## 6. 为什么需要 Padding？

不同文本包含的单词数量通常不同：

```
"I like it"                  → 长度 3
"I really like this movie"   → 长度 5
```

但是一个批次中的张量必须具有统一形状，因此要用 `<pad>` 将短序列补齐：

```
[2, 5, 7]       → [2, 5, 7, 0, 0]
[2, 8, 5, 4, 9] → [2, 8, 5, 4, 9]
```

这里的 `0` 是 `<pad>` 的编号。

---

## 7. 自定义批次整理函数

`collate_fn` 负责将多个长度不同的样本整理成一个批次：

```
from torch.nn.utils.rnn import pad_sequence  # 导入序列补齐函数


def collate_batch(batch):
    token_sequences = []  # 保存当前批次的文本序列
    labels = []  # 保存当前批次的标签
    lengths = []  # 保存每个文本的真实长度

    for token_ids, label in batch:
        token_sequences.append(token_ids)  # 添加文本编号序列
        labels.append(label)  # 添加标签
        lengths.append(len(token_ids))  # 记录补齐前的文本长度

    padded_sequences = pad_sequence(
        token_sequences,
        batch_first=True,
        padding_value=0,
    )  # 将文本补齐为相同长度，0 是 <pad> 的编号

    labels = torch.stack(labels)  # 将多个标签组合成一个张量
    lengths = torch.tensor(lengths, dtype=torch.long)  # 将真实长度转换为张量

    return padded_sequences, lengths, labels  # 返回文本、真实长度和标签
```

假设当前批次包含两个序列：

```
[2, 5, 7]
[2, 8, 5, 4, 9]
```

补齐后的结果为：

```
[
    [2, 5, 7, 0, 0],
    [2, 8, 5, 4, 9]
]
```

张量形状为：

```
(batch_size, 当前批次的最大序列长度)
```

---

## 8. 创建 DataLoader

```
from torch.utils.data import DataLoader  # 导入 DataLoader


train_loader = DataLoader(
    dataset=train_dataset,  # 指定训练数据集
    batch_size=32,  # 每个批次包含 32 条文本
    shuffle=True,  # 每轮训练前打乱数据
    collate_fn=collate_batch,  # 使用自定义函数补齐文本
)

val_loader = DataLoader(
    dataset=val_dataset,  # 指定验证数据集
    batch_size=32,  # 每个批次包含 32 条文本
    shuffle=False,  # 验证时不需要打乱
    collate_fn=collate_batch,  # 使用相同的批次整理函数
)
```

读取一个批次：

```
token_ids, lengths, labels = next(iter(train_loader))  # 获取第一个批次

print(token_ids.shape)  # (batch_size, max_sequence_length)
print(lengths.shape)  # (batch_size,)
print(labels.shape)  # (batch_size,)
```

---

## 9. 数据如何送入模型？

文本编号通常先经过 `nn.Embedding`，转换为连续的词向量：

```
import torch.nn as nn  # 导入神经网络模块


vocabulary_size = len(train_dataset.vocabulary)  # 获取词表大小
embedding_size = 128  # 设置词向量维度

embedding = nn.Embedding(
    num_embeddings=vocabulary_size,  # 词表中的单词数量
    embedding_dim=embedding_size,  # 每个单词转换为 128 维向量
    padding_idx=0,  # 指定 0 是填充符
)
```

输入形状：

```
(batch_size, sequence_length)
```

经过 Embedding 后：

```
embedded = embedding(token_ids)  # 将单词编号转换为词向量
```

输出形状：

```
(batch_size, sequence_length, embedding_size)
```

完整变化如下：

```
原始文本
"I love PyTorch"
      ↓
分词
["i", "love", "pytorch"]
      ↓
词表编号
[2, 8, 15]
      ↓
Padding
[2, 8, 15, 0, 0]
      ↓
Embedding
(sequence_length, embedding_size)
      ↓
RNN、LSTM 或 Transformer
```

---

## 10. 一个简单的 LSTM 文本分类模型

```
class TextLSTM(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_size,
        hidden_size,
        num_classes,
    ):
        super(TextLSTM, self).__init__()  # 初始化父类

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,  # 词表大小
            embedding_dim=embedding_size,  # 词向量维度
            padding_idx=0,  # 指定填充符编号
        )

        self.lstm = nn.LSTM(
            input_size=embedding_size,  # 每个时间步接收一个词向量
            hidden_size=hidden_size,  # 隐藏状态大小
            batch_first=True,  # 输入形状为 (batch, sequence, feature)
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes,
        )  # 将文本特征映射为类别分数

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)  # 将单词编号转换为词向量

        _, (hidden, cell) = self.lstm(embedded)  # 使用 LSTM 编码整个文本
        last_hidden = hidden[-1]  # 获取最后一层的最终隐藏状态

        scores = self.fc(last_hidden)  # 计算类别分数
        return scores  # 返回分类结果
```

使用模型：

```
model = TextLSTM(
    vocabulary_size=len(train_dataset.vocabulary),  # 词表大小
    embedding_size=128,  # 词向量维度
    hidden_size=256,  # LSTM 隐藏状态大小
    num_classes=2,  # 情感二分类
).to(device)
```

训练时：

```
for token_ids, lengths, labels in train_loader:
    token_ids = token_ids.to(device)  # 移动文本编号
    labels = labels.to(device)  # 移动标签

    scores = model(token_ids)  # 前向传播
    loss = criterion(scores, labels)  # 计算分类损失

    optimizer.zero_grad()  # 清除旧梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新模型参数
```

这个简单版本没有使用 `lengths`。更严格的实现可以使用 `pack_padded_sequence`，避免 LSTM 继续处理补齐的 `<pad>`，但初学阶段先理解整体数据流程即可。

---

## 11. 中文文本如何处理？

简单中文单字切分：

```
def chinese_tokenize(text):
    text = text.replace(" ", "")  # 删除文本中的空格
    return list(text)  # 将每个汉字作为一个 token
```

例如：

```
tokens = chinese_tokenize("我喜欢 PyTorch")

print(tokens)
# ['我', '喜', '欢', 'P', 'y', 'T', 'o', 'r', 'c', 'h']
```

这种方式适合演示，但实际效果有限。

真实项目通常直接使用预训练模型对应的 tokenizer：

```
encoded = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt",
)
```

因为预训练模型的词表和 tokenizer 必须相互对应，不能随意更换。

---

## 12. 常见问题

### 训练集与验证集分别创建词表

错误做法：

```
train_dataset = TextClassificationDataset("train.csv")
val_dataset = TextClassificationDataset("val.csv")  # 重新创建了另一份词表
```

相同单词可能在两个词表中对应不同编号。

正确做法：

```
train_dataset = TextClassificationDataset("train.csv")

val_dataset = TextClassificationDataset(
    "val.csv",
    vocabulary=train_dataset.vocabulary,
)  # 验证集复用训练集词表
```

### 忘记对文本进行 Padding

不同长度的张量无法直接组合成一个规则的批次，因此需要在 `collate_fn` 中使用：

```
pad_sequence(...)
```

### Embedding 的输入类型错误

`nn.Embedding` 要求输入为整数编号：

```
token_ids = torch.tensor(token_ids, dtype=torch.long)
```

不能使用浮点类型。

### 没有指定 `padding_idx`

推荐写成：

```
nn.Embedding(
    vocabulary_size,
    embedding_size,
    padding_idx=0,
)
```

这样填充符对应的词向量不会像普通单词一样被更新。

---

## 13. 重点总结

自定义文本数据集的核心流程：

```
读取文本和标签
      ↓
对文本进行分词
      ↓
使用词表转换为整数编号
      ↓
Dataset 返回 token_ids 和 label
      ↓
collate_fn 对不同长度的文本进行补齐
      ↓
DataLoader 组成批次
      ↓
Embedding 将编号转换为词向量
      ↓
送入 LSTM、RNN 或 Transformer
```

需要重点记住：

```
class TextDataset(Dataset):
    def __init__(self, ...):
        # 读取文本、标签和词表
        pass

    def __len__(self):
        # 返回文本数量
        pass

    def __getitem__(self, index):
        # 分词
        # 转换为单词编号
        # 返回 token_ids 和 label
        pass
```

总结：

```
图片数据集负责把图片转换为张量，
文本数据集则需要先完成分词和数字化，
再通过 Padding 将不同长度的文本组成批次。
```



# 使用 Torchvision 进行图像数据增强

## 1. 什么是数据增强？

数据增强（Data Augmentation）是指在不收集新图片的情况下，对已有图片进行随机变换，从而增加训练数据的多样性。

常见操作包括：

```
随机裁剪
随机翻转
随机旋转
颜色变化
随机擦除
```

例如，同一张图片在不同训练轮次中可能变成：

```
原图 → 随机裁剪
原图 → 水平翻转
原图 → 轻微旋转
原图 → 调整亮度
```

数据增强的主要作用：

- 减少模型过拟合
- 提高模型的泛化能力
- 增强模型对位置、光照和角度变化的适应能力

---

## 2. `torchvision.transforms.v2`

Torchvision 提供了大量常用的图像变换。目前可以使用 `torchvision.transforms.v2` 组合数据预处理和数据增强操作。[Torchvision Transforms 文档](https://docs.pytorch.org/vision/stable/transforms.html)

```
import torch  # 导入 PyTorch
from torchvision.transforms import v2  # 导入新版图像转换模块
```

多个转换可以通过 `v2.Compose` 按照顺序组合：

```
transform = v2.Compose([
    v2.Resize((256, 256)),  # 调整图片大小
    v2.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    v2.ToImage(),  # 将输入转换为 torchvision 图片对象
    v2.ToDtype(torch.float32, scale=True),  # 转换为 float32，并将像素缩放到 [0, 1]
])
```

---

## 3. 常用的数据增强方法

### 随机水平翻转

```
v2.RandomHorizontalFlip(p=0.5)  # 以 50% 的概率水平翻转图片
```

`p` 表示执行翻转的概率：

```
p=0.5 → 大约一半的图片会被翻转
p=1.0 → 所有图片都会被翻转
p=0.0 → 所有图片都不会被翻转
```

水平翻转适合猫、狗、汽车等物体，但不一定适合文字、数字和具有明确方向的任务。

---

### 随机垂直翻转

```
v2.RandomVerticalFlip(p=0.5)  # 以 50% 的概率垂直翻转图片
```

垂直翻转更适合卫星图像、显微图像等方向不重要的数据。

对于人物、车辆和自然场景，一般不要随意使用垂直翻转。

---

### 随机旋转

```
v2.RandomRotation(degrees=15)  # 在 -15° 到 15° 之间随机旋转图片
```

也可以指定旋转范围：

```
v2.RandomRotation(degrees=(-10, 20))  # 在 -10° 到 20° 之间随机旋转
```

旋转角度不宜过大，否则可能改变图片的真实类别。

---

### 随机裁剪

```
v2.RandomCrop(
    size=(224, 224),  # 随机裁剪为 224×224
    padding=4,  # 裁剪前在图片边缘填充 4 个像素
)
```

随机裁剪可以让模型减少对物体固定位置的依赖。

---

### 随机裁剪并缩放

```
v2.RandomResizedCrop(
    size=(224, 224),  # 最终输出尺寸
    scale=(0.8, 1.0),  # 随机裁剪原图 80%～100% 的区域
)
```

它会先随机选择图片的一部分，再缩放到指定尺寸，是图像分类中很常见的增强方式。[RandomResizedCrop 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomResizedCrop.html)

---

### 颜色抖动

```
v2.ColorJitter(
    brightness=0.2,  # 随机调整亮度
    contrast=0.2,  # 随机调整对比度
    saturation=0.2,  # 随机调整饱和度
    hue=0.1,  # 随机调整色调
)
```

颜色抖动可以提高模型对光照和拍摄环境变化的适应能力。

如果颜色本身决定类别，例如判断水果是否成熟，就不应该使用过强的颜色增强。

---

### 随机灰度化

```
v2.RandomGrayscale(p=0.1)  # 以 10% 的概率将图片转换为灰度效果
```

它可以减少模型对颜色信息的过度依赖，但不适合严重依赖颜色的分类任务。

---

### 随机擦除

```
v2.RandomErasing(
    p=0.25,  # 以 25% 的概率执行随机擦除
    scale=(0.02, 0.2),  # 擦除区域占图片面积的比例
)
```

随机擦除会遮挡图片中的一小块区域，使模型不能只依赖某个局部特征。

`RandomErasing` 需要作用于张量，一般放在 `ToImage` 和 `ToDtype` 之后。

---

## 4. 训练集增强流程

下面是一套常见的 ImageNet 预训练模型输入处理：

```
import torch  # 导入 PyTorch
from torchvision.transforms import v2  # 导入图像转换模块


train_transform = v2.Compose([
    v2.RandomResizedCrop(
        size=(224, 224),
        scale=(0.8, 1.0),
    ),  # 随机裁剪图片并缩放为 224×224

    v2.RandomHorizontalFlip(p=0.5),  # 以 50% 的概率水平翻转

    v2.RandomRotation(degrees=10),  # 在 -10° 到 10° 之间随机旋转

    v2.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),  # 随机调整图片颜色

    v2.ToImage(),  # 将图片转换为 torchvision 图片对象

    v2.ToDtype(
        torch.float32,
        scale=True,
    ),  # 转换为 float32，并将像素值缩放到 [0, 1]

    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),  # 使用 ImageNet 均值和标准差进行标准化

    v2.RandomErasing(
        p=0.25,
        scale=(0.02, 0.15),
    ),  # 随机遮挡部分区域
])
```

转换顺序很重要：

```
随机裁剪、翻转、旋转和颜色变化
                ↓
转换为图片张量
                ↓
转换数据类型并缩放到 [0, 1]
                ↓
标准化
                ↓
随机擦除
```

---

## 5. 验证集和测试集转换

验证集和测试集用于客观评价模型，不应该使用随机数据增强。

```
val_transform = v2.Compose([
    v2.Resize(256),  # 将较短的一边缩放为 256

    v2.CenterCrop((224, 224)),  # 从图片中心裁剪 224×224

    v2.ToImage(),  # 转换为 torchvision 图片对象

    v2.ToDtype(
        torch.float32,
        scale=True,
    ),  # 转换为 float32，并缩放到 [0, 1]

    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),  # 使用与训练集相同的标准化参数
])
```

核心区别：

```
训练集：使用随机增强
验证集：只进行固定预处理
测试集：只进行固定预处理
```

如果验证集也使用随机增强，每次验证得到的结果可能不同，无法稳定评价模型。

---

## 6. 与 `ImageFolder` 一起使用

```
dataset/
├── train/
│   ├── cats/
│   └── dogs/
└── val/
    ├── cats/
    └── dogs/
```

加载数据：

```
from torch.utils.data import DataLoader  # 导入数据加载器
from torchvision.datasets import ImageFolder  # 导入 ImageFolder


train_dataset = ImageFolder(
    root="dataset/train",
    transform=train_transform,
)  # 训练集使用随机数据增强

val_dataset = ImageFolder(
    root="dataset/val",
    transform=val_transform,
)  # 验证集只使用固定预处理

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)  # 创建训练数据加载器

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)  # 创建验证数据加载器
```

`Dataset` 每次读取图片时都会重新执行 `transform`，因此同一张训练图片在不同 epoch 中可能产生不同结果。

---

## 7. 与自定义 Dataset 一起使用

自定义数据集只需要接收一个 `transform`：

```
class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # 保存图片路径和标签
        self.transform = transform  # 保存数据增强方法

    def __len__(self):
        return len(self.samples)  # 返回样本数量

    def __getitem__(self, index):
        image_path, label = self.samples[index]  # 获取图片路径和标签

        with Image.open(image_path) as image:
            image = image.convert("RGB")  # 读取并转换为 RGB 图片

        if self.transform is not None:
            image = self.transform(image)  # 执行数据增强和预处理

        return image, label  # 返回处理后的图片和标签
```

分别创建训练集和验证集：

```
train_dataset = CustomImageDataset(
    train_samples,
    transform=train_transform,
)  # 训练集使用随机增强

val_dataset = CustomImageDataset(
    val_samples,
    transform=val_transform,
)  # 验证集使用固定预处理
```

---

## 8. 简单数据集的增强示例

对于 CIFAR-10 这类 `32×32` 的小图片，可以使用：

```
cifar_train_transform = v2.Compose([
    v2.RandomCrop(
        size=(32, 32),
        padding=4,
    ),  # 先填充，再随机裁剪为 32×32

    v2.RandomHorizontalFlip(p=0.5),  # 随机水平翻转

    v2.ToImage(),  # 转换为图片对象

    v2.ToDtype(
        torch.float32,
        scale=True,
    ),  # 转换为 float32

    v2.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),  # 使用 CIFAR-10 的均值和标准差
])
```

对于 MNIST，可以进行轻微的旋转和平移：

```
mnist_train_transform = v2.Compose([
    v2.RandomAffine(
        degrees=10,  # 在 -10° 到 10° 之间旋转
        translate=(0.1, 0.1),  # 在水平和垂直方向轻微平移
    ),

    v2.ToImage(),  # 转换为图片对象

    v2.ToDtype(
        torch.float32,
        scale=True,
    ),  # 转换为 float32
])
```

不建议对 MNIST 使用水平翻转：

```
数字经过翻转后可能不再符合真实书写规律
```

---

## 9. 自动数据增强

Torchvision 还提供了一些自动增强策略：

```
automatic_transform = v2.Compose([
    v2.RandomResizedCrop((224, 224)),  # 随机裁剪并缩放
    v2.RandAugment(),  # 自动选择增强操作和增强强度
    v2.ToImage(),  # 转换为图片对象
    v2.ToDtype(torch.float32, scale=True),  # 转换为浮点张量
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),  # 标准化
])
```

常见的自动增强方法包括：

```
AutoAugment
RandAugment
TrivialAugmentWide
AugMix
```

初学阶段可以先掌握基本增强。如果需要更强的数据增强，再尝试：

```
v2.RandAugment()
```

---

## 10. MixUp 和 CutMix

MixUp 和 CutMix 会同时组合两张图片及其标签：

```
MixUp：将两张图片按照一定比例混合
CutMix：将一张图片的局部区域替换为另一张图片
```

它们与普通图片变换不同，接收的是整个批次，而不是单张图片。[Torchvision MixUp 和 CutMix 教程](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html)

简单示例：

```
mixup = v2.MixUp(
    num_classes=10,
    alpha=0.2,
)  # 创建 MixUp 增强方法

for images, labels in train_loader:
    images, labels = mixup(images, labels)  # 对整个批次的图片和标签进行混合

    images = images.to(device)  # 将图片移动到指定设备
    labels = labels.to(device)  # 将混合标签移动到指定设备

    scores = model(images)  # 前向传播
    loss = criterion(scores, labels)  # 计算损失
```

MixUp 会把原本的整数标签转换为混合后的软标签，因此初学时可以先不使用。

---

## 11. 数据增强的注意事项

### 增强不能改变真实类别

数据增强必须保持图片原有语义：

```
猫水平翻转后仍然是猫       → 合理
猫轻微旋转后仍然是猫       → 合理
数字 6 旋转 180° 后像数字 9 → 可能不合理
文字水平翻转后无法阅读      → 不合理
```

### 增强强度不是越大越好

增强过弱可能无法有效减少过拟合，增强过强则可能破坏图片内容。

推荐从简单组合开始：

```
v2.RandomResizedCrop((224, 224))
v2.RandomHorizontalFlip()
v2.ColorJitter(...)
```

然后根据验证集效果逐步调整。

### 标准化不属于随机增强

```
v2.Normalize(...)
```

不会随机改变图片，而是固定的数据预处理操作。训练集、验证集和测试集应该使用相同的标准化参数。

### 先划分数据，再进行增强

正确流程：

```
完整数据集
   ↓
划分训练集和验证集
   ↓
训练集使用随机增强
验证集使用固定预处理
```

不要先生成大量增强图片再随机划分，否则原图和增强版本可能分别进入训练集和验证集，造成数据泄漏。

---

## 12. 重点总结

### 训练集

```
train_transform = v2.Compose([
    v2.RandomResizedCrop((224, 224)),  # 随机裁剪和缩放
    v2.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    v2.ColorJitter(brightness=0.2, contrast=0.2),  # 随机改变颜色
    v2.ToImage(),  # 转换为图片对象
    v2.ToDtype(torch.float32, scale=True),  # 转换为浮点张量
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),  # 标准化
])
```

### 验证集

```
val_transform = v2.Compose([
    v2.Resize(256),  # 固定缩放
    v2.CenterCrop((224, 224)),  # 固定中心裁剪
    v2.ToImage(),  # 转换为图片对象
    v2.ToDtype(torch.float32, scale=True),  # 转换为浮点张量
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),  # 使用相同标准化参数
])
```

### 核心原则

```
训练集：随机增强，提高数据多样性
验证集：固定预处理，保证评价稳定
增强目标：改变图片表现形式，但不能改变真实类别
```

总结：

```
数据增强通过随机生成同一图片的不同合理版本，
让模型学习更稳定、更通用的特征，而不是记住训练图片。
```



# 使用 Albumentations 进行图像数据增强

## 1. Albumentations 是什么？

Albumentations 是一个专门用于计算机视觉数据增强的 Python 库，可以与 PyTorch 配合使用。

它支持：

- 图像分类
- 目标检测
- 图像分割
- 关键点检测

Albumentations 的特点是速度快、增强方法丰富，并且可以让图片、边界框和分割掩码进行同步变换。

在 PyTorch 中，通常在自定义 `Dataset` 的 `__getitem__()` 中执行 Albumentations。[Albumentations PyTorch 集成文档](https://albumentations.ai/docs/3-basic-usage/framework-integrations/)

---

## 2. 安装

```
pip install albumentations opencv-python
```

主要导入：

```
import albumentations as A  # 导入 Albumentations
import cv2  # 用于读取和处理图片

from albumentations.pytorch import ToTensorV2  # 将图片转换为 PyTorch 张量
```

---

## 3. 基本使用方法

使用 `A.Compose` 将多个增强操作组合起来：

```
transform = A.Compose([
    A.Resize(height=224, width=224),  # 将图片缩放为 224×224
    A.HorizontalFlip(p=0.5),  # 以 50% 的概率水平翻转
    A.RandomBrightnessContrast(p=0.3),  # 随机调整亮度和对比度
    A.Normalize(),  # 对像素进行标准化
    ToTensorV2(),  # 转换为 PyTorch 张量
])
```

应用增强时必须使用关键字参数：

```
transformed = transform(image=image)  # 对图片执行增强
image = transformed["image"]  # 从返回的字典中取出增强后的图片
```

不能直接写成：

```
image = transform(image)  # 错误或不推荐的调用方式
```

因为 Albumentations 返回的是一个字典：

```
{
    "image": 增强后的图片
}
```

如果同时处理分割掩码，还可能返回：

```
{
    "image": 增强后的图片,
    "mask": 同步增强后的掩码
}
```

---

## 4. 参数 `p` 的作用

大部分随机增强操作都有一个 `p` 参数：

```
A.HorizontalFlip(p=0.5)
```

`p` 表示执行该操作的概率：

```
p=0.5 → 约 50% 的概率执行
p=1.0 → 每次都执行
p=0.0 → 永远不执行
```

例如：

```
A.Rotate(limit=15, p=0.5)  # 以 50% 的概率在 -15° 到 15° 之间随机旋转
```

`A.Compose` 本身也可以设置概率：

```
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ],
    p=0.8,
)  # 整套增强流程以 80% 的概率执行
```

---

## 5. 常用增强方法

### 随机水平翻转

```
A.HorizontalFlip(p=0.5)  # 以 50% 的概率水平翻转
```

适合猫、狗、汽车等普通物体分类，但不一定适合数字和文字。

### 随机旋转

```
A.Rotate(
    limit=15,
    p=0.5,
)  # 在 -15° 到 15° 之间随机旋转
```

### 随机裁剪并缩放

```
A.RandomResizedCrop(
    size=(224, 224),
    scale=(0.8, 1.0),
    p=1.0,
)  # 随机裁剪原图的 80%～100%，再缩放为 224×224
```

### 随机亮度和对比度

```
A.RandomBrightnessContrast(
    brightness_limit=0.2,
    contrast_limit=0.2,
    p=0.5,
)  # 随机改变亮度和对比度
```

### 色调、饱和度和明度变化

```
A.HueSaturationValue(
    hue_shift_limit=10,
    sat_shift_limit=20,
    val_shift_limit=20,
    p=0.5,
)  # 随机调整色调、饱和度和明度
```

### 模糊

```
A.GaussianBlur(
    blur_limit=(3, 5),
    p=0.2,
)  # 以 20% 的概率进行高斯模糊
```

### 随机遮挡

```
A.CoarseDropout(
    num_holes_range=(1, 4),
    hole_height_range=(16, 32),
    hole_width_range=(16, 32),
    p=0.3,
)  # 随机遮挡图片中的若干区域
```

随机遮挡可以让模型减少对某一个局部特征的依赖。

---

## 6. 训练集数据增强

下面是一套简单的图片分类增强流程：

```
train_transform = A.Compose([
    A.RandomResizedCrop(
        size=(224, 224),
        scale=(0.8, 1.0),
        p=1.0,
    ),  # 随机裁剪并缩放到 224×224

    A.HorizontalFlip(p=0.5),  # 随机水平翻转

    A.Rotate(
        limit=10,
        p=0.3,
    ),  # 以 30% 的概率轻微旋转

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5,
    ),  # 随机改变亮度和对比度

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),  # 使用 ImageNet 的均值和标准差进行标准化

    ToTensorV2(),  # 将 HWC 格式的 NumPy 图片转换为 CHW 格式的 PyTorch 张量
])
```

`ToTensorV2()` 通常放在最后，它会把图片从：

```
NumPy：(height, width, channels)
```

转换为 PyTorch 需要的：

```
Tensor：(channels, height, width)
```

具体行为可参考 [Albumentations ToTensorV2 文档](https://albumentations.ai/docs/api-reference/albumentations/pytorch/transforms/)。

---

## 7. 验证集和测试集

验证集和测试集不应该使用随机增强，只需要进行固定预处理：

```
val_transform = A.Compose([
    A.Resize(
        height=256,
        width=256,
    ),  # 固定缩放为 256×256

    A.CenterCrop(
        height=224,
        width=224,
    ),  # 从中心裁剪 224×224

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),  # 使用与训练集相同的标准化参数

    ToTensorV2(),  # 转换为 PyTorch 张量
])
```

核心原则：

```
训练集：随机增强
验证集：固定预处理
测试集：固定预处理
```

---

## 8. 在自定义 Dataset 中使用

```
from pathlib import Path  # 用于处理文件路径

import cv2  # 用于读取图片
from torch.utils.data import Dataset  # 导入 Dataset 基类


class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # 保存由图片路径和标签组成的列表
        self.transform = transform  # 保存 Albumentations 增强流程

    def __len__(self):
        return len(self.samples)  # 返回样本数量

    def __getitem__(self, index):
        image_path, label = self.samples[index]  # 获取图片路径和标签

        image = cv2.imread(str(Path(image_path)))  # 使用 OpenCV 读取图片

        if image is None:
            raise FileNotFoundError(
                f"无法读取图片：{image_path}"
            )  # 图片不存在或损坏时给出明确错误

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )  # 将 OpenCV 的 BGR 格式转换为 RGB 格式

        if self.transform is not None:
            transformed = self.transform(image=image)  # 执行数据增强
            image = transformed["image"]  # 获取增强后的图片

        return image, label  # 返回图片和标签
```

OpenCV 默认使用 BGR 通道顺序，而大多数模型和可视化工具使用 RGB，因此需要执行：

```
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

---

## 9. 创建 Dataset 和 DataLoader

假设样本列表如下：

```
train_samples = [
    ("dataset/train/cats/cat_01.jpg", 0),
    ("dataset/train/cats/cat_02.jpg", 0),
    ("dataset/train/dogs/dog_01.jpg", 1),
    ("dataset/train/dogs/dog_02.jpg", 1),
]  # 每个元素由图片路径和标签组成
```

创建数据集：

```
train_dataset = CustomImageDataset(
    samples=train_samples,
    transform=train_transform,
)  # 训练集使用随机增强

val_dataset = CustomImageDataset(
    samples=val_samples,
    transform=val_transform,
)  # 验证集使用固定预处理
```

创建数据加载器：

```
from torch.utils.data import DataLoader  # 导入数据加载器


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)  # 创建训练数据加载器

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)  # 创建验证数据加载器
```

读取一个批次：

```
images, labels = next(iter(train_loader))  # 获取第一个批次

print(images.shape)  # 例如 torch.Size([32, 3, 224, 224])
print(labels.shape)  # 例如 torch.Size([32])
```

---

## 10. 单独测试增强效果

正式训练前，最好先检查增强后的图片是否合理：

```
import cv2  # 导入 OpenCV
import matplotlib.pyplot as plt  # 导入绘图库


image = cv2.imread("example.jpg")  # 读取测试图片
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB

preview_transform = A.Compose([
    A.RandomResizedCrop(
        size=(224, 224),
        scale=(0.8, 1.0),
    ),  # 随机裁剪并缩放

    A.HorizontalFlip(p=0.5),  # 随机水平翻转

    A.RandomBrightnessContrast(p=0.5),  # 随机调整亮度和对比度
])

fig, axes = plt.subplots(2, 3, figsize=(10, 7))  # 创建 2×3 的画布

for axis in axes.flatten():
    augmented = preview_transform(image=image)["image"]  # 每次产生一种随机结果
    axis.imshow(augmented)  # 显示增强后的图片
    axis.axis("off")  # 隐藏坐标轴

plt.tight_layout()  # 自动调整布局
plt.show()  # 显示图片
```

这里没有使用 `Normalize` 和 `ToTensorV2`，因为直接显示 NumPy 格式的 RGB 图片更方便。

---

## 11. Albumentations 与 Torchvision 的主要区别

### Torchvision

```
image = transform(image)  # 通常直接传入 PIL 图片或张量
```

### Albumentations

```
result = transform(image=image)  # 使用关键字参数传入 NumPy 图片
image = result["image"]  # 从字典中获取处理结果
```

简单对比：

|项目|Torchvision|Albumentations|
|---|---|---|
|常见输入|PIL 图片或 Tensor|NumPy 数组|
|调用方式|`transform(image)`|`transform(image=image)`|
|返回值|处理后的图片|包含结果的字典|
|转为 Tensor|`ToDtype` 等|`ToTensorV2`|
|多目标同步变换|支持|使用方式直观且功能丰富|

Albumentations 特别适合需要同时增强图片、边界框、掩码和关键点的任务。

---

## 12. 同时处理图片和分割掩码

Albumentations 可以保证图片与掩码执行完全相同的空间变换：

```
segmentation_transform = A.Compose([
    A.Resize(height=256, width=256),  # 同时缩放图片和掩码
    A.HorizontalFlip(p=0.5),  # 同时翻转图片和掩码
    A.Rotate(limit=10, p=0.3),  # 同时旋转图片和掩码
    ToTensorV2(),  # 转换为张量
])
```

应用增强：

```
transformed = segmentation_transform(
    image=image,
    mask=mask,
)  # 同时传入图片和掩码

image = transformed["image"]  # 获取增强后的图片
mask = transformed["mask"]  # 获取同步增强后的掩码
```

如果图片发生水平翻转，对应的掩码也会以相同方式翻转，这对分割任务非常重要。

---

## 13. 注意事项

### OpenCV 的通道顺序

必须注意：

```
OpenCV 读取结果：BGR
常规模型期望输入：RGB
```

所以通常需要：

```
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

### `ToTensorV2` 放在最后

推荐顺序：

```
读取 NumPy 图片
      ↓
执行随机增强
      ↓
Normalize
      ↓
ToTensorV2
```

### 不要对验证集使用随机增强

验证结果应该稳定，因此验证集只进行固定缩放、裁剪和标准化。

### 增强不能改变类别

例如：

```
猫水平翻转后还是猫         → 合理
数字 6 旋转后可能变成 9    → 可能不合理
文字水平翻转后无法阅读      → 不合理
```

增强操作需要根据具体任务选择，不是越多越好。

---

## 14. 重点总结

### 核心增强流程

```
train_transform = A.Compose([
    A.RandomResizedCrop(size=(224, 224)),  # 随机裁剪
    A.HorizontalFlip(p=0.5),  # 随机翻转
    A.RandomBrightnessContrast(p=0.5),  # 随机改变亮度和对比度
    A.Normalize(),  # 标准化
    ToTensorV2(),  # 转换为 PyTorch 张量
])
```

### Dataset 中的核心写法

```
image = cv2.imread(image_path)  # 读取图片
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 转 RGB

transformed = self.transform(image=image)  # 执行增强
image = transformed["image"]  # 获取增强结果
```

### 需要记住的差异

```
Torchvision：
image = transform(image)

Albumentations：
result = transform(image=image)
image = result["image"]
```

总结：

```
Albumentations 接收 NumPy 格式的图片，
通过 Compose 组合增强操作，
最后使用 ToTensorV2 转换为 PyTorch 张量。
```


# 在 PyTorch 中处理类别不平衡数据集

## 1. 什么是类别不平衡？

类别不平衡是指不同类别包含的样本数量差异较大。

例如：

```
类别 0：900 张图片
类别 1：90 张图片
类别 2：10 张图片
```

如果直接随机加载数据，模型在训练时会频繁看到类别 `0`，很少看到类别 `2`。

这可能导致模型倾向于预测多数类别：

```
模型全部预测为类别 0
准确率仍然可以达到 90%
```

此时总体准确率看起来很高，但少数类别几乎无法被正确识别。

---

## 2. 常见解决方法

处理类别不平衡的常见方法包括：

```
过采样：增加少数类别被选中的概率
欠采样：减少多数类别被选中的概率
加权损失：让少数类别的预测错误产生更大损失
数据增强：为少数类别生成更多变化
```

本节重点介绍：

```
torch.utils.data.WeightedRandomSampler
```

它会根据每个样本的权重进行随机采样，让少数类别更容易被选中。

---

## 3. `WeightedRandomSampler` 的原理

假设训练集标签为：

```
targets = torch.tensor([
    0, 0, 0, 0, 0, 0,  # 类别 0 有 6 个样本
    1, 1,                # 类别 1 有 2 个样本
    2,                   # 类别 2 有 1 个样本
])
```

类别越少，它对应的采样权重应该越大：

```
类别权重 = 1 / 该类别的样本数量
```

因此：

```
类别 0：1 / 6
类别 1：1 / 2
类别 2：1 / 1
```

这样少数类别在训练中会更频繁地被抽到。

需要特别注意：

```
WeightedRandomSampler 需要的是每个样本的权重，
不是只包含几个数的类别权重。
```

它会根据这些权重，从数据集索引中随机抽取样本。权重不需要提前归一化为概率。[PyTorch WeightedRandomSampler 文档](https://docs.pytorch.org/docs/stable/data.html)

---

## 4. 创建采样器

```
import torch  # 导入 PyTorch
from torch.utils.data import WeightedRandomSampler  # 导入加权随机采样器


targets = torch.tensor([
    0, 0, 0, 0, 0, 0,
    1, 1,
    2,
])  # 保存训练集中每个样本的标签

class_counts = torch.bincount(targets)  # 统计每个类别的样本数量

print(class_counts)
# tensor([6, 2, 1])
```

计算类别权重：

```
class_weights = 1.0 / class_counts.float()  # 样本越少，类别权重越大

print(class_weights)
# tensor([0.1667, 0.5000, 1.0000])
```

将类别权重转换为每个样本的权重：

```
sample_weights = class_weights[targets]  # 根据每个样本的标签获取对应权重

print(sample_weights)
# tensor([
#     0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667,
#     0.5000, 0.5000,
#     1.0000
# ])
```

创建采样器：

```
sampler = WeightedRandomSampler(
    weights=sample_weights,  # 为每个样本指定采样权重
    num_samples=len(sample_weights),  # 每个 epoch 抽取的样本数量
    replacement=True,  # 允许同一个样本在一个 epoch 中被多次抽取
)
```

---

## 5. 与 DataLoader 一起使用

```
from torch.utils.data import DataLoader  # 导入数据加载器


train_loader = DataLoader(
    dataset=train_dataset,  # 指定训练数据集
    batch_size=32,  # 每个批次包含 32 个样本
    sampler=sampler,  # 使用加权随机采样器
)
```

使用 `sampler` 后不要再设置：

```
shuffle=True
```

错误写法：

```
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    shuffle=True,  # sampler 和 shuffle 不能同时使用
)
```

正确写法：

```
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
)
```

`sampler` 本身已经决定样本的抽取顺序，因此不再需要 `shuffle`。

---

## 6. 在 ImageFolder 数据集中使用

`ImageFolder` 通常可以通过 `targets` 属性获得所有样本的标签：

```
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder


train_dataset = ImageFolder(
    root="dataset/train",
    transform=train_transform,
)  # 创建图片训练集

targets = torch.tensor(
    train_dataset.targets,
    dtype=torch.long,
)  # 获取每个图片样本对应的类别标签

class_counts = torch.bincount(
    targets,
    minlength=len(train_dataset.classes),
)  # 统计每个类别的样本数量

class_weights = 1.0 / class_counts.float()  # 计算类别采样权重
sample_weights = class_weights[targets]  # 转换为每个样本的采样权重

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True,
)  # 创建加权随机采样器

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=0,
)  # 使用采样器加载训练数据

print(train_dataset.classes)  # 查看类别名称
print(class_counts)  # 查看每个类别的原始样本数量
```

---

## 7. 完整的辅助函数

可以编写一个函数，根据标签自动创建采样器：

```
import torch
from torch.utils.data import WeightedRandomSampler


def create_weighted_sampler(targets, num_classes=None):
    targets = torch.as_tensor(
        targets,
        dtype=torch.long,
    )  # 将标签转换为整数张量

    if num_classes is None:
        num_classes = int(targets.max().item()) + 1  # 自动计算类别数量

    class_counts = torch.bincount(
        targets,
        minlength=num_classes,
    )  # 统计每个类别的样本数量

    if torch.any(class_counts == 0):
        raise ValueError(
            "训练集中存在没有样本的类别，无法计算倒数权重。"
        )  # 避免除以零

    class_weights = 1.0 / class_counts.float()  # 计算类别权重
    sample_weights = class_weights[targets]  # 生成每个样本的权重

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(targets),
        replacement=True,
    )  # 创建采样器

    return sampler, class_counts, class_weights
```

使用：

```
sampler, class_counts, class_weights = create_weighted_sampler(
    train_dataset.targets,
    num_classes=len(train_dataset.classes),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
)
```

---

## 8. `replacement` 和 `num_samples`

### `replacement=True`

```
replacement=True
```

表示有放回采样。同一个样本可以在一个 epoch 中被抽取多次。

这对于类别不平衡非常重要，因为少数类别原本样本很少，只有重复抽取它们，才能使不同类别出现次数接近。

### `replacement=False`

```
replacement=False
```

表示一个样本被抽取后，不能再次被抽取。

如果一个 epoch 需要遍历整个数据集，那么少数类别样本数量本身没有增加，通常不能实现完整的类别平衡。

### `num_samples`

```
num_samples=len(train_dataset)
```

表示每个 epoch 抽取的样本总数与原始训练集大小相同。

也可以人为设置一个新的 epoch 大小：

```
num_samples=2000  # 每个 epoch 抽取 2000 个样本
```

---

## 9. 检查采样后的类别分布

创建采样器后，最好检查它是否产生了更均衡的分布：

```
sampled_counts = torch.zeros(
    len(train_dataset.classes),
    dtype=torch.long,
)  # 用于统计采样后的类别数量

for _, labels in train_loader:
    sampled_counts += torch.bincount(
        labels,
        minlength=len(train_dataset.classes),
    )  # 累加每个类别的出现次数

print("原始类别数量：", class_counts)
print("采样后类别数量：", sampled_counts)
```

由于是随机采样，结果不会完全相等，但通常会比较接近：

```
原始类别数量：tensor([900, 90, 10])
采样后类别数量：tensor([326, 338, 336])
```

`WeightedRandomSampler` 只能让整个 epoch 中的类别分布大致平衡，不能保证每个 batch 都包含完全相同数量的各类样本。

---

## 10. 加权损失函数

除了修改采样方式，还可以让损失函数更加重视少数类别。

```
import torch.nn as nn  # 导入神经网络模块


criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)  # 少数类别产生的分类错误会获得更大权重
```

这里的 `class_weights` 是类别权重：

```
class_weights = 1.0 / class_counts.float()
```

需要区分：

```
WeightedRandomSampler：
接收每个样本的权重 sample_weights

CrossEntropyLoss：
接收每个类别的权重 class_weights
```

代码对比：

```
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)  # 使用每个样本的权重

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)  # 使用每个类别的权重
```

---

## 11. 应该选择哪种方法？

### WeightedRandomSampler

优点：

- 训练时更频繁地看到少数类别
- 不需要修改损失函数
- 适合少数类别样本非常少的情况

缺点：

- 少数类别样本可能被重复抽取多次
- 可能导致模型记住少数类别样本
- 一个 epoch 不一定遍历所有多数类别样本

### 加权损失

优点：

- 不改变原始数据分布
- 每个 epoch 仍然可以遍历完整数据集
- 实现简单

缺点：

- 极端不平衡时，少数类别仍然很少出现在批次中
- 过大的类别权重可能使训练不稳定

### 两者同时使用

可以同时使用采样器和加权损失，但相当于对少数类别进行了两次加强：

```
采样时增加出现次数
+
计算损失时增加错误权重
```

初学时建议先选择一种方法：

```
少数类别出现得太少
→ 优先尝试 WeightedRandomSampler

每个类别都有一定数量，但比例不均衡
→ 可以尝试加权 CrossEntropyLoss
```

然后根据验证集结果决定是否调整。

---

## 12. 只对训练集使用采样器

加权随机采样只应该用于训练集：

```
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
)  # 训练集使用加权采样

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)  # 验证集保留真实的数据分布

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
)  # 测试集保留真实的数据分布
```

不要对验证集和测试集进行过采样，因为它们应该反映模型在真实数据分布中的表现。

---

## 13. 评价不平衡数据集

类别不平衡时，只看总体准确率可能产生误导。

除了准确率，还应该关注：

```
每个类别的准确情况
Precision
Recall
F1-score
混淆矩阵
Macro F1
Balanced Accuracy
```

尤其是：

```
Recall：有多少真正的少数类别被识别出来
Macro F1：每个类别权重相同，不会被多数类别主导
```

例如在疾病检测中，模型即使总体准确率很高，如果少数的患病样本全部预测错误，模型仍然没有实际价值。

---

## 14. 注意事项

### 权重必须与数据集索引对应

```
sample_weights[index]
```

必须对应：

```
train_dataset[index]
```

如果先计算权重，之后又重新排列或筛选数据集，就可能导致权重与样本错位。

### 只用训练集计算权重

不要使用验证集或测试集的标签分布计算训练采样权重。

### 过采样可能导致过拟合

少数类别样本会被重复看到，可以结合合理的数据增强：

```
少数类别过采样
+
随机裁剪、翻转、颜色变化
```

这样同一个少数类别样本每次被抽中时，可以产生不同的增强结果。

### 多标签任务更加复杂

上面的代码适用于每个样本只有一个类别的单标签分类：

```
一张图片 → 一个类别
```

如果一个样本可以同时属于多个类别，就不能直接使用：

```
sample_weights = class_weights[targets]
```

多标签任务需要根据样本包含的多个标签设计独立的样本权重。

---

## 15. 重点总结

### 核心代码

```
targets = torch.tensor(train_dataset.targets)  # 获取每个样本的标签

class_counts = torch.bincount(targets)  # 统计每个类别的样本数量
class_weights = 1.0 / class_counts.float()  # 计算类别权重
sample_weights = class_weights[targets]  # 为每个样本分配权重

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True,
)  # 创建加权随机采样器

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
)  # 使用采样器加载训练数据
```

### 最重要的区别

```
class_weights：
每个类别一个权重
形状为 (num_classes,)

sample_weights：
每个样本一个权重
形状为 (num_samples,)
```

```
WeightedRandomSampler 使用 sample_weights
CrossEntropyLoss 使用 class_weights
```

### 核心原则

```
类别样本越少
      ↓
对应权重越大
      ↓
被采样的概率越高
      ↓
模型在训练时更频繁地看到少数类别
```

总结：

```
WeightedRandomSampler 通过提高少数类别样本的抽取概率，
让训练数据在每个 epoch 中形成更加均衡的类别分布。
```


# PyTorch 十个常见错误：如何节省调试时间


视频总结的十个常见错误是：

```
1. 没有先让模型过拟合一个小批次
2. 忘记切换 train 和 eval 模式
3. 忘记清空梯度
4. 使用 CrossEntropyLoss 前手动添加 Softmax
5. BatchNorm 前仍然保留 bias
6. 把 view 当作 permute 使用
7. 使用不正确的数据增强
8. 没有打乱训练数据
9. 没有对输入数据进行标准化
10. 没有进行梯度裁剪
```

---

## 1. 没有先让模型过拟合一个小批次

在正式训练整个数据集之前，可以先取一个非常小的批次，反复训练这个批次。

```
images, labels = next(iter(train_loader))  # 只获取一个批次
images = images.to(device)  # 将图片移动到指定设备
labels = labels.to(device)  # 将标签移动到指定设备

for step in range(500):
    scores = model(images)  # 对同一个批次进行前向传播
    loss = criterion(scores, labels)  # 计算损失

    optimizer.zero_grad()  # 清除旧梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if step % 50 == 0:
        predictions = scores.argmax(dim=1)  # 获取预测类别
        accuracy = (predictions == labels).float().mean()  # 计算当前批次准确率

        print(
            f"Step: {step}, "
            f"Loss: {loss.item():.4f}, "
            f"Accuracy: {accuracy.item() * 100:.2f}%"
        )
```

模型应该能够记住这个小批次，使训练损失接近 `0`，准确率接近 `100%`。

如果连一个小批次都无法过拟合，通常说明代码中存在问题，例如：

- 模型结构或输出形状错误
- 损失函数使用错误
- 标签格式错误
- 参数没有交给优化器
- 忘记反向传播或更新参数
- 学习率不合适
- 数据预处理存在问题

正确调试顺序：

```
先确认模型能过拟合一个小批次
              ↓
再确认模型能过拟合一小部分数据
              ↓
最后训练完整数据集
```

这是非常有效的“基本正确性检查”。

---

## 2. 忘记切换 `train()` 和 `eval()`

PyTorch 模型有训练和评估两种模式。

训练时：

```
model.train()  # 将模型设置为训练模式
```

验证或测试时：

```
model.eval()  # 将模型设置为评估模式
```

它们主要影响：

- `Dropout`
- `BatchNorm`

### Dropout

训练模式下，Dropout 会随机丢弃部分神经元：

```
model.train()  # Dropout 正常随机丢弃神经元
```

评估模式下，Dropout 会被关闭：

```
model.eval()  # Dropout 不再随机丢弃神经元
```

### BatchNorm

训练模式下，BatchNorm 使用当前批次的统计信息，并更新运行均值和方差。

评估模式下，BatchNorm 使用训练期间保存的运行统计信息。

推荐写法：

```
for epoch in range(num_epochs):
    model.train()  # 开始训练前切换为训练模式

    for images, labels in train_loader:
        # 训练代码
        pass

    model.eval()  # 开始验证前切换为评估模式

    with torch.no_grad():
        for images, labels in val_loader:
            # 验证代码
            pass
```

需要注意：

```
model.eval()
```

不会自动关闭梯度，因此验证时还应该使用：

```
with torch.no_grad():
```

或者：

```
with torch.inference_mode():
```

---

## 3. 忘记清空梯度

PyTorch 默认会累加梯度。

如果连续调用：

```
loss.backward()
```

新的梯度会与之前的梯度相加，而不是覆盖旧梯度。

错误写法：

```
scores = model(images)
loss = criterion(scores, labels)

loss.backward()  # 梯度不断累加
optimizer.step()
```

正确写法：

```
scores = model(images)  # 前向传播
loss = criterion(scores, labels)  # 计算损失

optimizer.zero_grad()  # 清除上一个批次留下的梯度
loss.backward()  # 计算当前批次的梯度
optimizer.step()  # 更新模型参数
```

也可以使用：

```
optimizer.zero_grad(set_to_none=True)  # 将梯度设为 None，通常更加节省内存
```

完整顺序：

```
前向传播
   ↓
计算损失
   ↓
清空旧梯度
   ↓
反向传播
   ↓
更新参数
```

只有在故意进行梯度累积时，才不会每个批次都调用 `zero_grad()`。

---

## 4. 在 `CrossEntropyLoss` 前使用 Softmax

多分类任务中，常见错误是在模型输出层后手动添加 Softmax：

```
def forward(self, x):
    x = self.fc(x)
    x = torch.softmax(x, dim=1)  # 不需要
    return x
```

然后又使用：

```
criterion = nn.CrossEntropyLoss()
```

这是错误或不推荐的组合，因为 `CrossEntropyLoss` 需要接收未经 Softmax 的原始分数，也就是 logits。

正确模型：

```
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        logits = self.fc(x)  # 直接返回原始类别分数
        return logits
```

训练：

```
logits = model(images)  # 获取原始类别分数
loss = criterion(logits, labels)  # CrossEntropyLoss 内部会完成需要的计算
```

预测时可以直接获取最大分数对应的类别：

```
predictions = logits.argmax(dim=1)  # 不需要先计算 Softmax
```

只有在需要显示每个类别的概率时，才使用 Softmax：

```
probabilities = torch.softmax(logits, dim=1)  # 将 logits 转换为概率
```

类似地，二分类任务使用：

```
nn.BCEWithLogitsLoss()
```

时，也不应该在模型中提前使用 Sigmoid：

```
logits = model(images)  # 模型输出原始分数
loss = criterion(logits, labels)  # BCEWithLogitsLoss 内部包含 Sigmoid
```

---

## 5. BatchNorm 前保留不必要的 bias

卷积层和全连接层默认包含偏置：

```
nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    bias=True,
)
```

如果卷积层后面紧接着 BatchNorm，卷积层的偏置通常是不必要的。

```
self.conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    padding=1,
    bias=False,
)  # 后面紧接 BatchNorm，因此关闭 bias

self.bn = nn.BatchNorm2d(64)  # BatchNorm 自己包含可学习的平移参数
```

前向传播：

```
def forward(self, x):
    x = self.conv(x)  # 不使用卷积偏置
    x = self.bn(x)  # BatchNorm 完成标准化和可学习的缩放、平移
    x = F.relu(x)
    return x
```

原因是 BatchNorm 会对输入重新中心化，并且自身包含可学习的平移参数，因此前一层的 bias 通常会变得多余。

推荐写法：

```
nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
)
```

这不是会让模型训练失败的严重错误，但关闭多余 bias 可以减少无用参数。

---

## 6. 把 `view()` 当作 `permute()` 使用

`view()` 和 `permute()` 的作用完全不同。

### `view`

`view()` 只用于改变张量形状：

```
x = torch.rand(32, 3, 28, 28)  # 形状为 (batch, channels, height, width)

x = x.view(32, -1)  # 将每张图片展开，结果为 (32, 2352)
```

它不会交换维度所代表的含义。

### `permute`

`permute()` 用于重新排列维度：

```
x = torch.rand(32, 28, 28, 3)  # NHWC 格式

x = x.permute(0, 3, 1, 2)  # 转换为 NCHW 格式
```

形状变化：

```
(32, 28, 28, 3)
      ↓ permute(0, 3, 1, 2)
(32, 3, 28, 28)
```

错误写法：

```
x = x.view(32, 3, 28, 28)  # 只是重新解释内存，可能打乱通道和像素的对应关系
```

正确原则：

```
只改变形状 → view 或 reshape
交换维度顺序 → permute 或 transpose
```

`permute()` 后的张量可能不是连续的。如果之后需要使用 `view()`，可以先调用：

```
x = x.permute(0, 3, 1, 2)  # 交换维度
x = x.contiguous()  # 让数据在内存中重新连续排列
x = x.view(x.size(0), -1)  # 再使用 view
```

或者直接使用更灵活的：

```
x = x.permute(0, 3, 1, 2)
x = x.reshape(x.size(0), -1)
```

---

## 7. 使用不正确的数据增强

数据增强必须保持样本的真实类别和语义不变。

合理增强：

```
猫水平翻转后仍然是猫
汽车轻微裁剪后仍然是汽车
图片亮度轻微变化后类别不变
```

不合理增强：

```
数字 6 旋转 180° 后可能变成 9
文字水平翻转后无法阅读
医学图片翻转后可能改变左右位置含义
```

正确的数据增强应该根据具体任务设计：

```
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪和缩放
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
    ),  # 轻微改变亮度和对比度
    transforms.ToTensor(),
])
```

验证集不应使用随机数据增强：

```
val_transform = transforms.Compose([
    transforms.Resize(256),  # 固定缩放
    transforms.CenterCrop(224),  # 固定中心裁剪
    transforms.ToTensor(),
])
```

如果是目标检测或图像分割，图片与标签必须进行同步空间变换：

```
图片水平翻转
    ↓
边界框或分割掩码也必须水平翻转
```

否则图片和标签会失去对应关系。

---

## 8. 没有打乱训练数据

如果训练数据按照类别排列：

```
前 1000 个样本全部是猫
后 1000 个样本全部是狗
```

并且没有打乱数据，模型会连续看到大量相同类别，影响训练的稳定性。

训练集应该设置：

```
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)  # 每个 epoch 开始时打乱训练样本
```

验证集和测试集通常不需要打乱：

```
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)
```

基本原则：

```
训练集：shuffle=True
验证集：shuffle=False
测试集：shuffle=False
```

如果使用了自定义 `sampler`，则不要同时设置 `shuffle=True`：

```
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    sampler=sampler,
)  # sampler 已经决定采样顺序
```

---

## 9. 没有对输入数据进行标准化

不同特征的取值范围可能差异很大：

```
特征 1：0～1
特征 2：0～1000
特征 3：-500～500
```

较大数值的特征可能主导模型的梯度，使训练更加困难。

常见标准化公式：

```
x_normalized = (x - mean) / std
```

图片使用 ImageNet 预训练模型时，通常使用：

```
transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

训练集、验证集和测试集必须使用相同的均值和标准差：

```
训练集：使用训练集统计值
验证集：使用训练集统计值
测试集：使用训练集统计值
```

不能分别使用验证集和测试集计算自己的标准化参数，否则可能造成数据泄漏或输入分布不一致。

标准化通常可以：

- 让不同特征具有相近的数值范围
- 改善梯度传播
- 加快训练收敛
- 提高训练稳定性

需要注意，具体预训练模型可能要求特定的输入预处理，应优先使用与其预训练权重匹配的转换。

---

## 10. 没有进行梯度裁剪

RNN、LSTM 和较深网络在训练时可能出现梯度爆炸：

```
梯度越来越大
     ↓
参数更新幅度过大
     ↓
Loss 突然变得非常大
     ↓
最终出现 inf 或 NaN
```

可以在反向传播之后、参数更新之前进行梯度裁剪：

```
optimizer.zero_grad()  # 清除旧梯度
loss.backward()  # 计算梯度

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)  # 将所有参数梯度的整体范数限制在 1.0 以内

optimizer.step()  # 使用裁剪后的梯度更新参数
```

执行顺序非常重要：

```
loss.backward()
      ↓
clip_grad_norm_()
      ↓
optimizer.step()
```

错误顺序：

```
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
loss.backward()  # 此时梯度还没有计算，裁剪没有意义
```

梯度裁剪主要用于：

- RNN
- LSTM
- GRU
- Transformer
- 训练中出现梯度爆炸的模型

它不是所有模型都必须使用，但当损失突然爆炸或出现 `NaN` 时，应该检查梯度大小。

---

## 11. 一个较完整的训练模板

```
for epoch in range(num_epochs):
    model.train()  # 切换到训练模式

    for images, labels in train_loader:
        images = images.to(device)  # 移动输入数据
        labels = labels.to(device)  # 移动标签

        logits = model(images)  # 直接获取 logits，不提前使用 Softmax
        loss = criterion(logits, labels)  # 计算损失

        optimizer.zero_grad(set_to_none=True)  # 清除旧梯度
        loss.backward()  # 反向传播

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )  # 必要时进行梯度裁剪

        optimizer.step()  # 更新模型参数

    model.eval()  # 切换到评估模式
    num_correct = 0  # 记录预测正确的样本数量
    num_samples = 0  # 记录样本总数

    with torch.inference_mode():  # 验证时关闭梯度和计算图
        for images, labels in val_loader:
            images = images.to(device)  # 移动验证图片
            labels = labels.to(device)  # 移动验证标签

            logits = model(images)  # 前向传播
            predictions = logits.argmax(dim=1)  # 获取预测类别

            num_correct += (predictions == labels).sum().item()
            num_samples += labels.size(0)

    accuracy = num_correct / num_samples * 100  # 计算验证集准确率
    print(f"Epoch {epoch + 1}, Validation Accuracy: {accuracy:.2f}%")
```

---

## 12. 快速检查清单

开始长时间训练前，可以检查以下内容：

```
□ 模型能否过拟合一个小批次？
□ 训练时是否调用 model.train()？
□ 验证时是否调用 model.eval()？
□ 验证时是否关闭梯度计算？
□ 每次反向传播前是否清除了旧梯度？
□ CrossEntropyLoss 前是否错误添加了 Softmax？
□ BatchNorm 前的 Conv/Linear 是否有不必要的 bias？
□ 是否正确区分 view、reshape 和 permute？
□ 数据增强是否保持标签语义不变？
□ 训练集是否被正确打乱？
□ 输入数据是否进行了合理标准化？
□ 是否出现梯度爆炸，必要时是否进行了梯度裁剪？
```

---

## 13. 重点总结

### 训练正确性

```
先过拟合一个小批次
使用正确的损失函数输入
正确清除、计算和更新梯度
```

### 模型模式

```
model.train()  # 训练阶段
model.eval()  # 验证和测试阶段
```

### 多分类输出

```
logits = model(x)  # 不添加 Softmax
loss = nn.CrossEntropyLoss()(logits, labels)
```

### 张量形状

```
view / reshape：改变形状
permute：改变维度顺序
```

### 数据处理

```
训练集：打乱 + 合理随机增强 + 标准化
验证集：固定预处理 + 相同标准化
```

### 梯度处理

```
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

总结：

```
在训练复杂模型之前，先验证最小训练流程是否正确；
大多数浪费时间的问题，往往不是模型不够复杂，
而是模式切换、梯度、形状、损失函数或数据处理中的小错误。
```



# PyTorch TensorBoard 教程

## 1. TensorBoard 是什么？

TensorBoard 是一个用于观察和分析模型训练过程的可视化工具。

它可以显示：

- 训练损失和验证损失
- 训练集和验证集准确率
- 学习率变化
- 输入图片
- 模型计算图
- 参数和梯度分布
- 多组实验之间的对比

基本工作流程：

```
PyTorch 训练程序
      ↓
SummaryWriter 写入日志文件
      ↓
TensorBoard 读取日志
      ↓
在浏览器中显示图表
```

PyTorch 通过 `torch.utils.tensorboard.SummaryWriter` 写入 TensorBoard 日志。[PyTorch TensorBoard 文档](https://docs.pytorch.org/docs/stable/tensorboard)

---

## 2. 安装并启动 TensorBoard

安装：

```
pip install tensorboard
```

训练程序运行后，一般会在 `runs` 文件夹中生成日志文件。

启动 TensorBoard：

```
tensorboard --logdir=runs
```

然后在浏览器中打开：

```
http://localhost:6006
```

如果修改了日志目录，需要让 `--logdir` 指向对应位置。

---

## 3. 创建 `SummaryWriter`

```
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 日志写入器


writer = SummaryWriter(
    log_dir="runs/mnist_experiment"
)  # 将日志保存到指定目录
```

如果不指定目录：

```
writer = SummaryWriter()  # 默认在 runs/时间和主机名 目录中创建日志
```

不同实验应该使用不同目录：

```
writer = SummaryWriter("runs/cnn_lr_0.001")  # 记录第一组实验
writer = SummaryWriter("runs/cnn_lr_0.0001")  # 记录第二组实验
```

TensorBoard 可以同时读取这些目录并比较实验结果。

---

## 4. 记录标量数据

损失和准确率都是标量，可以使用：

```
writer.add_scalar(
    tag,
    scalar_value,
    global_step,
)
```

例如：

```
writer.add_scalar(
    "Loss/train",
    train_loss,
    epoch,
)  # 记录当前 epoch 的训练损失

writer.add_scalar(
    "Loss/validation",
    val_loss,
    epoch,
)  # 记录当前 epoch 的验证损失

writer.add_scalar(
    "Accuracy/train",
    train_accuracy,
    epoch,
)  # 记录训练集准确率

writer.add_scalar(
    "Accuracy/validation",
    val_accuracy,
    epoch,
)  # 记录验证集准确率
```

标签使用分层命名：

```
Loss/train
Loss/validation

Accuracy/train
Accuracy/validation
```

TensorBoard 会把同一组数据放在一起显示，方便比较。[PyTorch TensorBoard 使用教程](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)

---

## 5. 在每个 batch 后记录损失

如果希望观察更细致的训练变化，可以在每个 batch 后记录一次：

```
global_step = 0  # 记录模型一共完成了多少次参数更新

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.to(device)  # 将图片移动到指定设备
        labels = labels.to(device)  # 将标签移动到指定设备

        scores = model(images)  # 前向传播
        loss = criterion(scores, labels)  # 计算损失

        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        writer.add_scalar(
            "Loss/train_batch",
            loss.item(),
            global_step,
        )  # 记录当前 batch 的损失

        global_step += 1  # 参数每更新一次，步数增加 1
```

`global_step` 是横轴，通常表示：

```
batch 级记录 → 参数更新次数
epoch 级记录 → 当前 epoch 编号
```

---

## 6. 记录训练和验证指标

下面是一个比较完整的训练示例：

```
from torch.utils.tensorboard import SummaryWriter  # 导入 SummaryWriter


writer = SummaryWriter("runs/cnn_experiment")  # 创建实验日志目录
global_step = 0  # 记录训练步数

for epoch in range(num_epochs):
    model.train()  # 切换到训练模式
    train_loss_sum = 0.0  # 累加训练损失
    train_correct = 0  # 累加预测正确的样本数量
    train_samples = 0  # 累加训练样本数量

    for images, labels in train_loader:
        images = images.to(device)  # 移动训练图片
        labels = labels.to(device)  # 移动训练标签

        scores = model(images)  # 前向传播
        loss = criterion(scores, labels)  # 计算损失

        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_loss_sum += loss.item() * images.size(0)  # 累加当前批次的总损失

        predictions = scores.argmax(dim=1)  # 获取预测类别
        train_correct += (predictions == labels).sum().item()  # 累加正确数量
        train_samples += labels.size(0)  # 累加样本数量

        writer.add_scalar(
            "Loss/train_batch",
            loss.item(),
            global_step,
        )  # 记录每个批次的训练损失

        global_step += 1  # 更新全局训练步数

    train_loss = train_loss_sum / train_samples  # 计算当前 epoch 的平均训练损失
    train_accuracy = train_correct / train_samples * 100  # 计算训练准确率

    model.eval()  # 切换到评估模式
    val_loss_sum = 0.0  # 累加验证损失
    val_correct = 0  # 累加验证正确数量
    val_samples = 0  # 累加验证样本数量

    with torch.inference_mode():  # 验证时关闭梯度计算
        for images, labels in val_loader:
            images = images.to(device)  # 移动验证图片
            labels = labels.to(device)  # 移动验证标签

            scores = model(images)  # 前向传播
            loss = criterion(scores, labels)  # 计算验证损失

            val_loss_sum += loss.item() * images.size(0)  # 累加验证损失

            predictions = scores.argmax(dim=1)  # 获取预测类别
            val_correct += (predictions == labels).sum().item()  # 累加正确数量
            val_samples += labels.size(0)  # 累加样本数量

    val_loss = val_loss_sum / val_samples  # 计算平均验证损失
    val_accuracy = val_correct / val_samples * 100  # 计算验证准确率

    writer.add_scalars(
        "Loss",
        {
            "train": train_loss,
            "validation": val_loss,
        },
        epoch,
    )  # 在同一张图中记录训练和验证损失

    writer.add_scalars(
        "Accuracy",
        {
            "train": train_accuracy,
            "validation": val_accuracy,
        },
        epoch,
    )  # 在同一张图中记录训练和验证准确率

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {train_loss:.4f}, "
        f"Val Loss: {val_loss:.4f}, "
        f"Val Acc: {val_accuracy:.2f}%"
    )

writer.close()  # 训练结束后关闭日志写入器
```

通过损失曲线可以观察模型是否过拟合：

```
训练损失不断下降
验证损失先下降后上升
→ 可能出现过拟合
```

---

## 7. 记录学习率

如果使用学习率调度器，可以记录学习率变化：

```
current_lr = optimizer.param_groups[0]["lr"]  # 获取当前学习率

writer.add_scalar(
    "LearningRate",
    current_lr,
    epoch,
)  # 记录当前 epoch 的学习率
```

如果有多个参数组，可以分别记录：

```
for group_index, param_group in enumerate(optimizer.param_groups):
    writer.add_scalar(
        f"LearningRate/group_{group_index}",
        param_group["lr"],
        epoch,
    )  # 记录每个参数组的学习率
```

---

## 8. 显示图片

可以使用 `torchvision.utils.make_grid` 把一个批次中的多张图片拼接成网格：

```
from torchvision.utils import make_grid  # 导入图片网格工具


images, labels = next(iter(train_loader))  # 获取一个训练批次

image_grid = make_grid(
    images[:16],
    nrow=4,
    normalize=True,
)  # 将前 16 张图片排列成 4×4 网格

writer.add_image(
    "Training Images",
    image_grid,
    global_step=0,
)  # 将图片网格写入 TensorBoard
```

`add_image()` 默认接收：

```
(channels, height, width)
```

一个批次的图片通常是：

```
(batch_size, channels, height, width)
```

因此需要先使用 `make_grid()`，或者使用：

```
writer.add_images(
    "Training Images",
    images[:16],
    global_step=0,
)  # 直接记录一个图片批次
```

如果图片已经进行了标准化，直接显示可能会出现颜色异常。可以先反标准化，或者在 `make_grid()` 中使用 `normalize=True` 进行简单显示。

---

## 9. 显示模型计算图

TensorBoard 可以显示模型的计算过程：

```
example_images, _ = next(iter(train_loader))  # 获取一批示例输入
example_images = example_images.to(device)  # 移动到模型所在设备

writer.add_graph(
    model,
    example_images,
)  # 根据示例输入记录模型计算图
```

模型和输入必须位于同一个设备，并且输入形状必须符合模型要求。

计算图通常只需要记录一次，不要在每个 epoch 中重复调用。

---

## 10. 记录参数和梯度分布

可以使用直方图观察模型参数和梯度的变化：

```
for name, parameter in model.named_parameters():
    writer.add_histogram(
        f"Parameters/{name}",
        parameter.detach().cpu(),
        epoch,
    )  # 记录参数值的分布

    if parameter.grad is not None:
        writer.add_histogram(
            f"Gradients/{name}",
            parameter.grad.detach().cpu(),
            epoch,
        )  # 记录梯度分布
```

直方图可以帮助发现：

```
梯度全部接近 0 → 可能出现梯度消失
梯度数值非常大 → 可能出现梯度爆炸
参数长期不变化 → 参数可能没有被优化器更新
```

记录所有参数和梯度可能生成较大的日志文件，因此一般每个 epoch 记录一次，或者只记录关键层。

---

## 11. 比较多个实验

不同实验使用不同日志目录：

```
writer = SummaryWriter(
    "runs/model_cnn_lr_0.001"
)  # 记录学习率为 0.001 的实验
```

另一组实验：

```
writer = SummaryWriter(
    "runs/model_cnn_lr_0.0001"
)  # 记录学习率为 0.0001 的实验
```

启动：

```
tensorboard --logdir=runs
```

TensorBoard 会同时显示两组实验，方便比较：

- 不同学习率
- 不同模型结构
- 不同 batch size
- 不同优化器
- 不同数据增强方法

---

## 12. 记录超参数

可以把超参数与最终指标一起保存：

```
writer.add_hparams(
    {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "hidden_size": hidden_size,
    },  # 记录当前实验的超参数

    {
        "hparam/train_accuracy": train_accuracy,
        "hparam/val_accuracy": val_accuracy,
        "hparam/val_loss": val_loss,
    },  # 记录当前实验的最终指标
)
```

需要注意，`add_hparams()` 更适合在训练完成后调用一次。

---

## 13. 一个简单完整示例

```
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


writer = SummaryWriter("runs/experiment_1")  # 创建日志写入器

example_images, _ = next(iter(train_loader))  # 获取示例图片

writer.add_image(
    "Input Images",
    make_grid(example_images[:16], nrow=4, normalize=True),
    0,
)  # 记录输入图片

writer.add_graph(
    model,
    example_images.to(device),
)  # 记录模型计算图

global_step = 0  # 初始化训练步数

for epoch in range(num_epochs):
    model.train()  # 切换到训练模式

    for images, labels in train_loader:
        images = images.to(device)  # 移动图片
        labels = labels.to(device)  # 移动标签

        scores = model(images)  # 前向传播
        loss = criterion(scores, labels)  # 计算损失

        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        predictions = scores.argmax(dim=1)  # 获取预测类别
        accuracy = (
            predictions == labels
        ).float().mean() * 100  # 计算当前批次准确率

        writer.add_scalar(
            "Batch/Loss",
            loss.item(),
            global_step,
        )  # 记录损失

        writer.add_scalar(
            "Batch/Accuracy",
            accuracy.item(),
            global_step,
        )  # 记录准确率

        global_step += 1  # 更新训练步数

writer.close()  # 关闭写入器
```

运行训练代码后启动：

```
tensorboard --logdir=runs
```

---

## 14. 常见问题

### 忘记调用 `writer.close()`

训练完成后应该调用：

```
writer.close()
```

这样可以确保缓存中的日志写入文件。

也可以使用上下文管理器：

```
with SummaryWriter("runs/experiment") as writer:
    writer.add_scalar("Loss/train", loss, step)
```

离开代码块时会自动关闭。

### 每次运行使用同一个目录

如果多次实验写入同一个目录，曲线可能混在一起。

推荐：

```
runs/cnn_lr_001
runs/cnn_lr_0001
runs/resnet_adam
runs/resnet_sgd
```

### `global_step` 重复或混乱

记录 batch 指标时，使用持续增加的全局步数：

```
global_step += 1
```

不要在每个 epoch 开始时重新将它设为 `0`。

### 日志文件太大

频繁记录图片、模型参数和梯度会产生大量日志。

推荐：

```
损失：每若干 batch 记录
准确率：每个 epoch 记录
图片：训练开始时记录一次
计算图：记录一次
参数和梯度：每个 epoch 或若干 epoch 记录
```

---

## 15. 重点总结

### 创建日志写入器

```
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")
```

### 记录损失和准确率

```
writer.add_scalar("Loss/train", loss.item(), global_step)
writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
```

### 同图比较训练与验证数据

```
writer.add_scalars(
    "Loss",
    {
        "train": train_loss,
        "validation": val_loss,
    },
    epoch,
)
```

### 记录图片

```
grid = make_grid(images[:16], normalize=True)
writer.add_image("Images", grid, 0)
```

### 记录模型结构

```
writer.add_graph(model, example_input)
```

### 结束记录

```
writer.close()
```

### 启动 TensorBoard

```
tensorboard --logdir=runs
```

总结：

```
SummaryWriter 负责在训练过程中写入指标和模型信息，
TensorBoard 负责把这些日志转换成直观的曲线、图片和计算图。
```