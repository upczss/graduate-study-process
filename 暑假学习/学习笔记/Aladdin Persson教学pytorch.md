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


# PyTorch 从零实现 LeNet


LeNet 是经典的卷积神经网络之一，最早用于手写数字识别。它的结构简单，非常适合理解 CNN 的基本组成：

```
卷积 → 激活函数 → 池化
      ↓
卷积 → 激活函数 → 池化
      ↓
展平
      ↓
全连接层 → 分类结果
```

---

## 1. LeNet 网络结构

下面使用的是更现代化的 LeNet 风格实现：

```
输入图片：1 × 32 × 32
    ↓
卷积层：1 → 6，卷积核 5×5
    ↓
ReLU
    ↓
最大池化：2×2
    ↓
卷积层：6 → 16，卷积核 5×5
    ↓
ReLU
    ↓
最大池化：2×2
    ↓
展平：16 × 5 × 5 = 400
    ↓
全连接层：400 → 120
    ↓
全连接层：120 → 84
    ↓
全连接层：84 → 10
```

原始 LeNet-5 使用 `Tanh` 和平均池化；这里使用更常见的 `ReLU` 和最大池化。

---

## 2. 图像尺寸变化

LeNet 默认接收形状为：

```
(batch_size, 1, 32, 32)
```

其中：

```
batch_size：一个批次中的图片数量
1：灰度图片通道数
32 × 32：图片高度和宽度
```

尺寸变化过程：

```
输入                    (batch, 1, 32, 32)

Conv1：卷积核 5×5      (batch, 6, 28, 28)

MaxPool：2×2           (batch, 6, 14, 14)

Conv2：卷积核 5×5      (batch, 16, 10, 10)

MaxPool：2×2           (batch, 16, 5, 5)

Flatten                (batch, 16 × 5 × 5)

全连接层               (batch, 120)

全连接层               (batch, 84)

输出层                 (batch, 10)
```

因为第二次池化后是：

```
16 × 5 × 5
```

所以第一个全连接层的输入数量为：

```
16 * 5 * 5  # 400
```

---

## 3. LeNet 的完整实现

```
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块


class LeNet(nn.Module):  # 定义 LeNet 网络
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()  # 初始化 nn.Module 父类

        self.relu = nn.ReLU()  # 创建 ReLU 激活函数

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )  # 创建 2×2 最大池化层，步长为 2

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0,
        )  # 第一层卷积：(batch, 1, 32, 32) → (batch, 6, 28, 28)

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
        )  # 第二层卷积：(batch, 6, 14, 14) → (batch, 16, 10, 10)

        self.fc1 = nn.Linear(
            16 * 5 * 5,
            120,
        )  # 第一层全连接层：(batch, 400) → (batch, 120)

        self.fc2 = nn.Linear(
            120,
            84,
        )  # 第二层全连接层：(batch, 120) → (batch, 84)

        self.fc3 = nn.Linear(
            84,
            num_classes,
        )  # 输出层：(batch, 84) → (batch, 10)

    def forward(self, x):
        x = self.conv1(x)  # 执行第一次卷积
        x = self.relu(x)  # 使用 ReLU 激活
        x = self.pool(x)  # 执行第一次最大池化

        x = self.conv2(x)  # 执行第二次卷积
        x = self.relu(x)  # 使用 ReLU 激活
        x = self.pool(x)  # 执行第二次最大池化

        x = torch.flatten(
            x,
            start_dim=1,
        )  # 保留批次维度，将 (batch, 16, 5, 5) 展平为 (batch, 400)

        x = self.fc1(x)  # 通过第一层全连接层
        x = self.relu(x)  # 使用 ReLU 激活

        x = self.fc2(x)  # 通过第二层全连接层
        x = self.relu(x)  # 使用 ReLU 激活

        x = self.fc3(x)  # 得到 10 个类别的原始分数 logits
        return x  # 返回模型输出
```

---

## 4. 测试模型输出形状

```
model = LeNet()  # 创建 LeNet 模型

x = torch.randn(
    64,
    1,
    32,
    32,
)  # 创建 64 张随机灰度图片

output = model(x)  # 将图片输入模型

print(output.shape)  # 输出 torch.Size([64, 10])
```

输出形状：

```
(64, 10)
```

含义是：

```
64：当前批次中有 64 张图片
10：每张图片对应 10 个类别分数
```

对于 MNIST，10 个类别分别代表：

```
0、1、2、3、4、5、6、7、8、9
```

---

## 5. 为什么需要展平？

卷积层和池化层输出的是四维特征图：

```
(batch_size, channels, height, width)
```

但全连接层 `nn.Linear` 需要二维输入：

```
(batch_size, features)
```

因此需要执行：

```
x = torch.flatten(
    x,
    start_dim=1,
)
```

形状变化：

```
(batch, 16, 5, 5)
        ↓
(batch, 400)
```

也可以写成：

```
x = x.reshape(
    x.shape[0],
    -1,
)  # 保留 batch 维度，自动展开其余维度
```

---

## 6. 使用 MNIST 数据集

MNIST 原始图片尺寸为：

```
1 × 28 × 28
```

而上面的 LeNet 代码假设输入尺寸是：

```
1 × 32 × 32
```

因此需要在数据预处理中将 MNIST 图片缩放到 `32×32`：

```
from torchvision import datasets, transforms  # 导入数据集和图片转换工具


transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 将 MNIST 图片从 28×28 调整为 32×32
    transforms.ToTensor(),  # 将图片转换为 PyTorch 张量
])
```

加载训练集：

```
train_dataset = datasets.MNIST(
    root="dataset/",
    train=True,
    transform=transform,
    download=True,
)  # 创建训练数据集
```

加载测试集：

```
test_dataset = datasets.MNIST(
    root="dataset/",
    train=False,
    transform=transform,
    download=True,
)  # 创建测试数据集
```

创建数据加载器：

```
from torch.utils.data import DataLoader  # 导入 DataLoader


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)  # 创建训练数据加载器

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)  # 创建测试数据加载器
```

---

## 7. 训练 LeNet

```
import torch.optim as optim  # 导入优化器模块


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # 选择 GPU 或 CPU

model = LeNet().to(device)  # 创建模型并移动到指定设备

criterion = nn.CrossEntropyLoss()  # 创建多分类交叉熵损失函数

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)  # 创建 Adam 优化器
```

训练循环：

```
num_epochs = 5  # 设置训练轮数

for epoch in range(num_epochs):
    model.train()  # 切换到训练模式
    total_loss = 0.0  # 累加当前轮次的损失

    for images, labels in train_loader:
        images = images.to(device)  # 将图片移动到指定设备
        labels = labels.to(device)  # 将标签移动到指定设备

        scores = model(images)  # 前向传播，输出形状为 (batch_size, 10)
        loss = criterion(scores, labels)  # 计算预测结果与真实标签之间的损失

        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数

        total_loss += loss.item()  # 累加当前批次损失

    average_loss = total_loss / len(train_loader)  # 计算平均损失

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}"
    )  # 输出当前训练结果
```

模型输出的是 logits，因此不要在 `forward()` 中手动添加 Softmax：

```
x = self.fc3(x)  # 直接输出原始类别分数
```

因为：

```
nn.CrossEntropyLoss()
```

已经包含所需的 Softmax 相关计算。

---

## 8. 计算测试集准确率

```
def check_accuracy(loader, model):
    num_correct = 0  # 记录预测正确的图片数量
    num_samples = 0  # 记录总图片数量

    model.eval()  # 切换到评估模式

    with torch.inference_mode():  # 评估时关闭梯度计算
        for images, labels in loader:
            images = images.to(device)  # 将图片移动到指定设备
            labels = labels.to(device)  # 将标签移动到指定设备

            scores = model(images)  # 获取模型输出
            predictions = scores.argmax(dim=1)  # 获取每张图片分数最高的类别

            num_correct += (
                predictions == labels
            ).sum().item()  # 累加正确预测数量

            num_samples += labels.size(0)  # 累加当前批次图片数量

    accuracy = num_correct / num_samples * 100  # 计算百分比准确率
    return accuracy  # 返回准确率
```

调用：

```
train_accuracy = check_accuracy(
    train_loader,
    model,
)  # 计算训练集准确率

test_accuracy = check_accuracy(
    test_loader,
    model,
)  # 计算测试集准确率

print(f"Train Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")
```

---

## 9. LeNet 的核心知识点

### 卷积层提取特征

```
self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
```

第一层接收单通道灰度图片，输出 `6` 个特征图。

```
self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
```

第二层从第一层提取的特征中，进一步学习更复杂的模式。

### ReLU 引入非线性

```
x = self.relu(x)
```

ReLU 会将负数变为 `0`，正数保持不变：

```
ReLU(x) = max(0, x)
```

### 最大池化缩小特征图

```
x = self.pool(x)
```

池化层使高度和宽度减半：

```
28 × 28 → 14 × 14
10 × 10 → 5 × 5
```

### 全连接层完成分类

```
self.fc3 = nn.Linear(84, 10)
```

最后输出 `10` 个类别分数，分别对应数字 `0～9`。

---

## 10. 常见错误

### 输入图片尺寸错误

上面定义的 `fc1` 是：

```
self.fc1 = nn.Linear(16 * 5 * 5, 120)
```

因此它要求卷积部分最终输出：

```
(batch, 16, 5, 5)
```

如果直接输入 `28×28` 的 MNIST 图片，最终会变为：

```
(batch, 16, 4, 4)
```

这时 `fc1` 的输入大小不匹配。

解决方法：

```
transforms.Resize((32, 32))
```

或者将全连接层修改为：

```
self.fc1 = nn.Linear(16 * 4 * 4, 120)
```

### 忘记展平

错误写法：

```
x = self.pool(x)
x = self.fc1(x)  # 错误：fc1 不能直接接收四维特征图
```

正确写法：

```
x = self.pool(x)
x = torch.flatten(x, start_dim=1)
x = self.fc1(x)
```

### 提前使用 Softmax

错误写法：

```
x = torch.softmax(self.fc3(x), dim=1)
```

正确写法：

```
x = self.fc3(x)  # 返回 logits
```

训练时搭配：

```
criterion = nn.CrossEntropyLoss()
```

---

## 11. 重点总结

LeNet 的整体流程：

```
输入图片
(1, 32, 32)
     ↓
Conv2d：1 → 6
     ↓
ReLU + MaxPool
     ↓
Conv2d：6 → 16
     ↓
ReLU + MaxPool
     ↓
Flatten
(16 × 5 × 5 = 400)
     ↓
Linear：400 → 120
     ↓
Linear：120 → 84
     ↓
Linear：84 → 10
     ↓
输出 10 个类别的 logits
```

最核心的 `forward()`：

```
def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))  # 第一次卷积、激活和池化
    x = self.pool(self.relu(self.conv2(x)))  # 第二次卷积、激活和池化
    x = torch.flatten(x, start_dim=1)  # 展平卷积特征
    x = self.relu(self.fc1(x))  # 第一层全连接层
    x = self.relu(self.fc2(x))  # 第二层全连接层
    x = self.fc3(x)  # 输出类别 logits
    return x
```

一总结：

```
LeNet 先通过卷积和池化提取图片特征，
再将特征展平，通过全连接层输出最终的分类结果。
```



# Pytorch VGG implementation from scratch


## 1. 什么是 VGG？

VGG（Visual Geometry Group Network）是由牛津大学提出的一种经典卷积神经网络结构。

它的核心思想：

- 使用大量 **3×3 小卷积核**
- 使用多个卷积层堆叠增加网络深度
- 使用最大池化逐渐降低特征图尺寸
- 最后通过全连接层完成分类任务


VGG 的特点：

```
简单卷积块
      ↓
增加网络深度
      ↓
提取更加复杂的特征
      ↓
分类预测

```


相比 AlexNet，VGG 不再使用较大的卷积核，而是：

```
AlexNet:
11×11 convolution

VGG:
3×3 convolution + 3×3 convolution

```


多个小卷积核可以获得相同感受野，同时参数更少。


---

## 2. VGG 网络结构


经典 VGG-16 的结构：

```
Input Image
    |
    ↓
Conv Block 1
    |
    ↓
Conv Block 2
    |
    ↓
Conv Block 3
    |
    ↓
Conv Block 4
    |
    ↓
Conv Block 5
    |
    ↓
Fully Connected Layers
    |
    ↓
Classification

```


每个卷积块主要由：

```
Conv2d
    ↓
ReLU
    ↓
Conv2d
    ↓
ReLU
    ↓
MaxPool

```


组成。



---

## 3. VGG 配置方式


为了方便实现不同版本 VGG，通常使用字典保存网络结构。


例如：

```python
VGG_types = {

    "VGG11": [
        64,
        "M",
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],


    "VGG13": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ]

}

```


其中：

```
数字:
    表示 Conv2d 输出通道数量


"M":
    表示 MaxPool2d

```


例如：

```
64
↓
Conv2d 输出 64 个 feature maps


"M"
↓
执行一次最大池化

```



---

## 4. 创建卷积层


VGG 中每一个卷积层结构：

```
Conv2d
    ↓
BatchNorm
    ↓
ReLU

```


代码：

```python
import torch
import torch.nn as nn



class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(CNNBlock, self).__init__()


        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )


        self.batchnorm = nn.BatchNorm2d(
            out_channels
        )


        self.relu = nn.ReLU()


    def forward(self, x):

        x = self.conv(x)

        x = self.batchnorm(x)

        x = self.relu(x)

        return x

```


参数解释：

```
kernel_size=3

表示使用 3×3 卷积核


stride=1

表示卷积移动一步


padding=1

保持输入输出尺寸一致

```


例如：

```
Input:

32×32


经过:

3×3 Conv
padding=1


Output:

32×32

```



---

## 5. 从配置列表创建 VGG 网络


通过循环读取配置：

```python
class VGG(nn.Module):

    def __init__(self, in_channels=3, num_classes=1000):

        super(VGG, self).__init__()


        self.in_channels = in_channels


        self.conv_layers = self.create_conv_layers(
            VGG_types["VGG16"]
        )


        self.fcs = nn.Sequential(

            nn.Linear(
                512 * 7 * 7,
                4096
            ),

            nn.ReLU(),


            nn.Dropout(0.5),


            nn.Linear(
                4096,
                4096
            ),

            nn.ReLU(),


            nn.Dropout(0.5),


            nn.Linear(
                4096,
                num_classes
            )

        )


```


这里：

```
512 * 7 * 7

表示经过卷积层之后 feature map 的大小


512:

最后一个卷积层输出通道


7×7:

最后空间尺寸

```



---

## 6. 创建卷积部分


代码：

```python
def create_conv_layers(self, architecture):

    layers = []


    in_channels = self.in_channels


    for x in architecture:


        if type(x) == int:


            layers.append(

                CNNBlock(
                    in_channels,
                    x
                )

            )


            in_channels = x



        elif x == "M":


            layers.append(

                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )

            )


    return nn.Sequential(*layers)

```


逻辑：

```
遍历配置列表

        |
        ↓

遇到数字

        |
        ↓

添加 Conv Block


遇到 "M"

        |
        ↓

添加 MaxPool

```



---

## 7. Forward 前向传播


VGG 前向过程：

```python
def forward(self, x):


    x = self.conv_layers(x)


    x = x.reshape(
        x.shape[0],
        -1
    )


    x = self.fcs(x)


    return x

```


其中：

```python
x.reshape(
    x.shape[0],
    -1
)

```


作用：

将卷积输出的 feature map 展平成全连接层输入。


例如：

```
Before:

(batch,512,7,7)


After:

(batch,25088)

```



---

## 8. 创建 VGG16


```python
def test():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    model = VGG(
        in_channels=3,
        num_classes=1000
    ).to(device)


    x = torch.randn(
        1,
        3,
        224,
        224
    ).to(device)


    print(model(x).shape)



test()

```


输入：

```
Batch Size = 1

Channels = 3

Height = 224

Width = 224

```


输出：

```
torch.Size([1,1000])

```


表示：

```
模型预测 1000 个 ImageNet 类别

```



---

## 9. VGG 网络完整流程


```
Input Image
    |
    ↓
Conv3×3
    |
    ↓
BatchNorm
    |
    ↓
ReLU
    |
    ↓
Conv3×3
    |
    ↓
MaxPool
    |
    ↓
重复多个卷积 Block
    |
    ↓
Flatten
    |
    ↓
Fully Connected
    |
    ↓
Class Scores

```



---

## 10. VGG 和现代网络区别


## VGG

优点：

- 结构简单
- 容易理解
- 奠定 CNN 深度网络基础


缺点：

- 参数量巨大
- 计算量高
- 训练速度慢



## ResNet


ResNet 使用：

```
Residual Connection

```


解决深层网络训练困难问题。


因此：

```
VGG:

简单堆叠卷积


ResNet:

卷积 + Shortcut

```



---

## 11. 总结


VGG 从零实现核心步骤：

```
1. 使用列表保存网络结构

        ↓

2. 根据配置创建 Conv Block

        ↓

3. 添加 MaxPool

        ↓

4. Flatten feature map

        ↓

5. 使用 Fully Connected 分类

```


核心代码：

```python
model = VGG(
    in_channels=3,
    num_classes=1000
)

```


VGG 的核心思想：

```
小卷积核
+
更多卷积层
+
逐渐增加 channel

↓

学习更加复杂的视觉特征

```


# PyTorch 从零实现 GoogLeNet / InceptionNet

GoogLeNet（也称 Inception v1）是 2014 年提出的经典 CNN。它最重要的特点是使用 Inception 模块：让不同尺寸的卷积在同一层中并行提取特征，再将结果拼接起来。原始 GoogLeNet 有约 22 层，并在 ILSVRC 2014 中取得了很好的效果。[原始论文](https://arxiv.org/abs/1409.4842)

---

## 1. Inception 模块的核心思想

传统 CNN 在一层中通常只能选择一种卷积核，例如只使用 `3×3` 卷积。

Inception 模块同时使用多条并行分支：

```
输入特征图
    ├── 1×1 卷积
    ├── 1×1 卷积 → 3×3 卷积
    ├── 1×1 卷积 → 5×5 卷积
    └── 3×3 最大池化 → 1×1 卷积
                    ↓
            按通道维度拼接
```

不同分支关注不同尺度的信息：

```
1×1 卷积：通道信息、局部特征
3×3 卷积：中等范围特征
5×5 卷积：更大范围特征
池化分支：保留显著特征
```

---

## 2. 为什么需要 `1×1` 卷积？

`1×1` 卷积不会改变图片的高度和宽度，但可以改变通道数。

例如：

```
输入：(batch, 192, 28, 28)

1×1 卷积：192 → 16

输出：(batch, 16, 28, 28)
```

它主要有两个作用：

```
减少通道数，从而减少后续 3×3 或 5×5 卷积的计算量
引入额外的非线性，提高模型表达能力
```

例如，不先降维时：

```
192 通道 → 5×5 卷积 → 32 通道
```

先使用 `1×1` 卷积降维后：

```
192 通道 → 1×1 卷积变为 16 通道 → 5×5 卷积 → 32 通道
```

后者需要的参数和计算量更少。

---

## 3. 卷积基础模块

GoogLeNet 中经常重复使用：

```
卷积 → BatchNorm → ReLU
```

因此可以封装成一个模块：

```
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块


class ConvBlock(nn.Module):  # 定义卷积基础模块
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()  # 初始化 nn.Module 父类

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
            **kwargs,
        )  # 创建卷积层；后面有 BatchNorm，因此关闭卷积 bias

        self.batchnorm = nn.BatchNorm2d(
            out_channels
        )  # 对每个输出通道进行批量归一化

        self.relu = nn.ReLU()  # 创建 ReLU 激活函数

    def forward(self, x):
        x = self.conv(x)  # 执行卷积
        x = self.batchnorm(x)  # 执行 BatchNorm
        x = self.relu(x)  # 使用 ReLU 激活函数
        return x  # 返回处理后的特征图
```

---

## 4. 实现 Inception 模块

```
class InceptionBlock(nn.Module):  # 定义 Inception 模块
    def __init__(
        self,
        in_channels,
        out_1x1,
        red_3x3,
        out_3x3,
        red_5x5,
        out_5x5,
        out_1x1pool,
    ):
        super(InceptionBlock, self).__init__()  # 初始化父类

        self.branch1 = ConvBlock(
            in_channels,
            out_1x1,
            kernel_size=1,
        )  # 分支一：直接进行 1×1 卷积

        self.branch2 = nn.Sequential(
            ConvBlock(
                in_channels,
                red_3x3,
                kernel_size=1,
            ),  # 先通过 1×1 卷积减少通道数

            ConvBlock(
                red_3x3,
                out_3x3,
                kernel_size=3,
                padding=1,
            ),  # 再通过 3×3 卷积提取特征
        )

        self.branch3 = nn.Sequential(
            ConvBlock(
                in_channels,
                red_5x5,
                kernel_size=1,
            ),  # 先通过 1×1 卷积减少通道数

            ConvBlock(
                red_5x5,
                out_5x5,
                kernel_size=5,
                padding=2,
            ),  # 再通过 5×5 卷积提取更大范围的特征
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 池化后保持高度和宽度不变

            ConvBlock(
                in_channels,
                out_1x1pool,
                kernel_size=1,
            ),  # 使用 1×1 卷积调整池化分支的通道数
        )

    def forward(self, x):
        branch1 = self.branch1(x)  # 计算 1×1 卷积分支
        branch2 = self.branch2(x)  # 计算 1×1 → 3×3 卷积分支
        branch3 = self.branch3(x)  # 计算 1×1 → 5×5 卷积分支
        branch4 = self.branch4(x)  # 计算池化 → 1×1 卷积分支

        outputs = torch.cat(
            [branch1, branch2, branch3, branch4],
            dim=1,
        )  # 沿通道维度拼接四个分支的输出

        return outputs  # 返回拼接后的特征图
```

假设输入是：

```
(batch, 192, 28, 28)
```

并创建：

```
inception3a = InceptionBlock(
    in_channels=192,
    out_1x1=64,
    red_3x3=96,
    out_3x3=128,
    red_5x5=16,
    out_5x5=32,
    out_1x1pool=32,
)
```

四条分支输出通道数为：

```
分支一：64
分支二：128
分支三：32
分支四：32
```

拼接后：

```
64 + 128 + 32 + 32 = 256
```

最终形状为：

```
(batch, 192, 28, 28) → (batch, 256, 28, 28)
```

---

## 5. 辅助分类器

原始 GoogLeNet 在中间层加入了两个辅助分类器（Auxiliary Classifier）。

作用：

```
为中间层提供额外梯度
缓解深层网络中的梯度消失问题
训练时作为正则化的一部分
```

辅助分类器只在训练模式下使用。

```
class AuxiliaryClassifier(nn.Module):  # 定义辅助分类器
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()  # 初始化父类

        self.avgpool = nn.AvgPool2d(
            kernel_size=5,
            stride=3,
        )  # 对中间特征图进行平均池化

        self.conv = ConvBlock(
            in_channels,
            128,
            kernel_size=1,
        )  # 使用 1×1 卷积将通道数变为 128

        self.fc1 = nn.Linear(
            128 * 4 * 4,
            1024,
        )  # 将展平后的特征映射到 1024 维

        self.relu = nn.ReLU()  # 创建 ReLU 激活函数

        self.dropout = nn.Dropout(
            p=0.7
        )  # 训练时随机丢弃部分神经元

        self.fc2 = nn.Linear(
            1024,
            num_classes,
        )  # 输出类别 logits

    def forward(self, x):
        x = self.avgpool(x)  # 执行平均池化
        x = self.conv(x)  # 执行 1×1 卷积
        x = torch.flatten(
            x,
            start_dim=1,
        )  # 将特征图展平

        x = self.fc1(x)  # 通过第一个全连接层
        x = self.relu(x)  # 使用 ReLU 激活
        x = self.dropout(x)  # 使用 Dropout
        x = self.fc2(x)  # 输出辅助分类结果

        return x  # 返回辅助 logits
```

训练损失通常写成：

```
loss = (
    criterion(main_logits, labels)
    + 0.3 * criterion(aux1_logits, labels)
    + 0.3 * criterion(aux2_logits, labels)
)  # 主分类损失加上两个辅助分类损失
```

验证和推理时只使用主分类器输出。

---

## 6. 完整 GoogLeNet 实现

下面使用 `224×224` 的 RGB 图片作为输入，适合 ImageNet 风格的图像分类任务。

```
class GoogLeNet(nn.Module):  # 定义 GoogLeNet 网络
    def __init__(self, num_classes=1000, auxiliary_classifiers=True):
        super(GoogLeNet, self).__init__()  # 初始化 nn.Module 父类

        self.auxiliary_classifiers = auxiliary_classifiers  # 保存是否使用辅助分类器的设置

        self.conv1 = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )  # (batch, 3, 224, 224) → (batch, 64, 112, 112)

        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )  # (batch, 64, 112, 112) → (batch, 64, 56, 56)

        self.conv2 = ConvBlock(
            in_channels=64,
            out_channels=192,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # (batch, 64, 56, 56) → (batch, 192, 56, 56)

        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )  # (batch, 192, 56, 56) → (batch, 192, 28, 28)

        self.inception3a = InceptionBlock(
            192,
            64,
            96,
            128,
            16,
            32,
            32,
        )  # 输出通道数：64 + 128 + 32 + 32 = 256

        self.inception3b = InceptionBlock(
            256,
            128,
            128,
            192,
            32,
            96,
            64,
        )  # 输出通道数：128 + 192 + 96 + 64 = 480

        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )  # (batch, 480, 28, 28) → (batch, 480, 14, 14)

        self.inception4a = InceptionBlock(
            480,
            192,
            96,
            208,
            16,
            48,
            64,
        )  # 输出通道数：512

        self.inception4b = InceptionBlock(
            512,
            160,
            112,
            224,
            24,
            64,
            64,
        )  # 输出通道数：512

        self.inception4c = InceptionBlock(
            512,
            128,
            128,
            256,
            24,
            64,
            64,
        )  # 输出通道数：512

        self.inception4d = InceptionBlock(
            512,
            112,
            144,
            288,
            32,
            64,
            64,
        )  # 输出通道数：528

        self.inception4e = InceptionBlock(
            528,
            256,
            160,
            320,
            32,
            128,
            128,
        )  # 输出通道数：832

        self.maxpool4 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )  # (batch, 832, 14, 14) → (batch, 832, 7, 7)

        self.inception5a = InceptionBlock(
            832,
            256,
            160,
            320,
            32,
            128,
            128,
        )  # 输出通道数：832

        self.inception5b = InceptionBlock(
            832,
            384,
            192,
            384,
            48,
            128,
            128,
        )  # 输出通道数：1024

        self.avgpool = nn.AvgPool2d(
            kernel_size=7,
            stride=1,
        )  # (batch, 1024, 7, 7) → (batch, 1024, 1, 1)

        self.dropout = nn.Dropout(
            p=0.4
        )  # 在最终分类层前使用 Dropout

        self.fc = nn.Linear(
            1024,
            num_classes,
        )  # 输出类别 logits

        if self.auxiliary_classifiers:
            self.aux1 = AuxiliaryClassifier(
                in_channels=512,
                num_classes=num_classes,
            )  # 接收 inception4a 的输出

            self.aux2 = AuxiliaryClassifier(
                in_channels=528,
                num_classes=num_classes,
            )  # 接收 inception4d 的输出
        else:
            self.aux1 = None  # 不创建第一个辅助分类器
            self.aux2 = None  # 不创建第二个辅助分类器

    def forward(self, x):
        x = self.conv1(x)  # 第一层卷积
        x = self.maxpool1(x)  # 第一次最大池化

        x = self.conv2(x)  # 第二层卷积
        x = self.maxpool2(x)  # 第二次最大池化

        x = self.inception3a(x)  # 经过 Inception 3a
        x = self.inception3b(x)  # 经过 Inception 3b
        x = self.maxpool3(x)  # 第三次最大池化

        x = self.inception4a(x)  # 经过 Inception 4a

        if self.training and self.auxiliary_classifiers:
            aux1_logits = self.aux1(x)  # 训练时计算第一个辅助分类器输出

        x = self.inception4b(x)  # 经过 Inception 4b
        x = self.inception4c(x)  # 经过 Inception 4c
        x = self.inception4d(x)  # 经过 Inception 4d

        if self.training and self.auxiliary_classifiers:
            aux2_logits = self.aux2(x)  # 训练时计算第二个辅助分类器输出

        x = self.inception4e(x)  # 经过 Inception 4e
        x = self.maxpool4(x)  # 第四次最大池化

        x = self.inception5a(x)  # 经过 Inception 5a
        x = self.inception5b(x)  # 经过 Inception 5b

        x = self.avgpool(x)  # 进行平均池化，得到 (batch, 1024, 1, 1)
        x = torch.flatten(
            x,
            start_dim=1,
        )  # 展平为 (batch, 1024)

        x = self.dropout(x)  # 使用 Dropout
        main_logits = self.fc(x)  # 得到主分类器输出

        if self.training and self.auxiliary_classifiers:
            return main_logits, aux1_logits, aux2_logits  # 训练时返回三个输出

        return main_logits  # 验证和推理时只返回主分类结果
```

---

## 7. 测试模型

评估模式下，模型只返回主分类器的输出：

```
model = GoogLeNet(
    num_classes=10,
    auxiliary_classifiers=True,
)  # 创建 10 分类 GoogLeNet

model.eval()  # 切换到评估模式

x = torch.randn(
    2,
    3,
    224,
    224,
)  # 创建两张随机 RGB 图片

output = model(x)  # 前向传播

print(output.shape)  # 输出 torch.Size([2, 10])
```

训练模式下，且开启辅助分类器时，模型会返回三个结果：

```
model.train()  # 切换到训练模式

main_logits, aux1_logits, aux2_logits = model(x)  # 获取主输出和两个辅助输出

print(main_logits.shape)  # torch.Size([2, 10])
print(aux1_logits.shape)  # torch.Size([2, 10])
print(aux2_logits.shape)  # torch.Size([2, 10])
```

---

## 8. 训练时如何计算损失

```
criterion = nn.CrossEntropyLoss()  # 创建多分类交叉熵损失函数

model.train()  # 切换到训练模式

main_logits, aux1_logits, aux2_logits = model(images)  # 获取三个分类器输出

loss = (
    criterion(main_logits, labels)
    + 0.3 * criterion(aux1_logits, labels)
    + 0.3 * criterion(aux2_logits, labels)
)  # 主损失加上两个权重为 0.3 的辅助损失

optimizer.zero_grad()  # 清除旧梯度
loss.backward()  # 反向传播
optimizer.step()  # 更新模型参数
```

验证或测试时：

```
model.eval()  # 切换到评估模式

with torch.inference_mode():
    logits = model(images)  # 此时模型只返回主分类器输出
    predictions = logits.argmax(dim=1)  # 获取预测类别
```

---

## 9. GoogLeNet 的主要尺寸变化

输入为：

```
(batch, 3, 224, 224)
```

主要形状变化如下：

```
输入图片                    (batch, 3, 224, 224)

Conv 7×7，stride=2          (batch, 64, 112, 112)

MaxPool                     (batch, 64, 56, 56)

Conv 3×3                    (batch, 192, 56, 56)

MaxPool                     (batch, 192, 28, 28)

Inception 3a                (batch, 256, 28, 28)

Inception 3b                (batch, 480, 28, 28)

MaxPool                     (batch, 480, 14, 14)

Inception 4a ~ 4e           (batch, 832, 14, 14)

MaxPool                     (batch, 832, 7, 7)

Inception 5a ~ 5b           (batch, 1024, 7, 7)

Average Pool                (batch, 1024, 1, 1)

Flatten                     (batch, 1024)

Linear                      (batch, num_classes)
```

---

## 10. 常见错误

### 四个分支无法拼接

`torch.cat(..., dim=1)` 要求四个分支的高度和宽度相同。

因此，`3×3` 和 `5×5` 卷积需要设置正确的 padding：

```
kernel_size=3, padding=1  # 保持空间尺寸不变
kernel_size=5, padding=2  # 保持空间尺寸不变
```

池化分支也需要：

```
nn.MaxPool2d(
    kernel_size=3,
    stride=1,
    padding=1,
)
```

### 拼接维度错误

Inception 模块需要沿通道维度拼接：

```
torch.cat(
    [branch1, branch2, branch3, branch4],
    dim=1,
)
```

不要写成：

```
dim=0  # 错误：会尝试沿 batch 维度拼接
```

### 忘记处理辅助分类器输出

训练时如果模型返回三个输出：

```
main_logits, aux1_logits, aux2_logits = model(images)
```

就需要将辅助损失加入总损失。

验证时应先执行：

```
model.eval()
```

否则模型仍可能返回三个输出，导致验证代码报错。

### 输入尺寸不匹配

上面的网络使用：

```
nn.AvgPool2d(kernel_size=7, stride=1)
```

因此默认假设输入图片为 `224×224`，使最终特征图大小正好为 `7×7`。

如果输入尺寸不同，更灵活的写法是：

```
self.avgpool = nn.AdaptiveAvgPool2d(
    output_size=(1, 1)
)  # 无论输入空间尺寸是多少，最终都输出 1×1
```

---

## 11. 重点总结

Inception 模块的核心：

```
outputs = torch.cat(
    [
        branch_1x1,
        branch_3x3,
        branch_5x5,
        branch_pool,
    ],
    dim=1,
)  # 沿通道维度拼接四条并行分支
```

`1×1` 卷积的作用：

```
降低通道数
减少计算量
增加网络非线性
```

GoogLeNet 训练时的特点：

```
主分类器损失
+
两个辅助分类器损失
```

```
loss = main_loss + 0.3 * aux1_loss + 0.3 * aux2_loss
```

总结：

```
GoogLeNet 通过 Inception 模块并行提取不同尺度的特征，
再利用 1×1 卷积控制计算量，从而在加深网络的同时保持较高效率。
```




# PyTorch 从零实现 ResNet

ResNet（Residual Neural Network）是 2015 年提出的经典 CNN，核心创新是引入了残差连接（Skip Connection），有效解决了深层网络中的梯度消失和退化问题。原始 ResNet 有 34 层、50 层、101 层、152 层等变体，在 ILSVRC 2015 中取得了冠军。[原始论文](https://arxiv.org/abs/1512.03385)

---

## 1. 残差连接的核心思想

传统 CNN 堆叠卷积层时，随着网络加深，会出现梯度消失和退化问题（深层网络反而比浅层网络误差更大）。

ResNet 引入残差连接（跳跃连接）来解决这个问题：

```
输入 x
    ├─────────────────────────────┐
    │                             │
    ↓                             │
卷积 → BatchNorm → ReLU          │
    ↓                             │
卷积 → BatchNorm                  │
    ↓                             │
    └─────────── + ───────────────┘
                    ↓
                  ReLU
```

残差块的数学表达式：

```
F(x) = 残差映射（两个卷积层学到的特征）
输出 = F(x) + x  # 跳跃连接将输入直接加到输出上
```

如果残差映射 F(x) 趋向于 0，则输出 ≈ x，相当于学习了一个恒等映射，深层网络至少不会比浅层网络差。

---

## 2. 两种残差块

ResNet 中主要有两种残差块：

### BasicBlock（基础残差块）

用于 ResNet-18 和 ResNet-34：

```
输入 x (channels: 64)
    ├─────────────────────────────┐
    │                             │
    ↓                             │
3×3 卷积, 64 → BatchNorm → ReLU   │
    ↓                             │
3×3 卷积, 64 → BatchNorm          │
    ↓                             │
    └─────────── + ───────────────┘
                    ↓
                  ReLU
```

### Bottleneck（瓶颈残差块）

用于 ResNet-50、ResNet-101 和 ResNet-152：

```
输入 x (channels: 256)
    ├─────────────────────────────────┐
    │                                 │
    ↓                                 │
1×1 卷积, 64 → BN → ReLU             │  # 降维
    ↓                                 │
3×3 卷积, 64 → BN → ReLU             │  # 特征提取
    ↓                                 │
1×1 卷积, 256 → BN                    │  # 恢复维度
    ↓                                 │
    └─────────────── + ───────────────┘
                    ↓
                  ReLU
```

Bottleneck 的优势：用 1×1 卷积降低通道数，大幅减少 3×3 卷积的计算量。

---

## 3. 为什么需要 1×1 卷积在 Bottleneck 中？

普通 3×3 卷积直接处理 256 通道的输入：

```
输入：256 通道 × 56×56
3×3 卷积：256 → 256
参数量：256 × 256 × 3 × 3 ≈ 589,824
```

使用 Bottleneck 结构：

```
1×1 卷积：256 → 64（降维）
3×3 卷积：64 → 64
1×1 卷积：64 → 256（升维）
参数量：(256×64×1×1) + (64×64×3×3) + (64×256×1×1) ≈ 16,384 + 36,864 + 16,384 ≈ 69,632
```

Bottleneck 的参数只有普通卷积的 1/8 左右。

---

## 4. 基础卷积模块

与 GoogLeNet 类似，ResNet 也使用 `卷积 → BatchNorm → ReLU` 的组合：

```python
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块


class ConvBlock(nn.Module):  # 定义卷积基础模块
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()  # 初始化 nn.Module 父类

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
            **kwargs,
        )  # 创建卷积层；后面有 BatchNorm，因此关闭卷积 bias

        self.batchnorm = nn.BatchNorm2d(
            out_channels
        )  # 对每个输出通道进行批量归一化

        self.relu = nn.ReLU()  # 创建 ReLU 激活函数

    def forward(self, x):
        x = self.conv(x)  # 执行卷积
        x = self.batchnorm(x)  # 执行 BatchNorm
        x = self.relu(x)  # 使用 ReLU 激活函数
        return x  # 返回处理后的特征图
```

---

## 5. 实现 BasicBlock

```python
class BasicBlock(nn.Module):  # 定义基础残差块
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()  # 初始化父类

        # 第一个卷积层
        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )  # 当 stride > 1 时，特征图尺寸减半

        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )  # 这里不使用 ConvBlock，因为后面要加跳跃连接再做 ReLU

        self.batchnorm2 = nn.BatchNorm2d(
            out_channels
        )  # 第二层卷积后的 BatchNorm

        # 跳跃连接处理
        self.skip = nn.Sequential()  # 默认恒等映射

        if stride != 1 or in_channels != out_channels:
            # 如果 stride != 1 或通道数变化，需要调整跳跃连接
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),  # 使用 1×1 卷积调整通道数和尺寸

                nn.BatchNorm2d(
                    out_channels
                ),  # 跳跃连接也进行 BatchNorm
            )

        self.relu = nn.ReLU()  # 创建最终的 ReLU 激活函数

    def forward(self, x):
        identity = x  # 保存输入作为恒等映射

        x = self.conv1(x)  # 通过第一个卷积块
        x = self.conv2(x)  # 通过第二个卷积
        x = self.batchnorm2(x)  # 进行 BatchNorm

        identity = self.skip(identity)  # 调整跳跃连接（如果需要）

        x = x + identity  # 残差连接：输出 = 卷积输出 + 输入

        x = self.relu(x)  # 使用 ReLU 激活

        return x  # 返回残差块输出
```

---

## 6. 实现 Bottleneck

```python
class Bottleneck(nn.Module):  # 定义瓶颈残差块
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()  # 初始化父类

        # 1×1 卷积：降维
        self.conv1 = ConvBlock(
            in_channels,
            out_channels // 4,
            kernel_size=1,
        )  # 将通道数减少到原来的 1/4

        # 3×3 卷积：特征提取
        self.conv2 = ConvBlock(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            stride=stride,
            padding=1,
        )  # 在降维后的低维空间进行 3×3 卷积

        # 1×1 卷积：升维恢复通道数
        self.conv3 = nn.Conv2d(
            out_channels // 4,
            out_channels,
            kernel_size=1,
            bias=False,
        )  # 将通道数恢复为 out_channels

        self.batchnorm3 = nn.BatchNorm2d(
            out_channels
        )  # 第三层卷积后的 BatchNorm

        # 跳跃连接处理
        self.skip = nn.Sequential()  # 默认恒等映射

        if stride != 1 or in_channels != out_channels:
            # 如果 stride != 1 或通道数变化，需要调整跳跃连接
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),  # 使用 1×1 卷积调整通道数和尺寸

                nn.BatchNorm2d(
                    out_channels
                ),  # 跳跃连接也进行 BatchNorm
            )

        self.relu = nn.ReLU()  # 创建最终的 ReLU 激活函数

    def forward(self, x):
        identity = x  # 保存输入作为恒等映射

        x = self.conv1(x)  # 1×1 卷积降维
        x = self.conv2(x)  # 3×3 卷积特征提取
        x = self.conv3(x)  # 1×1 卷积升维
        x = self.batchnorm3(x)  # 进行 BatchNorm

        identity = self.skip(identity)  # 调整跳跃连接（如果需要）

        x = x + identity  # 残差连接：输出 = 卷积输出 + 输入

        x = self.relu(x)  # 使用 ReLU 激活

        return x  # 返回残差块输出
```

---

## 7. ResNet 的整体结构

ResNet 整体结构如下（以 ResNet-34 为例）：

```
输入 (batch, 3, 224, 224)
    ↓
7×7 卷积，64，stride=2
    ↓
3×3 最大池化，stride=2
    ↓
BasicBlock × 3，64 通道
    ↓
BasicBlock × 4，128 通道，stride=2（第一次下采样）
    ↓
BasicBlock × 6，256 通道，stride=2（第一次下采样）
    ↓
BasicBlock × 3，512 通道，stride=2（第一次下采样）
    ↓
全局平均池化
    ↓
全连接层 → 输出 (batch, num_classes)
```

对于 ResNet-50 及以上，将 BasicBlock 替换为 Bottleneck。

---

## 8. 完整 ResNet 实现

```python
class ResNet(nn.Module):  # 定义 ResNet 网络
    def __init__(
        self,
        block,  # 使用的残差块类型（BasicBlock 或 Bottleneck）
        layers,  # 每个阶段的层数列表，如 [3, 4, 6, 3]
        num_classes=1000,
        in_channels=3,
    ):
        super(ResNet, self).__init__()  # 初始化 nn.Module 父类

        self.in_channels = 64  # 初始卷积输出通道数

        # 初始卷积层
        self.conv1 = ConvBlock(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
        )  # (batch, 3, 224, 224) → (batch, 64, 112, 112)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )  # (batch, 64, 112, 112) → (batch, 64, 56, 56)

        # 四个阶段的残差块
        self.layer1 = self._make_layer(
            block,
            out_channels=64,
            num_blocks=layers[0],
            stride=1,
        )  # (batch, 64, 56, 56) → (batch, 64, 56, 56)

        self.layer2 = self._make_layer(
            block,
            out_channels=128,
            num_blocks=layers[1],
            stride=2,
        )  # (batch, 64, 56, 56) → (batch, 128, 28, 28)

        self.layer3 = self._make_layer(
            block,
            out_channels=256,
            num_blocks=layers[2],
            stride=2,
        )  # (batch, 128, 28, 28) → (batch, 256, 14, 14)

        self.layer4 = self._make_layer(
            block,
            out_channels=512,
            num_blocks=layers[3],
            stride=2,
        )  # (batch, 256, 14, 14) → (batch, 512, 7, 7)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )  # 无论输入空间尺寸是多少，最终都输出 1×1

        # 全连接分类层
        self.fc = nn.Linear(
            512 * 4 if block == Bottleneck else 512,
            num_classes,
        )  # Bottleneck 的输出通道是普通 block 的 4 倍

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 创建由多个残差块组成的层

        layers = []  # 存储所有残差块的列表

        # 第一个残差块（可能需要下采样）
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
            )
        )  # 添加第一个残差块

        self.in_channels = out_channels  # 更新当前通道数

        # 后续残差块（保持通道数和尺寸不变）
        for _ in range(num_blocks - 1):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    stride=1,
                )
            )  # 添加剩余残差块，stride=1 保持尺寸不变

        return nn.Sequential(*layers)  # 返回所有残差块的顺序容器

    def forward(self, x):
        x = self.conv1(x)  # 初始卷积
        x = self.maxpool(x)  # 最大池化

        x = self.layer1(x)  # 第一阶段残差块
        x = self.layer2(x)  # 第二阶段残差块
        x = self.layer3(x)  # 第三阶段残差块
        x = self.layer4(x)  # 第四阶段残差块

        x = self.avgpool(x)  # 全局平均池化 (batch, 512, 1, 1)
        x = torch.flatten(
            x,
            start_dim=1,
        )  # 展平为 (batch, 512)

        x = self.fc(x)  # 全连接层得到类别 logits

        return x  # 返回分类结果
```

---

## 9. 创建不同版本的 ResNet

```python
def resnet18(num_classes=1000):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes,
    )  # ResNet-18：每个阶段 2 个 BasicBlock


def resnet34(num_classes=1000):
    return ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes,
    )  # ResNet-34：每个阶段分别为 3、4、6、3 个 BasicBlock


def resnet50(num_classes=1000):
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
    )  # ResNet-50：每个阶段 3、4、6、3 个 Bottleneck


def resnet101(num_classes=1000):
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        num_classes,
    )  # ResNet-101：每个阶段 3、4、23、3 个 Bottleneck


def resnet152(num_classes=1000):
    return ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        num_classes,
    )  # ResNet-152：每个阶段 3、8、36、3 个 Bottleneck
```

---

## 10. 测试模型

```python
model = resnet34(
    num_classes=10,
)  # 创建 10 分类的 ResNet-34

x = torch.randn(
    2,
    3,
    224,
    224,
)  # 创建两张随机 RGB 图片

output = model(x)  # 前向传播

print(output.shape)  # 输出 torch.Size([2, 10])
```

---

## 11. ResNet 的尺寸变化

输入为 `(batch, 3, 224, 224)`，使用 ResNet-34：

```
输入图片                    (batch, 3, 224, 224)

Conv 7×7，stride=2          (batch, 64, 112, 112)

MaxPool 3×3，stride=2       (batch, 64, 56, 56)

Layer1 × 3 (stride=1)       (batch, 64, 56, 56)

Layer2 × 4 (stride=2)       (batch, 128, 28, 28)

Layer3 × 6 (stride=2)       (batch, 256, 14, 14)

Layer4 × 3 (stride=2)       (batch, 512, 7, 7)

AdaptiveAvgPool             (batch, 512, 1, 1)

Flatten                     (batch, 512)

Linear                      (batch, num_classes)
```

---

## 12. 常见错误

### 跳跃连接的尺寸不匹配

当 `stride=2` 或通道数变化时，跳跃连接需要调整：

```python
# 正确写法
self.skip = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
    nn.BatchNorm2d(out_channels),
)

# 错误写法：直接相加会导致尺寸不匹配
# x = x + identity  # 通道数或空间尺寸不同，无法相加
```

### 残差块最后的激活函数位置

残差块的正确结构：

```python
x = conv1(x)      # 卷积 + BN + ReLU
x = conv2(x)      # 卷积 + BN
x = x + identity  # 残差连接
x = relu(x)       # 放在最后
```

错误写法：

```python
x = conv1(x)  # 卷积 + BN + ReLU
x = conv2(x)  # 卷积 + BN + ReLU
x = x + identity  # 错误：跳跃连接前已经经过了 ReLU
```

### Bottleneck 的输出通道计算

Bottleneck 的输出通道是 `out_channels`，中间层是 `out_channels // 4`：

```python
# 正确
self.conv1 = ConvBlock(in_channels, out_channels // 4, kernel_size=1)
self.conv2 = ConvBlock(out_channels // 4, out_channels // 4, kernel_size=3)
self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)

# 错误
self.conv1 = ConvBlock(in_channels, out_channels // 2, kernel_size=1)  # 错误的比例
```

### 初始化 `self.in_channels`

在 `_make_layer` 方法中，需要正确更新 `self.in_channels`：

```python
def _make_layer(self, block, out_channels, num_blocks, stride):
    layers = []
    layers.append(block(self.in_channels, out_channels, stride))
    self.in_channels = out_channels  # 必须更新
    
    for _ in range(num_blocks - 1):
        layers.append(block(self.in_channels, out_channels, stride=1))
    
    return nn.Sequential(*layers)
```

如果忘记更新，后续残差块的输入通道数会错误。

---

## 13. 重点总结

残差块的核心公式：

```
输出 = F(x) + x  # F(x) 是残差映射，x 是跳跃连接
```

跳跃连接的作用：

```
缓解梯度消失问题
允许网络学习恒等映射
使深层网络的训练更加稳定
```

BasicBlock vs Bottleneck：

```
BasicBlock：两个 3×3 卷积，用于 ResNet-18/34
Bottleneck：1×1→3×3→1×1 结构，用于 ResNet-50/101/152
```

下采样处理：

```
当 stride=2 或通道数变化时
跳跃连接使用 1×1 卷积调整尺寸和通道数
```

```
输出 = 卷积输出 + 跳跃连接调整后的输入
```

总结：

```
ResNet 通过残差连接让梯度可以绕过卷积层直接传播，
从而允许网络达到上百层的深度而不出现退化问题。
Bottleneck 结构通过 1×1 卷积降维，大幅减少了计算量。
```


# PyTorch 从零实现 EfficientNet

EfficientNet 是 2019 年提出的高效 CNN 架构，核心创新是 **复合缩放方法（Compound Scaling）**：统一地缩放网络的深度、宽度和输入分辨率，在提升精度的同时保持计算效率。EfficientNet-B7 在 ImageNet 上达到 84.3% Top-1 精度，参数量仅为 GPipe 的 1/8.4，推理速度快 6.1 倍。 [原始论文](https://arxiv.org/abs/1905.11946)

---

## 1. 复合缩放的核心思想

传统的网络缩放通常只调整深度、宽度、分辨率中的一个维度。EfficientNet 提出这三个维度是相互关联的，需要协同调整：

```
深度（depth）：网络层数，越深特征越丰富，但梯度消失风险也越高
宽度（width）：通道数，越宽特征粒度越细，但过宽难以捕捉高层特征
分辨率（resolution）：输入图像尺寸，越高细节越多，但计算量剧增
```

复合缩放使用一个系数 φ 统一缩放三个维度：

```
depth:   d = α^φ
width:   w = β^φ
resolution: r = γ^φ

约束条件：α · β² · γ² ≈ 2, 且 α ≥ 1, β ≥ 1, γ ≥ 1
```

这里的 α、β、γ 通过小规模网格搜索确定。EfficientNet-B0 的 α=1.2, β=1.1, γ=1.15。

下面的图展示了四种缩放方式的区别：图中 (a) 是基线网络，(b)(c)(d) 分别单独增加宽度、深度、分辨率，(e) 是复合缩放的效果。

![](../图片/Pasted%20image%2020260811175728.png)

---

## 2. MBConv 模块

EfficientNet 的基础构建块是 **MBConv**（Mobile Inverted Bottleneck Conv），源自 MobileNetV3。 它包含以下关键组件：

### 2.1 深度可分离卷积（Depthwise Separable Convolution）

```
标准卷积：
输入 (H, W, C_in) → 卷积核 (K×K×C_in×C_out) → 输出 (H, W, C_out)

深度可分离卷积：
步骤一：逐通道卷积（Depthwise Conv）
输入 (H, W, C) → 每个通道单独卷积 → 输出 (H, W, C)
步骤二：逐点卷积（Pointwise Conv，即 1×1）
输入 (H, W, C) → 1×1 卷积 → 输出 (H, W, C_out)
参数约减少为原来的 1/K²
```

### 2.2 反向残差结构（Inverted Residual）

与 ResNet 的残差块相反：

```
ResNet 残差块：通道数先降后升（沙漏形）
  输入 256 → 1×1 降维到 64 → 3×3 → 1×1 升维到 256

MBConv 反向残差：通道数先升后降（纺锤形）
  输入 32 → 1×1 升维到 192（expand_ratio=6）→ 3×3 深度卷积 → 1×1 降维到 16
```

反向残差在高维空间进行深度卷积，可以提取更丰富的特征。

### 2.3 Squeeze-and-Excitation（SE）注意力

SE 模块通过全局池化 + 全连接层学习每个通道的重要性权重：

```
输入特征图 (H, W, C)
    ↓
全局平均池化 → (1, 1, C)
    ↓
全连接层（降维）→ ReLU
    ↓
全连接层（升维）→ Sigmoid → 通道权重 (1, 1, C)
    ↓
输入 × 通道权重 → 加权后的输出
```

### 2.4 Swish 激活函数

EfficientNet 使用 Swish（也称 SiLU）激活函数：

```
Swish(x) = x · Sigmoid(x) = x / (1 + e^(-x))
```

Swish 是平滑的非单调激活函数，在深层网络中表现优于 ReLU。

---

## 3. MBConv 完整实现

```python
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口


class Swish(nn.Module):  # 定义 Swish 激活函数
    def forward(self, x):
        return x * torch.sigmoid(x)  # Swish(x) = x * sigmoid(x)


class ConvBlock(nn.Module):  # 定义基础卷积模块（卷积 + BN + Swish）
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,  # groups=in_channels 时为深度卷积
            bias=False,
        )  # 创建卷积层

        self.bn = nn.BatchNorm2d(
            out_channels
        )  # 批量归一化

        self.activation = Swish()  # 使用 Swish 激活

    def forward(self, x):
        x = self.conv(x)  # 执行卷积
        x = self.bn(x)  # 执行 BatchNorm
        x = self.activation(x)  # 使用 Swish 激活
        return x


class SqueezeExcitation(nn.Module):  # 定义 SE 注意力模块
    def __init__(self, in_channels, reduction_ratio=24):
        super(SqueezeExcitation, self).__init__()

        # 确保缩减后的通道数至少为 1
        squeezed_channels = max(1, in_channels // reduction_ratio)

        # Squeeze: 全局平均池化
        self.squeeze = nn.AdaptiveAvgPool2d(
            output_size=1
        )  # 将每个特征图压缩为单个数值

        # Excitation: 两个全连接层
        self.fc1 = nn.Conv2d(
            in_channels,
            squeezed_channels,
            kernel_size=1,
        )  # 降维层（用 1×1 卷积代替全连接）

        self.fc2 = nn.Conv2d(
            squeezed_channels,
            in_channels,
            kernel_size=1,
        )  # 升维层（恢复通道数）

        self.activation = Swish()  # Swish 激活用于第一层

        self.sigmoid = nn.Sigmoid()  # Sigmoid 输出通道权重

    def forward(self, x):
        # x: (batch, channels, H, W)
        out = self.squeeze(x)  # (batch, channels, 1, 1)

        out = self.fc1(out)  # (batch, channels/24, 1, 1)
        out = self.activation(out)  # Swish 激活

        out = self.fc2(out)  # (batch, channels, 1, 1)
        out = self.sigmoid(out)  # Sigmoid 得到通道权重

        return x * out  # 将权重乘以原始特征图


class MBConvBlock(nn.Module):  # 定义 MBConv 模块
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        se_ratio=24,
    ):
        super(MBConvBlock, self).__init__()

        self.stride = stride  # 保存步长用于判断是否下采样
        self.use_residual = (stride == 1 and in_channels == out_channels)  # 只有当步长为 1 且通道数不变时才使用残差连接

        # 扩展阶段（Expansion Phase）
        expanded_channels = in_channels * expand_ratio  # 扩展后的通道数

        if expand_ratio != 1:
            # 用 1×1 卷积扩展通道数
            self.expand_conv = ConvBlock(
                in_channels,
                expanded_channels,
                kernel_size=1,
            )  # 输入 → 扩展通道
        else:
            self.expand_conv = nn.Identity()  # 如果扩展倍率为 1，跳过扩展层

        # 深度卷积阶段（Depthwise Convolution Phase）
        padding = kernel_size // 2  # 保持空间尺寸不变
        self.depthwise_conv = ConvBlock(
            expanded_channels,
            expanded_channels,
            kernel_size,
            stride,
            padding=padding,
            groups=expanded_channels,  # groups=通道数，实现逐通道卷积
        )  # 每个通道独立卷积

        # SE 注意力阶段
        self.se = SqueezeExcitation(
            expanded_channels,
            se_ratio,
        )  # 学习通道重要性权重

        # 压缩阶段（Projection Phase）
        self.project_conv = nn.Conv2d(
            expanded_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )  # 1×1 卷积降维到输出通道数

        self.project_bn = nn.BatchNorm2d(
            out_channels
        )  # 输出前进行 BatchNorm（不使用激活）

    def forward(self, x):
        residual = x  # 保存输入用于残差连接

        # 1. 扩展阶段
        x = self.expand_conv(x)  # 扩大通道数

        # 2. 深度卷积阶段
        x = self.depthwise_conv(x)  # 逐通道卷积

        # 3. SE 注意力阶段
        x = self.se(x)  # 通道注意力加权

        # 4. 压缩阶段
        x = self.project_conv(x)  # 降维到输出通道数
        x = self.project_bn(x)  # BatchNorm（无激活）

        # 5. 残差连接
        if self.use_residual:
            x = x + residual  # 输出 = 主路径输出 + 输入（残差连接）

        return x
```

---

## 4. EfficientNet-B0 整体结构

EfficientNet-B0 作为基线模型，结构如下：

| Stage | 操作 | 输入通道 | 输出通道 | 卷积核 | 步长 | 扩展倍率 | 重复次数 |
|-------|------|---------|---------|--------|------|---------|---------|
| 1 | Conv3×3 | 3 | 32 | 3 | 2 | - | 1 |
| 2 | MBConv | 32 | 16 | 3 | 1 | 1 | 1 |
| 3 | MBConv | 16 | 24 | 3 | 2 | 6 | 2 |
| 4 | MBConv | 24 | 40 | 5 | 2 | 6 | 2 |
| 5 | MBConv | 40 | 80 | 3 | 2 | 6 | 3 |
| 6 | MBConv | 80 | 112 | 5 | 1 | 6 | 3 |
| 7 | MBConv | 112 | 192 | 5 | 2 | 6 | 4 |
| 8 | MBConv | 192 | 320 | 3 | 1 | 6 | 1 |
| 9 | Conv1×1 + Pool + FC | 320 | 1280 | 1 | 1 | - | 1 |

---

## 5. 完整 EfficientNet 实现

```python
class EfficientNet(nn.Module):  # 定义 EfficientNet 网络
    def __init__(
        self,
        width_multiplier=1.0,  # 宽度缩放系数
        depth_multiplier=1.0,  # 深度缩放系数
        resolution=224,  # 输入分辨率
        num_classes=1000,  # 分类数
        in_channels=3,
    ):
        super(EfficientNet, self).__init__()

        # MBConv 配置：[重复次数, 输入通道, 输出通道, 卷积核, 步长, 扩展倍率]
        # 格式: (repeat, in_ch, out_ch, kernel, stride, expand_ratio)
        config = [
            [1, 32, 16, 3, 1, 1],   # Stage 1
            [2, 16, 24, 3, 2, 6],   # Stage 2
            [2, 24, 40, 5, 2, 6],   # Stage 3
            [3, 40, 80, 3, 2, 6],   # Stage 4
            [3, 80, 112, 5, 1, 6],  # Stage 5
            [4, 112, 192, 5, 2, 6], # Stage 6
            [1, 192, 320, 3, 1, 6], # Stage 7
        ]

        # 根据缩放系数调整通道数和重复次数
        self._round_channels = lambda x: int(
            (x * width_multiplier + 0.5) // 4 * 4
        )  # 通道数取整到 4 的倍数
        self._round_repeats = lambda x: int(
            x * depth_multiplier
        )  # 重复次数取整

        # Stage 1: 初始卷积层
        init_channels = self._round_channels(32)
        self.conv1 = ConvBlock(
            in_channels,
            init_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )  # (batch, 3, 224, 224) → (batch, 32, 112, 112)

        # Stage 2-8: MBConv 块
        self.blocks = nn.Sequential()  # 顺序容器

        in_channels = init_channels  # 当前输入通道数

        for repeat, in_ch, out_ch, kernel, stride, expand in config:
            # 根据缩放系数调整通道数
            in_ch = self._round_channels(in_ch)
            out_ch = self._round_channels(out_ch)
            repeat = self._round_repeats(repeat)

            # 创建该阶段的所有 MBConv 块
            for i in range(repeat):
                # 只有第一个 MBConv 块使用指定的 stride（可能下采样）
                # 后续 MBConv 块 stride=1
                block_stride = stride if i == 0 else 1

                self.blocks.append(
                    MBConvBlock(
                        in_channels,
                        out_ch,
                        kernel,
                        block_stride,
                        expand,
                    )
                )  # 添加 MBConv 块

                in_channels = out_ch  # 更新当前输入通道数

        # Stage 9: 输出层
        final_channels = self._round_channels(1280)
        self.conv2 = ConvBlock(
            in_channels,
            final_channels,
            kernel_size=1,
        )  # (batch, 320, 7, 7) → (batch, 1280, 7, 7)

        # 全局平均池化 + 分类层
        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=1
        )  # (batch, 1280, 7, 7) → (batch, 1280, 1, 1)

        self.fc = nn.Linear(
            final_channels,
            num_classes,
        )  # (batch, 1280) → (batch, num_classes)

        # 保存分辨率以便外部使用
        self.resolution = resolution

    def forward(self, x):
        x = self.conv1(x)  # 初始卷积
        x = self.blocks(x)  # MBConv 块序列
        x = self.conv2(x)  # 最终 1×1 卷积

        x = self.avgpool(x)  # 全局平均池化
        x = torch.flatten(x, start_dim=1)  # 展平
        x = self.fc(x)  # 全连接分类

        return x
```

---

## 6. 创建不同版本的 EfficientNet

不同版本通过缩放系数 φ 确定：

```python
def efficientnet_b0(num_classes=1000):
    return EfficientNet(
        width_multiplier=1.0,
        depth_multiplier=1.0,
        resolution=224,
        num_classes=num_classes,
    )  # EfficientNet-B0


def efficientnet_b1(num_classes=1000):
    return EfficientNet(
        width_multiplier=1.0,
        depth_multiplier=1.1,
        resolution=240,
        num_classes=num_classes,
    )  # EfficientNet-B1


def efficientnet_b2(num_classes=1000):
    return EfficientNet(
        width_multiplier=1.1,
        depth_multiplier=1.2,
        resolution=260,
        num_classes=num_classes,
    )  # EfficientNet-B2


def efficientnet_b3(num_classes=1000):
    return EfficientNet(
        width_multiplier=1.2,
        depth_multiplier=1.4,
        resolution=300,
        num_classes=num_classes,
    )  # EfficientNet-B3


def efficientnet_b4(num_classes=1000):
    return EfficientNet(
        width_multiplier=1.4,
        depth_multiplier=1.8,
        resolution=380,
        num_classes=num_classes,
    )  # EfficientNet-B4


def efficientnet_b5(num_classes=1000):
    return EfficientNet(
        width_multiplier=1.6,
        depth_multiplier=2.2,
        resolution=456,
        num_classes=num_classes,
    )  # EfficientNet-B5


def efficientnet_b6(num_classes=1000):
    return EfficientNet(
        width_multiplier=1.8,
        depth_multiplier=2.6,
        resolution=528,
        num_classes=num_classes,
    )  # EfficientNet-B6


def efficientnet_b7(num_classes=1000):
    return EfficientNet(
        width_multiplier=2.0,
        depth_multiplier=3.1,
        resolution=600,
        num_classes=num_classes,
    )  # EfficientNet-B7
```

不同版本的缩放参数如下：

| 模型 | 宽度系数 | 深度系数 | 分辨率 | Top-1 精度 |
|------|---------|---------|--------|-----------|
| B0   | 1.0     | 1.0     | 224    | 77.1%     |
| B1   | 1.0     | 1.1     | 240    | 79.1%     |
| B2   | 1.1     | 1.2     | 260    | 80.1%     |
| B3   | 1.2     | 1.4     | 300    | 81.6%     |
| B4   | 1.4     | 1.8     | 380    | 82.9%     |
| B5   | 1.6     | 2.2     | 456    | 83.6%     |
| B6   | 1.8     | 2.6     | 528    | 84.0%     |
| B7   | 2.0     | 3.1     | 600    | 84.3%     |

---

## 7. 测试模型

```python
model = efficientnet_b0(
    num_classes=10,
)  # 创建 10 分类的 EfficientNet-B0

x = torch.randn(
    2,
    3,
    224,
    224,
)  # 创建两张随机 RGB 图片

output = model(x)  # 前向传播

print(output.shape)  # 输出 torch.Size([2, 10])
```

---

## 8. EfficientNet 的尺寸变化

输入为 `(batch, 3, 224, 224)`，使用 EfficientNet-B0：

```
输入图片                    (batch, 3, 224, 224)

Conv3×3，stride=2          (batch, 32, 112, 112)

Stage 1 MBConv×1           (batch, 16, 112, 112)

Stage 2 MBConv×2           (batch, 24, 56, 56)

Stage 3 MBConv×2           (batch, 40, 28, 28)

Stage 4 MBConv×3           (batch, 80, 14, 14)

Stage 5 MBConv×3           (batch, 112, 14, 14)

Stage 6 MBConv×4           (batch, 192, 7, 7)

Stage 7 MBConv×1           (batch, 320, 7, 7)

Conv1×1                    (batch, 1280, 7, 7)

AdaptiveAvgPool            (batch, 1280, 1, 1)

Flatten                    (batch, 1280)

Linear                     (batch, num_classes)
```

---

## 9. 常见错误

### 扩展倍率为 1 时的处理

当 `expand_ratio = 1` 时，不需要扩展层，应使用 `nn.Identity()`：

```python
# 正确写法
if expand_ratio != 1:
    self.expand_conv = ConvBlock(in_channels, expanded_channels, kernel_size=1)
else:
    self.expand_conv = nn.Identity()

# 错误写法：即使 expand_ratio=1 也添加卷积层，浪费计算
self.expand_conv = ConvBlock(in_channels, expanded_channels, kernel_size=1)
```

### 深度卷积的 groups 参数

深度卷积需要为每个通道独立卷积：

```python
# 正确写法
self.depthwise_conv = ConvBlock(
    expanded_channels,
    expanded_channels,
    kernel_size,
    groups=expanded_channels,  # groups = 输入通道数 = 输出通道数
)

# 错误写法：group 数不是通道数，变成了普通卷积
self.depthwise_conv = ConvBlock(
    expanded_channels,
    expanded_channels,
    kernel_size,
    groups=1,  # 错误！这是标准卷积
)
```

### 残差连接的条件

MBConv 只在步长为 1 且输入输出通道数相同时使用残差连接：

```python
# 正确写法
self.use_residual = (stride == 1 and in_channels == out_channels)

# 错误写法：无条件使用残差连接会导致通道数或尺寸不匹配
self.use_residual = True
```

### 通道数取整规则

EfficientNet 要求通道数缩放后取整到 4 的倍数（硬件优化）：

```python
# 正确写法
self._round_channels = lambda x: int((x * width_multiplier + 0.5) // 4 * 4)

# 错误写法：直接取整，可能不是 4 的倍数
self._round_channels = lambda x: int(x * width_multiplier)
```

---

## 10. 重点总结

复合缩放公式：

```
depth:   d = α^φ
width:   w = β^φ
resolution: r = γ^φ
约束：α · β² · γ² ≈ 2
```

MBConv 结构（反向残差 + 深度卷积 + SE）：

```
输入
  ↓（可选扩展）1×1 卷积，通道数 × expand_ratio
  ↓ Depthwise Conv（逐通道卷积）
  ↓ SE 注意力（通道加权）
  ↓ 1×1 卷积，通道数降为 out_channels
  ↓ + 输入（残差连接，仅在 stride=1 且通道数不变时）
输出
```

MBConv 与 ResNet 残差块的对比：

```
ResNet 残差块：输入 256 → 1×1 降维到 64 → 3×3 → 1×1 升维到 256（沙漏形）

MBConv 反向残差：输入 32 → 1×1 升维到 192 → 3×3 深度卷积 → 1×1 降维到 16（纺锤形）
```

总结：

```
EfficientNet 通过复合缩放统一调整深度、宽度和分辨率，在提升精度的同时保持高效。
MBConv 模块结合了反向残差、深度可分离卷积和 SE 注意力，是 EfficientNet 的核心构建块。
从 B0 到 B7，网络在三个维度上协同增长，实现了 SOTA 精度与计算效率的平衡。
```





# PyTorch Image Captioning 教程笔记

本笔记对应 Aladdin Persson 的 [Pytorch Image Captioning Tutorial](https://www.youtube.com/watch?v=y2BaTt1fxJU)，使用 Flickr8k 数据集，构建一个由 CNN 编码器和 LSTM 解码器组成的图像描述模型。

---

## 1. 什么是 Image Captioning？

Image Captioning 的任务是：

```
输入一张图片
      ↓
生成一段描述图片内容的文字
```

例如：

```
图片：一只狗在草地上奔跑
生成文本：a dog is running on the grass
```

它结合了两个方向：

```
计算机视觉：理解图片内容
自然语言处理：生成描述文字
```

基本结构：

```
图片
  ↓
CNN Encoder
  ↓
图像特征向量
  ↓
LSTM Decoder
  ↓
逐词生成 Caption
```

---

## 2. Encoder-Decoder 结构

### Encoder

Encoder 使用 CNN 从图片中提取特征：

```
图片 → CNN → 图像特征向量
```

通常可以使用预训练的 ResNet、Inception 等模型，并移除最后的分类层。

### Decoder

Decoder 使用 LSTM 根据图像特征和之前生成的单词，预测下一个单词：

```
图像特征 + <START>
      ↓
预测第一个单词
      ↓
继续输入前一个单词
      ↓
预测下一个单词
      ↓
直到生成 <END>
```

---

## 3. 文本词表

模型不能直接处理字符串，因此需要建立词表：

```
单词 → 整数编号
```

通常需要加入特殊标记：

```
<PAD>    → 补齐不同长度的句子
<START>  → 表示句子开始
<END>    → 表示句子结束
<UNK>    → 表示未知单词
```

示例：

```
原始句子：
a dog is running

加入特殊标记：
<START> a dog is running <END>

转换为编号：
[1, 5, 8, 12, 20, 2]
```

一个简单的词表实现：

```
class Vocabulary:
    def __init__(self, frequency_threshold):
        self.itos = {
            0: "<PAD>",
            1: "<START>",
            2: "<END>",
            3: "<UNK>",
        }  # 根据编号获取单词

        self.stoi = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3,
        }  # 根据单词获取编号

        self.frequency_threshold = frequency_threshold  # 最低词频要求

    def __len__(self):
        return len(self.itos)  # 返回词表大小

    def tokenizer(self, text):
        return text.lower().split()  # 转为小写并按空格分词

    def build_vocabulary(self, sentence_list):
        frequencies = {}  # 保存每个单词出现的次数
        index = 4  # 普通单词从编号 4 开始

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1  # 统计词频

        for word, frequency in frequencies.items():
            if frequency >= self.frequency_threshold:
                self.stoi[word] = index  # 保存单词到编号的映射
                self.itos[index] = word  # 保存编号到单词的映射
                index += 1  # 更新下一个编号

    def numericalize(self, text):
        tokens = self.tokenizer(text)  # 对句子进行分词

        return [
            self.stoi.get(
                token,
                self.stoi["<UNK>"],
            )
            for token in tokens
        ]  # 将每个单词转换为编号
```

---

## 4. 自定义 Caption Dataset

每个样本通常包含：

```
图片路径
图片对应的描述文本
```

`Dataset` 读取图片并将描述转换为编号：

```
from PIL import Image  # 用于读取图片
import torch  # 导入 PyTorch
from torch.utils.data import Dataset  # 导入 Dataset 基类


class CaptionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe  # 保存图片路径和描述文本
        self.transform = transform  # 保存图片预处理方法

        self.vocab = Vocabulary(
            frequency_threshold=5
        )  # 创建词表

        self.vocab.build_vocabulary(
            dataframe["caption"].tolist()
        )  # 使用所有描述文本构建词表

    def __len__(self):
        return len(self.dataframe)  # 返回样本数量

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]  # 获取一个样本

        image = Image.open(row["image"]).convert("RGB")  # 读取图片并转换为 RGB
        caption = row["caption"]  # 获取图片描述文本

        if self.transform is not None:
            image = self.transform(image)  # 对图片进行预处理

        numericalized_caption = [
            self.vocab.stoi["<START>"]
        ]  # 添加句子开始标记

        numericalized_caption += self.vocab.numericalize(
            caption
        )  # 将描述文本转换为单词编号

        numericalized_caption.append(
            self.vocab.stoi["<END>"]
        )  # 添加句子结束标记

        return image, torch.tensor(
            numericalized_caption,
            dtype=torch.long,
        )  # 返回图片和 caption 编号
```

---

## 5. 处理不同长度的 Caption

不同图片的描述长度通常不同：

```
a dog runs                  → 长度 3
a small dog runs on grass   → 长度 5
```

一个 batch 中的序列需要补齐到相同长度。

```
from torch.nn.utils.rnn import pad_sequence  # 导入序列补齐函数


def caption_collate_fn(batch):
    images = []  # 保存图片
    captions = []  # 保存描述序列

    for image, caption in batch:
        images.append(image)  # 添加图片
        captions.append(caption)  # 添加 caption

    images = torch.stack(images)  # 将图片组合成批次
    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0,
    )  # 使用 <PAD> 编号 0 补齐序列

    return images, captions  # 返回图片批次和 caption 批次
```

补齐示例：

```
[<START>, a, dog, <END>]

[<START>, a, small, dog, runs, <END>]
```

补齐后：

```
[
    [<START>, a, dog, <END>, <PAD>, <PAD>],
    [<START>, a, small, dog, runs, <END>]
]
```

---

## 6. CNN Encoder

Encoder 使用 CNN 提取图像特征。

```
import torch.nn as nn  # 导入神经网络模块
import torchvision.models as models  # 导入 torchvision 模型


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()  # 初始化父类

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )  # 加载预训练 ResNet-50

        modules = list(
            resnet.children()
        )[:-1]  # 移除最后的分类层

        self.resnet = nn.Sequential(
            *modules
        )  # 保留 CNN 特征提取部分

        self.linear = nn.Linear(
            resnet.fc.in_features,
            embed_size,
        )  # 将 CNN 特征映射到指定维度

        self.bn = nn.BatchNorm1d(
            embed_size,
            momentum=0.01,
        )  # 对图像特征进行归一化

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)  # 使用 ResNet 提取图像特征

        features = features.reshape(
            features.size(0),
            -1,
        )  # 将 (batch, 2048, 1, 1) 展平为 (batch, 2048)

        features = self.linear(features)  # 将 CNN 特征映射到 embed_size
        features = self.bn(features)  # 对特征进行 BatchNorm

        return features  # 返回图像特征
```

如果希望微调整个 CNN，可以移除：

```
with torch.no_grad():
```

并将 CNN 参数设置为可训练。

---

## 7. LSTM Decoder

Decoder 接收图像特征和 Caption 前面的单词，预测下一个单词。

```
class DecoderRNN(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers,
    ):
        super(DecoderRNN, self).__init__()  # 初始化父类

        self.embed = nn.Embedding(
            vocab_size,
            embed_size,
        )  # 将单词编号转换为词向量

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )  # 创建 LSTM 解码器

        self.linear = nn.Linear(
            hidden_size,
            vocab_size,
        )  # 将 LSTM 输出映射为词表大小的分数

        self.dropout = nn.Dropout(
            p=0.5
        )  # 防止解码器过拟合

    def forward(self, features, captions):
        embeddings = self.dropout(
            self.embed(captions)
        )  # 将 caption 编号转换为词向量

        features = features.unsqueeze(1)  # 将图像特征变为 (batch, 1, embed_size)

        embeddings = torch.cat(
            (features, embeddings),
            dim=1,
        )  # 将图像特征放在序列最前面

        hiddens, _ = self.lstm(
            embeddings
        )  # 使用 LSTM 处理图像特征和文本序列

        outputs = self.linear(
            hiddens
        )  # 为每个时间步预测下一个单词

        return outputs  # 返回每个时间步的词汇分数
```

输入输出形状：

```
features：
(batch, embed_size)

captions：
(batch, sequence_length)

embeddings：
(batch, sequence_length, embed_size)

outputs：
(batch, sequence_length + 1, vocab_size)
```

---

## 8. Encoder-Decoder 模型

```
class CNNtoRNN(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers,
    ):
        super(CNNtoRNN, self).__init__()  # 初始化父类

        self.encoder = EncoderCNN(
            embed_size
        )  # 创建 CNN 编码器

        self.decoder = DecoderRNN(
            embed_size,
            hidden_size,
            vocab_size,
            num_layers,
        )  # 创建 LSTM 解码器

    def forward(self, images, captions):
        features = self.encoder(images)  # 从图片中提取特征

        outputs = self.decoder(
            features,
            captions,
        )  # 根据图像特征和 caption 生成词汇分数

        return outputs  # 返回预测结果
```

---

## 9. Teacher Forcing

训练时，通常使用 Teacher Forcing：

```
输入：
<START> a dog is

目标：
a dog is running <END>
```

代码中使用：

```
captions[:, :-1]
```

作为输入：

```
inputs = captions[:, :-1]  # 去掉最后一个 token，作为模型输入
targets = captions[:, 1:]  # 去掉第一个 token，作为预测目标
```

这样模型在每个时间步都使用真实的前一个单词，而不是使用自己上一步预测的单词。

训练代码：

```
for images, captions in train_loader:
    images = images.to(device)  # 将图片移动到设备
    captions = captions.to(device)  # 将 caption 移动到设备

    outputs = model(
        images,
        captions[:, :-1],
    )  # 使用除最后一个 token 外的 caption 作为输入

    outputs = outputs[:, 1:, :]  # 去掉图像特征对应的第一个输出

    targets = captions[:, 1:]  # 目标是从第二个 token 开始的 caption

    loss = criterion(
        outputs.reshape(-1, outputs.size(2)),
        targets.reshape(-1),
    )  # 展平后计算每个时间步的交叉熵损失

    optimizer.zero_grad()  # 清除旧梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新模型参数
```

损失函数：

```
criterion = nn.CrossEntropyLoss(
    ignore_index=0
)  # 忽略 <PAD> 位置的损失
```

因为 `0` 是 `<PAD>` 的编号，不应该让补齐位置影响训练。

---

## 10. 推理阶段逐词生成

训练阶段可以使用完整 Caption，但推理时没有真实 Caption，只能根据模型上一步输出的单词继续生成。

流程：

```
输入图片
  ↓
生成 <START>
  ↓
预测下一个单词
  ↓
把预测单词重新输入模型
  ↓
继续预测
  ↓
生成 <END> 或达到最大长度
```

一个简单的贪心搜索实现：

```
def caption_image(
    model,
    image,
    vocabulary,
    max_length=50,
):
    model.eval()  # 切换到评估模式

    result = []  # 保存生成的单词
    device = next(model.parameters()).device  # 获取模型所在设备

    with torch.inference_mode():
        features = model.encoder(
            image.unsqueeze(0).to(device)
        )  # 使用 CNN 提取单张图片特征

        states = None  # 初始化 LSTM 隐藏状态

        for _ in range(max_length):
            if len(result) == 0:
                word_id = torch.tensor(
                    [vocabulary.stoi["<START>"]],
                    device=device,
                )  # 第一步输入 <START>
            else:
                word_id = torch.tensor(
                    [vocabulary.stoi[result[-1]]],
                    device=device,
                )  # 后续输入上一步生成的单词

            embeddings = model.decoder.embed(
                word_id
            ).unsqueeze(1)  # 将当前单词转换为词向量

            if states is None:
                lstm_input = torch.cat(
                    [features.unsqueeze(1), embeddings],
                    dim=1,
                )  # 第一步同时输入图像特征和 <START>
            else:
                lstm_input = embeddings  # 后续时间步只输入当前单词

            output, states = model.decoder.lstm(
                lstm_input,
                states,
            )  # 使用 LSTM 生成当前时间步输出

            scores = model.decoder.linear(
                output[:, -1, :]
            )  # 将 LSTM 输出映射为词表分数

            predicted_id = scores.argmax(
                dim=1
            ).item()  # 选择分数最高的单词编号

            predicted_word = vocabulary.itos[
                predicted_id
            ]  # 将编号转换回单词

            if predicted_word == "<END>":
                break  # 生成结束标记时停止

            result.append(predicted_word)  # 保存生成的单词

    model.train()  # 恢复训练模式
    return result  # 返回生成的单词列表
```

需要注意：上面代码是为了说明生成过程。实际工程中，通常会使用更清晰的 token ID 列表，而不是用生成出的单词再次查找编号。

---

## 11. 训练和推理的区别

|阶段|输入|生成方式|
|---|---|---|
|训练|真实 Caption|Teacher Forcing|
|验证|图片和 `<START>`|逐词生成|
|推理|只有图片|逐词生成|

训练阶段：

```
真实前一个单词 → 预测下一个单词
```

推理阶段：

```
模型上一步预测的单词 → 预测下一个单词
```

---

## 12. 训练时的完整核心代码

```
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # 选择运行设备

embed_size = 256  # 图像特征和词向量维度
hidden_size = 256  # LSTM 隐藏状态维度
num_layers = 1  # LSTM 层数
learning_rate = 0.001  # 学习率
num_epochs = 10  # 训练轮数

model = CNNtoRNN(
    embed_size=embed_size,
    hidden_size=hidden_size,
    vocab_size=len(train_dataset.vocab),
    num_layers=num_layers,
).to(device)  # 创建并移动模型

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)  # 忽略 <PAD> 位置的损失

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)  # 创建 Adam 优化器

for epoch in range(num_epochs):
    model.train()  # 切换到训练模式

    for images, captions in train_loader:
        images = images.to(device)  # 移动图片
        captions = captions.to(device)  # 移动 caption

        outputs = model(
            images,
            captions[:, :-1],
        )  # 使用 teacher forcing 进行训练

        outputs = outputs[:, 1:, :]  # 对齐预测时间步

        targets = captions[:, 1:]  # 目标为后移一位的真实 caption

        loss = criterion(
            outputs.reshape(-1, outputs.size(2)),
            targets.reshape(-1),
        )  # 计算所有时间步的交叉熵损失

        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()  # 反向传播

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )  # 防止 LSTM 出现梯度爆炸

        optimizer.step()  # 更新模型参数

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {loss.item():.4f}"
    )  # 输出当前训练损失
```

---

## 13. 图像预处理

如果 Encoder 使用 ImageNet 预训练 ResNet，图片通常需要：

```
from torchvision import transforms  # 导入图像转换工具


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),  # 使用 ImageNet 的标准化参数
])
```

输入图片形状：

```
(batch_size, 3, 224, 224)
```

---

## 14. 常见问题

### 忘记加入 `<START>` 和 `<END>`

模型需要知道：

```
什么时候开始生成
什么时候停止生成
```

因此每个 Caption 都应该包含：

```
<START> ... <END>
```

### 没有忽略 `<PAD>` 损失

应该使用：

```
nn.CrossEntropyLoss(ignore_index=0)
```

否则模型会被要求学习大量没有意义的填充位置。

### 训练和推理逻辑混淆

训练时可以使用真实 Caption：

```
captions[:, :-1]
```

推理时没有真实 Caption，只能逐词生成。

### CNN 特征维度不匹配

如果使用 ResNet-50，去掉分类层后通常得到：

```
(batch, 2048, 1, 1)
```

展平后是：

```
(batch, 2048)
```

因此需要：

```
nn.Linear(2048, embed_size)
```

### 生成句子无限循环

推理时要设置最大生成长度：

```
max_length=50
```

并在生成 `<END>` 时停止：

```
if predicted_word == "<END>":
    break
```

---

## 15. 模型流程总结

```
图片
(3, 224, 224)
      ↓
CNN Encoder
      ↓
图像特征
(embed_size,)
      ↓
加入 <START>
      ↓
LSTM Decoder
      ↓
预测单词
      ↓
继续输入上一个单词
      ↓
直到生成 <END>
```

训练过程：

```
图片 + 真实 Caption
        ↓
CNN 提取图像特征
        ↓
LSTM 逐时间步预测单词
        ↓
CrossEntropyLoss
        ↓
反向传播更新 CNN 和 LSTM
```

一句话总结：

```
Image Captioning 本质上是一个 Encoder-Decoder 模型：
CNN 将图片转换为特征，LSTM 根据图像特征逐词生成描述。
```

可进一步改进：

```
使用更大的预训练 CNN
增加训练轮数
使用 Attention 让模型关注图片不同区域
使用 Beam Search 代替简单的贪心搜索
使用 Transformer Decoder
```


# PyTorch 实现 Neural Style Transfer（神经风格迁移）

神经风格迁移（Neural Style Transfer，NST）是 2015 年由 Leon A. Gatys 等人提出的图像生成技术，核心思想是将一张图片的“内容”与另一张图片的“艺术风格”融合在一起，生成一幅全新的图像。原始论文：[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

---

## 1. 核心思想：内容 + 风格 = 生成图像

风格迁移算法需要三张图片：

```
内容图片（Content Image）  +  风格图片（Style Image）  →  生成图片（Generated Image）
    （保留物体/场景结构）        （保留色彩/纹理/笔触）       （两者的融合）
```

算法的工作流程如下：

```
输入：内容图片 C、风格图片 S、随机初始化的生成图片 G
    ↓
用预训练的 VGG19 分别提取三张图片的特征
    ↓
计算内容损失：衡量 G 与 C 在高层特征上的差异
计算风格损失：衡量 G 与 S 在风格特征上的差异
    ↓
总损失 = 内容损失 + 风格损失
    ↓
反向传播，更新生成图片 G 的像素值
    ↓
重复迭代，直到 G 同时保留 C 的内容和 S 的风格
```

原理很简单：定义两个距离，内容距离 \( D_C \) 衡量两张图像内容的差异，风格距离 \( D_S \) 衡量两张图像风格的差异。然后不断调整生成图片，使其与内容图片的内容距离最小化，同时与风格图片的风格距离最小化。

---

## 2. VGG19 作为特征提取器

风格迁移使用在 ImageNet 上预训练的 VGG19 网络来提取图像特征。

### 2.1 为什么选择 VGG19？

- VGG19 的卷积层能提取从低级（边缘、颜色）到高级（物体形状）的特征
- 不同层捕获的信息不同：**浅层**捕获纹理、颜色等风格信息，**深层**捕获物体、布局等内容信息
- 预训练模型已经学会了如何“理解”图像内容

### 2.2 VGG19 的层结构

```
VGG19 结构（只取卷积层，舍弃全连接层）：

Layer 0:  Conv2d (3 → 64)
Layer 1:  ReLU
Layer 2:  Conv2d (64 → 64)
Layer 3:  ReLU
Layer 4:  MaxPool2d
Layer 5:  Conv2d (64 → 128)
Layer 6:  ReLU
Layer 7:  Conv2d (128 → 128)
Layer 8:  ReLU
Layer 9:  MaxPool2d
Layer 10: Conv2d (128 → 256)
Layer 11: ReLU
Layer 12: Conv2d (256 → 256)
Layer 13: ReLU
Layer 14: Conv2d (256 → 256)
Layer 15: ReLU
Layer 16: Conv2d (256 → 256)
Layer 17: ReLU
Layer 18: MaxPool2d
Layer 19: Conv2d (256 → 512)
Layer 20: ReLU
Layer 21: Conv2d (512 → 512)
Layer 22: ReLU
Layer 23: Conv2d (512 → 512)
Layer 24: ReLU
Layer 25: Conv2d (512 → 512)
Layer 26: ReLU
Layer 27: MaxPool2d
Layer 28: Conv2d (512 → 512)
Layer 29: ReLU
Layer 30: Conv2d (512 → 512)
Layer 31: ReLU
Layer 32: Conv2d (512 → 512)
Layer 33: ReLU
Layer 34: Conv2d (512 → 512)
Layer 35: ReLU
Layer 36: MaxPool2d
```

---

## 3. 内容损失（Content Loss）

内容损失衡量生成图片与内容图片在**高层语义特征**上的差异。使用的是 VGG19 中较深层（如 `conv4_2`）的特征图。

### 3.1 原理

- 高层特征图保留了图像的**物体形状和布局**信息
- 通过最小化内容损失，让生成图片“继承”内容图片的物体结构
- 使用 **MSE 损失**计算特征图之间的差异

### 3.2 公式

```
F_content = VGG(content_image)[content_layer]   # 内容图片的特征
F_gen = VGG(generated_image)[content_layer]     # 生成图片的特征

Content_Loss = mean((F_content - F_gen)²)
```

### 3.3 代码实现

```python
def content_loss(content_features, generated_features):
    # content_features: 内容图片在某一层的特征图
    # generated_features: 生成图片在同一层的特征图
    return torch.mean((content_features - generated_features) ** 2)
```

---

## 4. 风格损失（Style Loss）

风格损失衡量生成图片与风格图片在**风格特征**上的差异。风格由特征图之间的**相关性（Gram 矩阵）**来表示。

### 4.1 为什么用 Gram 矩阵？

- 特征图的每个通道可以看作一种“风格滤波器”（如纹理、颜色倾向）
- 不同通道之间的相关性（共现关系）就构成了风格
- Gram 矩阵计算的是**通道之间的内积**，捕获了“哪些风格特征同时出现”

### 4.2 Gram 矩阵的计算

```
输入特征图 F: (C, H, W)  # C 个通道，每个通道 H×W

1. 将 F 重塑为 (C, H×W)
2. Gram = F @ F.T  # (C, C) 矩阵

Gram[i, j] = sum(F[i, :] * F[j, :])  # 通道 i 和通道 j 的相关性
```

### 4.3 多层风格损失

风格不只由一层决定，通常使用**多个层**（如 `conv1_2`、`conv2_2`、`conv3_2`、`conv4_2`、`conv5_2`）的风格损失之和，每层赋予不同的权重。

### 4.4 公式

```
对于每一层 l：
  Gram_style_l = Gram(VGG(style_image)[l])
  Gram_gen_l = Gram(VGG(generated_image)[l])
  Layer_Style_Loss_l = mean((Gram_style_l - Gram_gen_l)²) / (C_l * H_l * W_l)²

Total_Style_Loss = sum(w_l * Layer_Style_Loss_l)
```

### 4.5 代码实现

```python
def gram_matrix(features):
    # features: (batch_size, channels, height, width)
    batch_size, channels, h, w = features.size()
    features = features.view(batch_size, channels, h * w)  # 展平空间维度
    gram = torch.bmm(features, features.transpose(1, 2))   # 批量矩阵乘法
    return gram / (channels * h * w)  # 归一化

def style_loss(style_features, generated_features):
    # style_features: 风格图片在某一层的特征图
    # generated_features: 生成图片在同一层的特征图
    gram_style = gram_matrix(style_features)
    gram_gen = gram_matrix(generated_features)
    return torch.mean((gram_style - gram_gen) ** 2)
```

---

## 5. 总损失与优化

### 5.1 总损失公式

```
Total_Loss = α * Content_Loss + β * Style_Loss
```

- α（内容权重）：控制生成图片保留多少内容
- β（风格权重）：控制生成图片采用多少风格
- 通常 α/β 的比例决定了最终效果，常见设置：α = 1，β = 1e6（因为风格损失的值通常比内容损失小很多）

### 5.2 优化器

与常规训练不同，风格迁移**不更新模型参数**，而是**更新生成图片的像素值**：

```python
# 生成图片作为可训练参数
generated_image = content_image.clone().requires_grad_(True)

# 使用 Adam 或 L-BFGS 优化器
optimizer = optim.Adam([generated_image], lr=0.01)
```

### 5.3 完整训练循环

```python
for step in range(num_steps):
    optimizer.zero_grad()

    # 前向传播：提取特征
    gen_features = model(generated_image)
    content_features = model(content_image)
    style_features = model(style_image)

    # 计算内容损失
    c_loss = content_loss(content_features[content_layer], gen_features[content_layer])

    # 计算风格损失（多个层）
    s_loss = 0
    for layer in style_layers:
        s_loss += style_loss(style_features[layer], gen_features[layer])

    # 总损失
    total_loss = alpha * c_loss + beta * s_loss

    # 反向传播，更新生成图片
    total_loss.backward()
    optimizer.step()
```

---

## 6. 完整实现

### 6.1 导入必要的库

```python
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import copy
```

### 6.2 图像加载与预处理

```python
# 图像尺寸
IMAGE_SIZE = 256

# 预处理：调整大小 → 转张量 → 归一化
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加 batch 维度
    return image

# 加载内容图片和风格图片
content_image = load_image('content.jpg')
style_image = load_image('style.jpg')

# 初始化生成图片为内容图片的副本（可训练）
generated_image = content_image.clone().requires_grad_(True)
```

### 6.3 构建 VGG19 特征提取器

```python
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        # 加载预训练 VGG19，只取特征层（不含全连接层）
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.model = vgg

        # 冻结所有参数（不需要训练）
        for param in self.model.parameters():
            param.requires_grad_(False)

        # 内容层和风格层的索引
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        # 层名到索引的映射
        self.layer_name_map = {
            'conv_1': 0,   # 实际为 conv1_1
            'conv_2': 5,   # conv2_1
            'conv_3': 10,  # conv3_1
            'conv_4': 19,  # conv4_1
            'conv_5': 28,  # conv5_1
        }

    def forward(self, x):
        features = {}

        # 逐层前向传播，记录需要的层的输出
        for name, layer in self.model._modules.items():
            x = layer(x)
            if int(name) in [0, 5, 10, 19, 28]:
                # 记录该层输出（对应 conv1_1, conv2_1, conv3_1, conv4_1, conv5_1）
                layer_name = list(self.layer_name_map.keys())[
                    [0, 5, 10, 19, 28].index(int(name))
                ]
                features[layer_name] = x

        return features
```

### 6.4 定义损失函数

```python
def gram_matrix(features):
    batch_size, channels, h, w = features.size()
    features = features.view(batch_size, channels, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (channels * h * w)

def content_loss(content_feat, gen_feat):
    return torch.mean((content_feat - gen_feat) ** 2)

def style_loss(style_feat, gen_feat):
    gram_style = gram_matrix(style_feat)
    gram_gen = gram_matrix(gen_feat)
    return torch.mean((gram_style - gram_gen) ** 2)
```

### 6.5 训练

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 移动到设备
content_image = content_image.to(device)
style_image = style_image.to(device)
generated_image = generated_image.to(device)

# 初始化 VGG 模型
vgg = VGGFeatureExtractor().to(device)

# 超参数
alpha = 1.0       # 内容权重
beta = 1e6        # 风格权重
num_steps = 3000
learning_rate = 0.01

# 优化器：优化生成图片
optimizer = optim.Adam([generated_image], lr=learning_rate)

# 预先提取内容图片和风格图片的特征（节省计算）
content_features = vgg(content_image)
style_features = vgg(style_image)

# 训练循环
for step in range(num_steps):
    optimizer.zero_grad()

    # 提取生成图片的特征
    gen_features = vgg(generated_image)

    # 计算内容损失（使用 conv_4 层）
    c_loss = content_loss(
        content_features['conv_4'],
        gen_features['conv_4']
    )

    # 计算风格损失（使用所有风格层）
    s_loss = 0
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    for layer in style_layers:
        s_loss += style_loss(
            style_features[layer],
            gen_features[layer]
        )

    # 总损失
    total_loss = alpha * c_loss + beta * s_loss

    # 反向传播
    total_loss.backward()
    optimizer.step()

    # 打印进度
    if step % 100 == 0:
        print(f"Step [{step}/{num_steps}], "
              f"Content Loss: {c_loss.item():.4f}, "
              f"Style Loss: {s_loss.item():.4f}")

print("训练完成！")
```

### 6.6 显示结果

```python
def imshow(tensor, title=None):
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)

    # 显示
    plt.imshow(tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    if title:
        plt.title(title)
    plt.show()

# 显示结果
imshow(generated_image, title="Generated Image")
```

---

## 7. 常见错误

### 7.1 生成图片的梯度未开启

生成图片需要作为可训练参数，必须设置 `requires_grad_(True)`：

```python
# 正确写法
generated_image = content_image.clone().requires_grad_(True)

# 错误写法：无法计算梯度
generated_image = content_image.clone()
```

### 7.2 模型参数被意外更新

VGG19 的参数应该被冻结：

```python
# 正确写法
for param in vgg.parameters():
    param.requires_grad_(False)

# 错误写法：没有冻结，会浪费计算资源
# 默认 requires_grad=True
```

### 7.3 Gram 矩阵未归一化

不同层的特征图尺寸不同，Gram 矩阵的值会随尺寸变化，需要归一化：

```python
# 正确写法
def gram_matrix(features):
    batch_size, channels, h, w = features.size()
    features = features.view(batch_size, channels, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (channels * h * w)  # 归一化

# 错误写法：没有归一化，不同层的损失量级差异巨大
def gram_matrix(features):
    features = features.view(features.size(0), features.size(1), -1)
    return torch.bmm(features, features.transpose(1, 2))
```

### 7.4 内容层和风格层选择不当

- 内容层太浅：生成图片会保留太多细节，风格迁移效果不明显
- 内容层太深：生成图片可能丢失部分内容结构
- 风格层太少：风格迁移不充分

常用选择：
- 内容层：`conv4_2`（或 `conv_4`）
- 风格层：`conv1_1`、`conv2_1`、`conv3_1`、`conv4_1`、`conv5_1`

### 7.5 α 和 β 的比例不当

风格损失的值通常远小于内容损失（因为 Gram 矩阵的值很小），需要较大的 β 来平衡：

```python
# 常用比例
alpha = 1.0
beta = 1e6  # 风格损失通常比内容损失小 1e6 倍

# 如果风格迁移效果不明显，尝试增大 beta
# 如果内容丢失严重，尝试增大 alpha
```

---

## 8. 可能的改进方向

- **总变差正则化（Total Variation Regularization）**：减少生成图片中的高频噪声，使图像更平滑。
- **使用更快的风格迁移方法**：如 AdaIN（Adaptive Instance Normalization），训练一个前馈网络，一次前向传播即可完成风格迁移。
- **多风格迁移**：同时学习多种风格，在推理时选择任意风格组合。
- **内容与风格的动态平衡**：在训练过程中逐渐调整 α 和 β 的比例。

---

## 9. 重点总结

任务本质：

```
风格迁移 = 保留内容图片的物体结构 + 采用风格图片的色彩/纹理/笔触
```

核心公式：

```
Total_Loss = α × Content_Loss + β × Style_Loss
```

内容损失：

```
使用 VGG 深层特征，MSE 损失
保留物体形状和布局
```

风格损失：

```
使用 Gram 矩阵（通道间相关性），MSE 损失
捕获纹理、颜色、笔触等风格信息
```

训练方式：

```
不更新模型参数，更新生成图片的像素值
生成图片作为可训练参数（requires_grad=True）
```

关键点：

```
VGG19 作为固定的特征提取器（参数冻结）
风格损失使用多层特征的 Gram 矩阵
α 和 β 的比例需要调整（β 通常远大于 α）
```

总结：

```
神经风格迁移通过预训练的 VGG19 提取图像的内容特征和风格特征，
定义内容损失和风格损失，然后通过梯度下降优化生成图片的像素值，
使其同时具备内容图片的物体结构和风格图片的艺术风格。
```


# PyTorch 实现 Simple GAN（生成对抗网络）

生成对抗网络（Generative Adversarial Network，GAN）是 2014 年由 Ian Goodfellow 等人提出的生成模型，核心思想是通过两个神经网络的“对抗”训练，让模型学会生成与真实数据分布相似的新样本。本笔记参考 Aladdin Persson 的教程，使用全连接层在 MNIST 数据集上实现一个最简单的 GAN。

---

## 1. GAN 的核心思想

GAN 由两个神经网络组成，它们相互博弈、共同进步：

```
真实图片（Real Images）
    ↓
判别器（Discriminator）←  ←  ←  ←  ←  ←  ←  ←  ←  ←  ←  ← 生成图片（Fake Images）
    ↑                                                  ↑
    │                                                  │
    │   判断图片是“真”还是“假”                          │   试图生成以假乱真的图片
    │                                                  │
    └────────────────── 对抗训练 ──────────────────────┘
```

### 1.1 生成器（Generator）

生成器的作用是将随机噪声（Random Noise）转化为“以假乱真”的图片：

```
随机噪声向量 z（如 64 维） →  生成器 G  →  生成的图片（如 784 维）
```

- 输入：从一个简单的分布（如标准正态分布）中采样的随机向量 z
- 输出：一张与真实图片尺寸相同的“假图片”
- 目标：骗过判别器，让判别器认为生成的图片是真实的

### 1.2 判别器（Discriminator）

判别器的作用是区分真实图片和生成图片：

```
输入图片（真实 or 生成） →  判别器 D  →  输出概率（0~1）
```

- 输入：一张图片（可能是真实的，也可能是生成器生成的）
- 输出：一个 0 到 1 之间的概率值，表示输入图片是“真实”的概率
- 目标：正确区分真实图片和生成图片

### 1.3 对抗训练

两个网络通过“对抗”共同进步：

```
训练判别器：
  真实图片 → 判别器 → 希望输出 1（真实）
  生成图片 → 判别器 → 希望输出 0（假）

训练生成器：
  随机噪声 → 生成器 → 生成图片 → 判别器 → 希望输出 1（骗过判别器）
```

这个过程可以看作一个博弈：生成器不断学习如何骗过判别器，判别器不断学习如何识破生成器，两者在对抗中共同进步。

---

## 2. 生成器（Generator）实现

生成器使用全连接层，将随机噪声向量映射为一张图片：

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),          # 输入：噪声向量 (64 维)
            nn.LeakyReLU(0.01),             # LeakyReLU 激活
            nn.Linear(256, img_dim),        # 输出：图片 (784 维)
            nn.Tanh(),                      # 将输出映射到 [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)
```

生成器的结构：

```
输入：随机噪声向量 (batch_size, z_dim=64)
    ↓
全连接层：64 → 256
    ↓
LeakyReLU(0.01)
    ↓
全连接层：256 → 784（28×28×1）
    ↓
Tanh（将输出范围限制在 [-1, 1]）
    ↓
输出：生成的图片 (batch_size, 784)
```

关键点说明：

- **LeakyReLU**：使用 LeakyReLU 而不是 ReLU，可以避免神经元死亡问题，让梯度更好地流动。
- **Tanh**：生成器的最后一层使用 Tanh 激活函数，将输出值压缩到 [-1, 1] 之间。这也要求真实图片在预处理时被归一化到 [-1, 1]。

---

## 3. 判别器（Discriminator）实现

判别器接收一张图片，输出一个 0 到 1 之间的概率值：

```python
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),        # 输入：图片 (784 维)
            nn.LeakyReLU(0.01),             # LeakyReLU 激活
            nn.Linear(128, 1),              # 输出：1 个值
            nn.Sigmoid(),                   # 映射到 [0, 1]
        )

    def forward(self, x):
        return self.disc(x)
```

判别器的结构：

```
输入：图片 (batch_size, 784)
    ↓
全连接层：784 → 128
    ↓
LeakyReLU(0.01)
    ↓
全连接层：128 → 1
    ↓
Sigmoid（输出 0~1 的概率值）
    ↓
输出：概率值，表示输入是真实图片的可能性
```

关键点说明：

- **Sigmoid**：最后一层使用 Sigmoid，将输出映射到 [0, 1] 之间，便于解释为概率。
- **LeakyReLU**：判别器也使用 LeakyReLU，保持梯度流动的稳定性。

---

## 4. 超参数与数据加载

### 4.1 超参数设置

```python
import torch
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 超参数
lr = 3e-4                    # 学习率
z_dim = 64                   # 噪声向量的维度
image_dim = 28 * 28 * 1      # MNIST 图片维度：784
batch_size = 32              # 批次大小
num_epochs = 50              # 训练轮数
```

### 4.2 数据预处理与加载

MNIST 数据集包含 28×28 的灰度手写数字图片：

```python
# 数据预处理：转为张量并归一化到 [-1, 1]
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # 将 [0,1] 映射到 [-1,1]
])

# 加载 MNIST 数据集
dataset = datasets.MNIST(
    root="dataset/",
    transform=transforms,
    download=True
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)
```

注意：`transforms.Normalize((0.5,), (0.5,))` 将像素值从 [0, 1] 映射到 [-1, 1]，与生成器输出的 Tanh 范围匹配。

---

## 5. 完整训练流程

### 5.1 初始化模型、优化器和损失函数

```python
# 初始化生成器和判别器
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

# 优化器
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# 损失函数：二元交叉熵
criterion = nn.BCELoss()

# 固定噪声，用于可视化训练过程中的生成效果
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
```

### 5.2 训练循环

训练过程交替进行两个步骤：训练判别器和训练生成器。

```python
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        # 将图片展平为 784 维向量
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### 步骤 1：训练判别器 ###
        # 判别器的目标：最大化 log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        # 判别器对真实图片的判断
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        # 判别器对生成图片的判断
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # 判别器总损失
        lossD = (lossD_real + lossD_fake) / 2

        # 反向传播更新判别器
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### 步骤 2：训练生成器 ###
        # 生成器的目标：最大化 log(D(G(z)))
        # 等价于最小化 BCE(D(G(z)), 1)
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        # 反向传播更新生成器
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # 打印进度
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Loss D: {lossD.item():.4f}, "
                f"Loss G: {lossG.item():.4f}"
            )
```

### 5.3 训练流程详解

**步骤 1：训练判别器（Discriminator）**

判别器的目标是正确区分真实图片和生成图片：

```
lossD_real = BCE(D(real), 1)   # 真实图片 → 希望输出 1
lossD_fake = BCE(D(fake), 0)   # 生成图片 → 希望输出 0
lossD = (lossD_real + lossD_fake) / 2
```

**步骤 2：训练生成器（Generator）**

生成器的目标是骗过判别器：

```
lossG = BCE(D(fake), 1)        # 希望判别器认为生成图片是真实的
```

注意：这里 `fake` 使用的是**同一个批次**的生成图片，因为 `retain_graph=True` 保留了计算图，使得生成器可以复用判别器的前向传播结果。

---

## 6. 完整代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 超参数
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50


# 判别器
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


# 生成器
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


# 数据加载
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset = datasets.MNIST(
    root="dataset/",
    transform=transforms,
    download=True
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

# 优化器
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# 损失函数
criterion = nn.BCELoss()

# 固定噪声用于可视化
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# 训练
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # 训练判别器
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2

        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # 训练生成器
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Loss D: {lossD.item():.4f}, "
                f"Loss G: {lossG.item():.4f}"
            )
```

---

## 7. 生成图片的可视化

训练完成后，可以使用生成器生成新的手写数字图片：

```python
import matplotlib.pyplot as plt

# 生成图片
with torch.no_grad():
    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
    fake = fake * 0.5 + 0.5  # 从 [-1, 1] 映射回 [0, 1]

# 显示生成的图片
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()
```

---

## 8. TensorBoard 可视化

Aladdin Persson 的代码中使用了 TensorBoard 来可视化训练过程中的真实图片和生成图片：

```python
from torch.utils.tensorboard import SummaryWriter

writer_fake = SummaryWriter("logs/fake")
writer_real = SummaryWriter("logs/real")

# 在训练循环中
if batch_idx == 0:
    with torch.no_grad():
        # 显示真实图片
        img_grid_real = torchvision.utils.make_grid(real[:32].reshape(-1, 1, 28, 28))
        writer_real.add_image("Real", img_grid_real, global_step=step)

        # 显示生成图片
        img_grid_fake = torchvision.utils.make_grid(fake[:32].reshape(-1, 1, 28, 28))
        writer_fake.add_image("Fake", img_grid_fake, global_step=step)

    step += 1
```

使用以下命令启动 TensorBoard：

```bash
tensorboard --logdir logs
```

---

## 9. 常见错误

### 9.1 判别器与生成器的输入维度不匹配

判别器接收展平后的图片（784 维），生成器输出也是 784 维：

```python
# 正确写法
real = real.view(-1, 784)  # (batch_size, 784)
fake = gen(noise)          # (batch_size, 784)

# 错误写法：忘记展平
real = real  # (batch_size, 1, 28, 28) - 判别器无法处理
```

### 9.2 生成器输出范围与真实图片范围不一致

生成器使用 Tanh 输出 [-1, 1]，真实图片也必须归一化到 [-1, 1]：

```python
# 正确写法：归一化到 [-1, 1]
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# 错误写法：只转为张量，范围是 [0, 1]
transforms = transforms.Compose([
    transforms.ToTensor(),
])
```

### 9.3 损失函数中标签的正确使用

判别器的目标是让真实图片输出 1，生成图片输出 0：

```python
# 正确写法
lossD_real = criterion(disc_real, torch.ones_like(disc_real))
lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

# 错误写法：标签搞反了
lossD_real = criterion(disc_real, torch.zeros_like(disc_real))
lossD_fake = criterion(disc_fake, torch.ones_like(disc_fake))
```

### 9.4 忘记 retain_graph=True

训练判别器后，计算图被释放，生成器无法复用 `fake` 的前向传播结果：

```python
# 正确写法
lossD.backward(retain_graph=True)

# 错误写法：计算图被释放，生成器的 lossG.backward() 会报错
lossD.backward()
```

### 9.5 判别器太强或太弱

- 判别器太强：生成器无法获得有效梯度，难以学习
- 判别器太弱：无法给生成器提供有效的反馈

解决方案：
- 调整学习率（判别器学习率可以稍低）
- 使用标签平滑（Label Smoothing）
- 交替训练时，可以每训练 k 次判别器再训练 1 次生成器

---

## 10. 可能的改进方向

- **DCGAN（Deep Convolutional GAN）**：使用卷积层替代全连接层，更适合图像生成任务。
- **WGAN / WGAN-GP**：使用 Wasserstein 距离替代 JS 散度，训练更稳定。
- **条件 GAN（Conditional GAN）**：在生成器和判别器中加入类别标签，可以控制生成特定数字。
- **更大的数据集**：在 CIFAR-10 或 CelebA 等更复杂的数据集上训练。

---

## 11. 重点总结

GAN 的核心思想：

```
生成器（Generator）：噪声 → 假图片，目标是骗过判别器
判别器（Discriminator）：图片 → 真/假概率，目标是识破生成器
两者通过对抗训练共同进步
```

损失函数：

```
判别器损失：BCE(D(real), 1) + BCE(D(fake), 0)
生成器损失：BCE(D(fake), 1)
```

训练流程：

```
每个批次：
  1. 训练判别器（冻结生成器）
  2. 训练生成器（冻结判别器）
```

关键点：

```
生成器输出使用 Tanh，真实图片归一化到 [-1, 1]
使用 LeakyReLU 避免神经元死亡
训练判别器时使用 retain_graph=True
使用 TensorBoard 监控训练过程
```

总结：

```
简单 GAN 使用全连接层在 MNIST 数据集上实现了最基本的生成对抗网络。
虽然生成的图片质量有限，但它展示了 GAN 的核心思想：
两个神经网络通过对抗训练，让生成器学会生成以假乱真的数据。
这是理解更复杂 GAN 架构（如 DCGAN、WGAN）的基础。
```


# PyTorch GAN 变体系列

本笔记涵盖 DCGAN、WGAN、Conditional GAN、CycleGAN、SRGAN、ESRGAN 六种重要的 GAN 变体，重点梳理各变体的核心思想和关键改进。

---

## 1. DCGAN（Deep Convolutional GAN）

DCGAN 是将卷积神经网络引入 GAN 的开创性工作，由 Radford 等人于 2015 年提出。它用深度卷积架构替代了原始 GAN 中的多层感知机（MLP），极大提升了生成图像的质量。

### 核心改进

1. **取消池化层**：用带步长（stride）的卷积替代池化层，让网络自己学习下采样方式。
2. **移除全连接层**：生成器和判别器均采用全卷积架构，提高了网络稳定性。
3. **引入 Batch Normalization**：在生成器和判别器中均加入 BN 层，帮助梯度传递到每一层，防止生成器把所有样本收敛到同一个点。
4. **激活函数选择**：生成器除输出层使用 Tanh 外，其余层使用 ReLU；判别器全部使用 LeakyReLU。
5. **优化器**：使用 Adam 优化器。

### 意义

DCGAN 的判别器提取到的图像特征比其他无监督方法更有效，可用于图像分类任务。它证明了 GAN 可以学习到有意义的图像表征，为后续各种 GAN 变体奠定了基础。

---

## 2. WGAN（Wasserstein GAN）

WGAN 由 Arjovsky 等人于 2017 年提出，核心贡献是从理论上分析了原始 GAN 训练不稳定的根本原因。

### 核心思想

原始 GAN 使用 JS 散度（Jensen-Shannon Divergence）衡量真实分布与生成分布之间的距离。当两个分布没有重叠时，JS 散度为常数，梯度消失，判别器无法提供有效的学习信号。

WGAN 改用 **Wasserstein 距离**（又称 Earth Mover's Distance，推土机距离）作为损失函数。Wasserstein 距离即使在两个分布没有重叠时也能提供平滑的梯度。

### 关键改进

1. **去掉判别器最后一层的 Sigmoid**：WGAN 的判别器（称为 Critic）输出一个实数分数而非概率。
2. **生成器和判别器的 loss 不取 log**。
3. **权重裁剪（Weight Clipping）** ：限制 Critic 的权重绝对值在固定范围内，以强制满足 1-Lipschitz 约束。
4. **优化器建议**：使用 RMSprop 或 SGD，不建议使用基于动量的优化器（如 Adam）。

### WGAN-GP（带梯度惩罚的 WGAN）

WGAN-GP 是对 WGAN 的进一步改进，用**梯度惩罚（Gradient Penalty）**替代权重裁剪来强制执行 1-Lipschitz 约束。梯度惩罚能更平滑地约束 Critic，通常训练效果更好。

### 意义

- 解决了 GAN 训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度。
- 几乎解决了模式崩溃（Mode Collapse）问题。
- 提供了有意义的损失值，可用于判断模型是否收敛。

---

## 3. Conditional GAN（条件 GAN）

条件 GAN 由 Mirza 和 Osindero 于 2014 年提出，是原始 GAN 最早的重要改进之一。

### 核心思想

原始 GAN 的生成过程不可控：虽然能生成数据，但无法指定生成的具体内容。条件 GAN 通过在生成器和判别器的输入端同时添加**条件信息 y**（如类别标签、文本描述等）来解决这个问题。

### 具体实现

- **生成器**：输入为随机噪声 z + 条件 y
- **判别器**：输入为图像 x + 条件 y

损失函数变为：
```
min_G max_D V(D, G) = E[log D(x|y)] + E[log(1 - D(G(z|y)))]
```

### 应用

条件 GAN 可以实现**指定类别生成**（如生成特定数字的手写体）、**Text-to-Image**（根据文字描述生成图像）等任务。

---

## 4. CycleGAN（循环一致性 GAN）

CycleGAN 由 Zhu 等人于 2017 年提出，核心贡献是实现**无配对数据的图像到图像翻译**。

### 问题背景

传统的图像翻译任务（如 pix2pix）需要成对的训练数据（如同一场景的素描和照片），但很多场景下无法获得这样的配对数据。

### 核心思想

CycleGAN 同时训练两个生成器：
- **G: X → Y**：将源域 X 的图像转换到目标域 Y
- **F: Y → X**：将目标域 Y 的图像转换回源域 X

### 循环一致性损失（Cycle Consistency Loss）

为了防止生成器随意映射（由于没有配对数据，存在无数种可能的映射关系），CycleGAN 引入了循环一致性损失：

- **前向循环一致性**：F(G(x)) ≈ x（从 X 到 Y 再回到 X，应该得到原图）
- **后向循环一致性**：G(F(y)) ≈ y（从 Y 到 X 再回到 Y，应该得到原图）

### 总损失

```
Total Loss = Adversarial Loss (G) + Adversarial Loss (F) + λ × Cycle Consistency Loss
```

其中 λ 控制循环一致性损失的权重。

### 应用

CycleGAN 可用于风格迁移、物体变形、季节转换、照片增强等无需配对数据的图像翻译任务。

---

## 5. SRGAN（Super-Resolution GAN）

SRGAN 由 Ledig 等人于 2017 年提出，是首个将 GAN 应用于**图像超分辨率**（Single Image Super-Resolution, SISR）的工作。

### 问题背景

传统超分辨率方法以最小化 MSE（均方误差）为目标，虽然能获得较高的 PSNR（峰值信噪比），但生成的图像**高频细节不足、视觉感知不佳**。

### 核心创新：感知损失（Perceptual Loss）

SRGAN 提出**感知损失**替代传统的像素级 MSE 损失：

```
Perceptual Loss = Content Loss + Adversarial Loss
```

**Content Loss（内容损失）**

使用预训练的 VGG19 网络提取图像的高层特征，计算生成图像与真实图像在**特征空间**的差异，而非像素空间的差异。这能更好地保留图像的感知相似性。

**Adversarial Loss（对抗损失）**

判别器网络判断图像是超分辨率生成的还是真实的高清图像。对抗损失推动生成图像向自然图像流形靠近。

### 意义

SRGAN 生成的超分辨率图像在视觉质量上显著优于传统方法。MOS（Mean Opinion Score）测试表明，SRGAN 的感知质量更接近原始高清图像。

---

## 6. ESRGAN（Enhanced SRGAN）

ESRGAN 是 SRGAN 的增强版本，在 2018 年 PIRM 超分辨率挑战赛中获得了冠军。

### 三大改进

**1. 网络结构：RRDB（Residual-in-Residual Dense Block）**

ESRGAN 使用 RRDB 作为基本构建块。RRDB 是残差块中嵌套密集连接块的结构，具有更强的表示能力，且**移除了 Batch Normalization 层**以提升性能。

**2. 改进的判别器：RaGAN（Relativistic Average GAN）**

传统判别器判断“一张图像是真是假”，而 RaGAN 判断“**一张图像比另一张更真实**”。这种相对性的判别方式能提供更强的学习信号。

**3. 改进的感知损失**

使用 VGG 网络**激活前的特征**来计算感知损失，而不是激活后的特征。这能为亮度一致性和纹理恢复提供更强的监督。

### 意义

ESRGAN 在图像超分辨率任务上达到了当时最优的感知质量，其 RRDB 架构和 RaGAN 判别器成为后续超分辨率模型的标准组件。

---

## 7. 各变体对比总结

| 变体 | 核心思想 | 关键改进 | 主要应用 |
|------|---------|---------|---------|
| **DCGAN** | CNN + GAN | 卷积架构、BN、移除池化层 | 通用图像生成 |
| **WGAN** | Wasserstein 距离 | 解决训练不稳定、模式崩溃 | 通用图像生成 |
| **Conditional GAN** | 加入条件信息 | 可控生成 | 指定类别生成、Text-to-Image |
| **CycleGAN** | 循环一致性 | 无配对数据图像翻译 | 风格迁移、季节转换 |
| **SRGAN** | 感知损失 | 感知损失替代 MSE | 图像超分辨率 |
| **ESRGAN** | SRGAN 增强 | RRDB、RaGAN、改进感知损失 | 图像超分辨率 |


# PyTorch 从零实现字符级 LSTM 文本生成

文本生成是自然语言处理中的经典任务。使用字符级 LSTM 模型，我们可以逐字符地学习文本的统计规律，并生成与训练文本风格相似的新内容。本笔记以莎士比亚文集为例，实现一个简单的字符级文本生成器。

---

## 1. 核心思想

字符级 LSTM 文本生成的基本思路：

```
将文本拆解为字符序列 → 用 LSTM 学习字符之间的依赖关系 → 逐字符生成新文本
```

### 1.1 字符级 vs 词级

- **词级**：以单词为基本单位，生成更流畅但需要处理 OOV（Out-of-Vocabulary）问题，词表较大。
- **字符级**：以字符为基本单位，词表小（约几十到几百个字符），能生成任意单词组合，但需要更长的序列才能学到单词结构。

字符级 LSTM 特别适合**诗歌、代码、对话**等风格性强的文本生成任务。

### 1.2 训练目标

给定一个字符序列（如 “hello”），模型预测下一个字符（如 “w”）。训练时，我们使用滑动窗口构造输入-目标对：

```
输入序列: "h e l l"
目标字符: "o"

输入序列: "e l l o"
目标字符: " "
```

模型通过最小化交叉熵损失学习字符间的转移概率。

---

## 2. 数据预处理

### 2.1 加载文本数据

首先加载一个文本文件（如莎士比亚作品集），并将其转换为统一的格式。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 读取文本文件
file_path = "data/shakespeare.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"文本长度: {len(text)} 个字符")
print(f"前100个字符: {text[:100]}")
```

### 2.2 构建字符映射表（Character Vocabulary）

我们需要将字符转换为数值索引，以便输入到神经网络中。

```python
# 获取所有不同的字符（去重）
chars = sorted(set(text))
vocab_size = len(chars)
print(f"词汇表大小: {vocab_size}")

# 创建字符到索引和索引到字符的映射
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# 编码函数：将字符串转换为整数列表
def encode(text):
    return [char_to_idx[ch] for ch in text]

# 解码函数：将整数列表转换为字符串
def decode(indices):
    return ''.join([idx_to_char[idx] for idx in indices])
```

### 2.3 创建训练序列

为了训练 LSTM，需要创建固定长度的输入序列和对应的目标序列（每个输入序列后移一个字符）。

```python
# 超参数
seq_length = 100  # 每个输入序列的长度（字符数）
batch_size = 64
num_epochs = 50
learning_rate = 0.001

# 将整个文本编码为整数列表
data = encode(text)
data = torch.tensor(data, dtype=torch.long)

# 创建训练样本：使用滑动窗口
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:i+seq_length]      # 输入序列
        y = data[i+1:i+seq_length+1]  # 目标序列（每个位置对应下一个字符）
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

X, y = create_sequences(data, seq_length)
print(f"输入张量形状: {X.shape}")  # (num_samples, seq_length)
print(f"目标张量形状: {y.shape}")  # (num_samples, seq_length)
```

注意：目标序列是输入序列向后偏移一个字符，这意味着每个位置的预测目标是原始序列中该位置的下一个字符。

---

## 3. LSTM 模型定义

我们定义一个简单的单层或双层 LSTM，包含一个嵌入层（可选）和全连接层。

```python
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embed_size)
        out, hidden = self.lstm(x, hidden)  # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out)  # (batch_size, seq_length, vocab_size)
        return out, hidden

    def init_hidden(self, batch_size):
        # 初始化 LSTM 的隐藏状态和细胞状态
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
                  weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_())
        return hidden
```

### 模型结构说明

- **嵌入层（Embedding）** ：将字符索引映射为稠密向量。虽然字符级输入可以用 one-hot 编码，但嵌入层能学习更好的字符表示，且参数更少。
- **LSTM 层**：处理序列数据，捕捉长期依赖。`batch_first=True` 使输入形状为 `(batch, seq, feature)`。
- **全连接层（Linear）** ：将 LSTM 的输出映射到词汇表大小，用于预测每个位置的字符概率。

---

## 4. 训练循环

### 4.1 使用 DataLoader 进行批量训练

由于数据集可能很大，我们使用 `DataLoader` 分批加载。

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CharLSTM(vocab_size, embed_size=128, hidden_size=256, num_layers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_size = batch_x.size(0)

        # 初始化隐藏状态（每个批次独立）
        hidden = model.init_hidden(batch_size)

        # 前向传播
        output, hidden = model(batch_x, hidden)
        # output: (batch_size, seq_length, vocab_size)
        # 将输出和目标重塑为 (batch_size * seq_length, vocab_size) 和 (batch_size * seq_length)
        loss = criterion(output.view(-1, vocab_size), batch_y.view(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(loader):.4f}")
```

### 4.2 训练要点

- **梯度裁剪**：LSTM 在长序列上容易梯度爆炸，使用 `clip_grad_norm_` 限制梯度范数。
- **隐藏状态管理**：每个批次需要重置隐藏状态，因为批次之间没有时序连续性（数据被打乱）。
- **损失计算**：将输出展平为二维，以便计算整个序列所有位置的交叉熵损失。

---

## 5. 文本生成（采样）

训练完成后，我们可以使用模型生成新文本。生成过程是自回归的：每次预测一个字符，然后将该字符作为下一个时间步的输入。

### 5.1 温度采样（Temperature Sampling）

模型输出的是词汇表上的概率分布（经过 softmax）。温度采样通过调节温度参数控制生成的多样性：

```
p_i = exp(logits_i / temperature) / sum_j exp(logits_j / temperature)
```

- **temperature = 1**：标准 softmax，保持原始概率。
- **temperature < 1**：概率分布更尖锐，倾向于选择概率最高的字符，生成更稳定但可能重复。
- **temperature > 1**：概率分布更平滑，增加随机性，生成更多样但可能更混乱。

### 5.2 生成函数实现

```python
def generate_text(model, start_string, length=200, temperature=0.8):
    model.eval()
    with torch.no_grad():
        # 将起始字符串编码为张量
        input_indices = [char_to_idx[ch] for ch in start_string if ch in char_to_idx]
        if not input_indices:
            input_indices = [char_to_idx[' ']]  # 默认用空格
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)  # (1, len)

        # 初始化隐藏状态
        hidden = model.init_hidden(1)

        # 先通过起始字符串更新隐藏状态（可选，也可以从零开始）
        # 这里我们将起始字符串作为初始输入，然后逐步生成
        generated = list(input_indices)  # 已生成的字符索引列表

        # 依次生成后续字符
        for _ in range(length):
            # 前向传播，取最后一个时间步的输出
            output, hidden = model(input_tensor, hidden)
            # output: (1, current_seq_len, vocab_size)
            logits = output[:, -1, :] / temperature  # 取最后一个时间步，应用温度
            # 转为概率分布
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            # 采样一个字符
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_idx)
            # 将新字符作为下一步的输入（注意形状）
            input_tensor = torch.tensor([[next_idx]], dtype=torch.long).to(device)

        # 解码生成的字符序列
        generated_text = decode(generated)
        return generated_text
```

### 5.3 生成示例

```python
start_prompt = "To be, or not to be"
generated = generate_text(model, start_prompt, length=500, temperature=0.7)
print(generated)
```

---

## 6. 训练过程中的监控

可以使用 TensorBoard 监控损失变化，也可以定期生成文本评估模型进展。

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs/char_lstm")

# 在训练循环中
writer.add_scalar("Loss/train", epoch_loss/len(loader), epoch)

# 每 N 个 epoch 生成样本并记录
if epoch % 5 == 0:
    sample = generate_text(model, "The", length=100, temperature=0.8)
    print(f"Epoch {epoch}: {sample[:100]}...")
```

---

## 7. 常见错误

### 7.1 输入形状不匹配

LSTM 的输入应为 `(batch_size, seq_length, input_size)`，但我们直接传入索引序列。需要先经过嵌入层。

确保 `batch_first=True` 与数据形状一致：

```python
# 正确
lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
out, _ = lstm(embedded)  # embedded: (batch, seq, embed)

# 错误：batch_first 设置不一致
lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)  # 默认 False
out, _ = lstm(embedded)  # 要求 (seq, batch, embed)
```

### 7.2 隐藏状态未正确初始化

每个新批次必须重置隐藏状态，否则会错误地延续上一批次的时序信息：

```python
# 正确：每个批次重置
hidden = model.init_hidden(batch_size)

# 错误：复用上一个批次的 hidden（如果数据不是连续序列）
# 在打乱数据的情况下会引入噪声
```

### 7.3 温度采样时使用未归一化的 logits

`torch.multinomial` 需要输入概率，而不是 logits。必须先应用 softmax：

```python
# 正确
probs = torch.softmax(logits / temperature, dim=-1)
next_idx = torch.multinomial(probs, 1)

# 错误：直接使用 logits
next_idx = torch.multinomial(logits / temperature, 1)
```

### 7.4 生成时忘记将张量移动到设备

确保所有张量位于同一设备（CPU/GPU）：

```python
# 正确
input_tensor = input_tensor.to(device)

# 错误：模型在 GPU，输入在 CPU，会报错
```

### 7.5 损失函数未忽略填充字符（可选）

如果序列中包含填充字符，应使用 `ignore_index` 忽略它们，但本任务中所有序列长度一致，无填充。

---

## 8. 可能的改进方向

- **堆叠更多 LSTM 层**：增加深度捕获更复杂的依赖关系。
- **双向 LSTM**：但生成任务通常只用单向 LSTM（不能利用未来信息）。
- **教师强制（Teacher Forcing）**：训练时使用真实目标作为输入，推理时用预测。
- **注意力机制**：可以增强长期依赖的捕获能力。
- **用 GRU 替代 LSTM**：参数更少，训练更快。
- **字符级 CNN 或 Transformer**：捕获不同尺度的字符模式。

---

## 9. 重点总结

任务本质：

```
字符级 LSTM 文本生成 = 学习字符序列的条件概率分布 P(字符_t | 字符_{t-1}, ..., 字符_{t-seq_length})
```

数据流程：

```
文本文件 → 字符映射表 → 整数编码 → 滑动窗口创建 (输入, 目标) 对 → DataLoader
```

模型结构：

```
嵌入层 (Embedding) → LSTM 层 → 全连接层 (输出词汇表大小)
```

训练要点：

```
使用交叉熵损失，每个批次重置隐藏状态，梯度裁剪
```

生成要点：

```
自回归生成，温度采样控制多样性
```

总结：

```
字符级 LSTM 文本生成是一个简单而强大的序列建模入门任务。通过这个模型，
我们可以深入理解 RNN/LSTM 的工作原理、序列数据的处理方式以及如何从训练好的模型中采样生成新内容。
该技术可用于诗歌生成、代码生成、对话系统等多种创意应用。
```

# PyTorch TorchText 

TorchText 是 PyTorch 生态中专门用于自然语言处理（NLP）的文本处理库，旨在简化文本数据的加载、预处理和迭代流程。无论是做情感分析、机器翻译还是文本分类，TorchText 都能帮你高效地完成数据准备工作。
TorchText 主要包含三大组件：

- **Field 对象**：指定如何处理某个字段，如分词方法、是否转小写、起始/结束字符、补全字符及词典等。
- **Dataset 类**：用于加载数据，继承自 PyTorch 的 Dataset，提供 `splits` 方法同时读取训练集、验证集、测试集。
- **迭代器（Iterator）** ：包括标准 `Iterator`、`BucketIterator`（按长度分批，提高填充效率）和 `BPTTIterator`（用于语言模型）。

> **⚠️ 版本说明**：TorchText 在 0.12.0 版本后引入了全新的数据处理管道，API 发生了较大变化。旧版（0.9.x 及以前）使用 `torchtext.legacy` 模块，新版（0.12.0+）采用更简洁的 API。本文主要基于新版 TorchText 进行讲解，同时会标注新旧版本的区别。


## 1. 自定义数据集：加载 CSV / JSON / TSV 文件

### 1.1 TorchText 支持的数据格式

TorchText 原生支持三种数据格式：

- **CSV**（逗号分隔值）
- **TSV**（制表符分隔值）
- **JSON**（JSON Lines 格式）

其中，JSON 格式被官方推荐为最佳选择。

### 1.2 Field 对象：定义数据处理方式

在加载数据之前，首先需要定义 `Field` 对象，它决定了每个字段如何处理。

```python
from torchtext.legacy import data  # 旧版 API

# 定义文本字段
TEXT = data.Field(
    sequential=True,      # 是否为序列数据
    tokenize='spacy',     # 分词器
    lower=True,           # 转小写
    include_lengths=True, # 是否返回序列长度
    batch_first=True,     # batch 作为第一维度
)

# 定义标签字段
LABEL = data.LabelField(
    dtype=torch.float,    # 标签的数据类型
    sequential=False,     # 标签不是序列
)
```

**Field 的常用参数**：

| 参数 | 说明 |
|------|------|
| `sequential` | 是否为序列数据，默认 True |
| `use_vocab` | 是否使用词典，默认 True |
| `init_token` | 文本起始字符，如 `<sos>` |
| `eos_token` | 文本结束字符，如 `<eos>` |
| `fix_length` | 固定序列长度，不够则用 `pad_token` 补齐 |
| `lower` | 是否转小写 |
| `tokenize` | 分词函数，默认 `str.split` |
| `batch_first` | batch 是否作为第一维度 |
| `pad_token` | 补全字符，默认为 `<pad>` |
| `unk_token` | 替换未知词的字符，默认为 `<unk>` |

### 1.3 加载 JSON 数据

JSON 数据必须采用 **JSON Lines 格式**，即每行一个独立的 JSON 对象：

```json
{"name": "John", "location": "United Kingdom", "quote": ["i", "love", "the", "united kingdom"]}
{"name": "Mary", "location": "United States", "quote": ["i", "want", "more", "telescopes"]}
```

**定义字段并加载数据**：

```python
from torchtext.legacy import data

# 1. 定义 Field
NAME = data.Field()
SAYING = data.Field()
PLACE = data.Field()

# 2. 创建字段映射字典
# 键 → JSON 对象的键
# 值 → (batch 中的属性名, Field 对象)
fields = {
    'name': ('n', NAME),      # batch.n 可访问 name
    'location': ('p', PLACE), # batch.p 可访问 location
    'quote': ('s', SAYING),   # batch.s 可访问 quote
}

# 3. 使用 TabularDataset.splits 加载数据
train_data, test_data = data.TabularDataset.splits(
    path='data',              # 数据文件所在目录
    train='train.json',       # 训练集文件名
    test='test.json',         # 测试集文件名
    format='json',            # 数据格式
    fields=fields,            # 字段映射
)
```

**JSON 加载的注意事项**：

- `fields` 字典中键的顺序不重要，只要与 JSON 键匹配即可。
- 字段名不必与 JSON 键相同（如用 `PLACE` 表示 `location`）。
- 不需要使用的 JSON 字段可以忽略（如示例中的 `age`）。
- 如果 JSON 字段值是**字符串**，会应用分词；如果是**列表**，则不进行分词。建议预先将文本分词为列表，可节省处理时间。

### 1.4 加载 CSV / TSV 数据

CSV 和 TSV 的加载方式类似，只需改变 `format` 参数。

**CSV 数据格式示例**：

```csv
text,label
"I love this movie!",pos
"Terrible film.",neg
```

**加载 CSV 数据**：

```python
from torchtext.legacy import data

# 1. 定义 Field
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(dtype=torch.float)

# 2. 定义字段列表（顺序必须与 CSV 列顺序一致）
fields = [
    ('text', TEXT),
    ('label', LABEL),
]

# 3. 加载数据
train_data, test_data = data.TabularDataset.splits(
    path='data',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=fields,
    skip_header=True,  # 跳过 CSV 表头
)
```

**加载 TSV 数据**（只需将 `format` 改为 `'tsv'`）：

```python
train_data, test_data = data.TabularDataset.splits(
    path='data',
    train='train.tsv',
    test='test.tsv',
    format='tsv',        # 改为 tsv
    fields=fields,
    skip_header=True,
)
```

### 1.5 数据集划分

如果只有一个数据集文件，可以使用 `split()` 方法进行划分：

```python
# 按比例划分为 70% 训练、15% 验证、15% 测试
train_data, valid_data, test_data = dataset.split(
    split_ratio=[0.7, 0.15, 0.15]
)
```

### 1.6 构建词汇表（Vocabulary）

词汇表将单词映射为数字索引，是 NLP 模型训练的关键步骤。

```python
# 基于训练集构建词汇表
TEXT.build_vocab(
    train_data,
    max_size=10000,              # 词汇表最大大小
    vectors='glove.6B.100d',     # 使用预训练词向量
    min_freq=5,                  # 只保留出现次数 >= 5 的词
)

LABEL.build_vocab(train_data)

# 查看词汇表大小
print(f"词汇表大小: {len(TEXT.vocab)}")
print(f"标签类别: {LABEL.vocab.itos}")
```

### 1.7 创建迭代器（Iterator）

迭代器负责将数据分批提供给模型。

```python
from torchtext.legacy import data

# BucketIterator：按序列长度分批，提高填充效率
train_iter, valid_iter = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=64,
    sort_key=lambda x: len(x.text),  # 按文本长度排序
    shuffle=True,                    # 训练集打乱
    device=device,                   # CPU 或 GPU
)

# 标准 Iterator（不按长度分批）
train_iter = data.Iterator(
    train_data,
    batch_size=64,
    shuffle=True,
    device=device,
)
```

**三种迭代器的区别**：

| 迭代器 | 说明 |
|--------|------|
| `Iterator` | 标准迭代器，简单分批 |
| `BucketIterator` | 按序列长度分批，减少填充量，提高效率 |
| `BPTTIterator` | 基于时间的反向传播，用于语言模型 |


## 2. 内置数据集

TorchText 提供了多种常用的内置 NLP 数据集，可以直接加载使用。

### 2.1 可用内置数据集列表

**语言模型数据集**：

- `WikiText2`
- `WikiText103`
- `PennTreebank`
- `EnWik9`

**文本分类数据集**：

- `AG_NEWS`：新闻分类
- `SogouNews`：搜狗新闻
- `DBpedia`：维基百科分类
- `IMDB`：电影评论情感分析
- `SST2`：斯坦福情感树库

### 2.2 使用内置数据集（新版 API）

新版 TorchText（0.12.0+）的数据集 API 更加简洁，返回的是迭代器：

```python
import torch
from torchtext.datasets import IMDB

# 加载 IMDB 数据集
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# 查看第一个样本
first_example = next(iter(train_iter))
print(f"Label: {first_example[0]}")   # 'pos' 或 'neg'
print(f"Text: {first_example[1][:200]}...")  # 评论内容前 200 字符
```

**加载 AG_NEWS 数据集**：

```python
from torchtext.datasets import AG_NEWS

train_iter, test_iter = AG_NEWS()
# AG_NEWS 的每个样本是 (label, text) 元组
```

### 2.3 使用内置数据集（旧版 API）

旧版 TorchText（0.9.x 及以前）使用 `splits` 方法：

```python
from torchtext.legacy import data
from torchtext.legacy import datasets

# 定义 Field
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.LabelField(sequential=False)

# 加载 IMDB 数据集
train, test = datasets.IMDB.splits(TEXT, LABEL)

# 划分验证集
train, valid = train.split(split_ratio=0.8)

# 构建词汇表
TEXT.build_vocab(train, vectors='glove.6B.300d')
LABEL.build_vocab(train)

# 创建迭代器
train_iter, valid_iter, test_iter = data.BucketIterator.splits(
    (train, valid, test),
    batch_size=64,
    device=device,
)
```

### 2.4 构建词汇表（新版 API）

新版 API 使用 `get_tokenizer` 和 `build_vocab_from_iterator` 构建词汇表：

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 1. 定义分词器
tokenizer = get_tokenizer('basic_english')

# 2. 定义生成 token 的迭代器函数
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# 3. 构建词汇表
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(
    yield_tokens(train_iter),
    specials=["<unk>"],   # 特殊标记
    min_freq=10,          # 最低词频
)

# 设置未知词的默认索引
vocab.set_default_index(vocab["<unk>"])

print(f"词汇表大小: {len(vocab)}")
print(f"'movie' 的索引: {vocab['movie']}")
```


## 3. 从文本文件创建 Dataset

有时数据存储在纯文本文件中（如 `.txt` 文件），每个文件代表一个样本，或者整个语料库在一个大文件中。本节介绍如何处理这类数据。

### 3.1 使用 LanguageModelingDataset（语言模型）

对于语言模型任务（如文本生成），数据通常是一个或多个大文本文件。TorchText 提供了 `LanguageModelingDataset` 来处理这类数据。

```python
from torchtext.legacy import data
from torchtext.legacy import datasets

# 定义 Field
TEXT = data.Field(
    tokenize='spacy',
    lower=True,
    init_token='<sos>',
    eos_token='<eos>',
)

# 加载语言模型数据集
train, val, test = datasets.LanguageModelingDataset.splits(
    path='data/wikitext-2',
    train='wiki.train.tokens',
    validation='wiki.valid.tokens',
    test='wiki.test.tokens',
    text_field=TEXT,
)

# 构建词汇表
TEXT.build_vocab(train, max_size=50000)

# 创建 BPTT 迭代器（用于语言模型）
from torchtext.legacy.data import BPTTIterator

train_iter = BPTTIterator(
    train,
    batch_size=20,
    bptt_len=35,          # 反向传播的时间步长
    device=device,
)
```

### 3.2 使用 TabularDataset 加载 TSV 文本文件

对于分类任务，可以将文本和标签组织成 TSV 格式：

```python
from torchtext.legacy import data

# 定义 Field
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(sequential=False)

# 字段列表
fields = [('Text', TEXT), ('Label', LABEL)]

# 加载 TSV 数据
train_data, test_data = data.TabularDataset.splits(
    path='data',
    train='train.tsv',
    test='test.tsv',
    format='tsv',
    fields=fields,
)
```

### 3.3 从原始文本文件构建 Dataset

如果数据是纯文本文件（每个文件一个样本），可以手动创建 `Dataset` 对象：

```python
from torchtext.legacy import data
import os

# 定义 Field
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(sequential=False)

# 读取文本文件并创建 Example
def read_files(directory, label):
    examples = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            examples.append(data.Example.fromlist([text, label], fields))
    return examples

# 创建训练集和测试集
train_examples = read_files('data/train/pos', 'pos') + read_files('data/train/neg', 'neg')
test_examples = read_files('data/test/pos', 'pos') + read_files('data/test/neg', 'neg')

# 构建 Dataset
train_data = data.Dataset(train_examples, fields)
test_data = data.Dataset(test_examples, fields)
```

### 3.4 从 Pandas DataFrame 创建 Dataset

如果数据已经在 Pandas DataFrame 中，可以直接创建 Dataset：

```python
import pandas as pd
from torchtext.legacy import data

# 读取 CSV 到 DataFrame
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# 定义 Field
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(sequential=False)

fields = [('text', TEXT), ('label', LABEL)]

# 从 DataFrame 创建 Example
train_examples = [
    data.Example.fromlist([row['text'], row['label']], fields)
    for _, row in train_df.iterrows()
]

train_data = data.Dataset(train_examples, fields)
```


## 4. 新旧版本 API 对比

| 功能 | 旧版 (0.9.x) | 新版 (0.12.0+) |
|------|-------------|----------------|
| 导入方式 | `from torchtext.legacy import data` | `from torchtext.data.utils import ...` |
| 数据集加载 | `datasets.IMDB.splits(TEXT, LABEL)` | `IMDB(split='train')` |
| 词汇表构建 | `TEXT.build_vocab(train_data)` | `build_vocab_from_iterator()` |
| 分词器 | `Field(tokenize='spacy')` | `get_tokenizer('spacy')` |

> **建议**：新项目优先使用新版 API（0.12.0+），更简洁高效。旧版项目如需维护，可使用 `torchtext.legacy` 保持兼容。


## 5. 完整示例：从零开始构建文本分类管道

以下是一个完整的情感分析示例，整合了本节所有知识点：

```python
import torch
from torchtext.legacy import data
from torchtext.legacy import datasets

# 1. 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 定义 Field
TEXT = data.Field(
    tokenize='spacy',
    lower=True,
    include_lengths=True,
    batch_first=True,
)

LABEL = data.LabelField(
    dtype=torch.float,
    sequential=False,
)

# 3. 加载内置 IMDB 数据集
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# 4. 划分验证集
train_data, valid_data = train_data.split(split_ratio=0.8)

# 5. 构建词汇表
TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

# 6. 创建迭代器
train_iter, valid_iter, test_iter = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=64,
    sort_key=lambda x: len(x.text),
    shuffle=True,
    device=device,
)

# 7. 查看数据样本
batch = next(iter(train_iter))
print(f"文本批次形状: {batch.text[0].shape}")  # (batch, seq_len)
print(f"标签: {batch.label}")

# 8. 定义简单的 LSTM 模型（示例）
class SimpleLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1])

# 9. 训练（略）
```

## 6. 重点总结

**TorchText 的核心组件**：

```
Field（定义字段处理方式） → Dataset（加载数据） → Iterator（分批迭代）
```

**自定义数据集加载流程**：

```
准备数据（CSV/TSV/JSON）→ 定义 Field → 使用 TabularDataset.splits 加载 → 构建词汇表 → 创建迭代器
```

**JSON 格式的优点**：

```
支持嵌套结构、字段顺序无关、可忽略不需要的字段、预分词后可节省处理时间
```

**内置数据集**：

```
直接导入使用，无需下载和预处理，支持 IMDB、AG_NEWS、WikiText2 等常用数据集
```

**新旧版本差异**：

```
旧版（0.9.x）：torchtext.legacy，使用 Field + 数据集 .splits 方法
新版（0.12.0+）：更简洁的 API，使用 get_tokenizer + build_vocab_from_iterator
```

总结：

```
TorchText 是 PyTorch NLP 任务中不可或缺的数据处理工具。掌握 Field 的定义、
TabularDataset 的加载方式、词汇表的构建以及迭代器的使用，
可以帮助你高效地处理各种文本数据格式，将更多精力放在模型设计上。
```


# PyTorch Einsum 指南

Einsum（爱因斯坦求和约定）是一种强大的张量运算表示法，最初源于物理学中的张量计算，如今在 NumPy、PyTorch、TensorFlow 等主流深度学习框架中均有实现。它用一个简洁的字符串就能表达复杂的张量操作（如点积、外积、转置、矩阵-向量乘法、矩阵-矩阵乘法、迹、张量收缩等），无需记忆各种函数名，让代码更简洁、可读性更强。

---

## 1. 核心概念：爱因斯坦求和约定

爱因斯坦求和约定的核心规则是：**在同一个表达式中，如果某个索引出现两次，则对该索引进行求和（收缩）**。

例如，矩阵乘法 `C[i, j] = sum_k A[i, k] * B[k, j]` 在爱因斯坦记法中写作：

```
C[i, j] = A[i, k] B[k, j]
```

索引 `k` 出现两次（在 A 的第二个维度和 B 的第一个维度），所以自动对 `k` 求和。结果 `C` 的维度由剩余的索引 `i` 和 `j` 决定。

Einsum 函数允许我们用**字符串**描述这个求和过程，而无需显式写出循环或 `matmul`、`sum` 等函数。

---

## 2. Einsum 基本语法

在 PyTorch 中，调用方式为：

```python
torch.einsum(equation, *operands)
```

其中 `equation` 是字符串，描述张量的维度关系。它的格式为：

```
"输入1的维度标记, 输入2的维度标记, ... -> 输出维度标记"
```

- 用逗号分隔各个输入的维度标记
- 用 `->` 分隔输入和输出
- 忽略不存在的维度（即需要求和的维度）

**维度标记规则**：通常使用小写字母表示维度，如 `i, j, k, l` 等。相同字母在不同张量中表示相同的维度，出现两次的字母会自动求和。

例如，矩阵乘法 `A (shape m×n)` 和 `B (n×p)` 相乘得到 `C (m×p)`：

```python
C = torch.einsum('ik,kj->ij', A, B)
# 等价于 C = torch.matmul(A, B)
```

---

## 3. 常见张量运算的 Einsum 实现

### 3.1 转置（Transpose）

```python
import torch

A = torch.randn(3, 4)

# 转置
B = torch.einsum('ij->ji', A)
# 等价于 A.T
```

### 3.2 求和（Sum）

```python
# 对所有元素求和
total = torch.einsum('ij->', A)  # 等价于 A.sum()

# 对列求和（按行维度保留）
col_sum = torch.einsum('ij->j', A)  # 等价于 A.sum(dim=0)

# 对行求和（按列维度保留）
row_sum = torch.einsum('ij->i', A)  # 等价于 A.sum(dim=1)
```

### 3.3 内积 / 点积（Dot Product）

```python
a = torch.randn(3)
b = torch.randn(3)

# 向量内积
dot = torch.einsum('i,i->', a, b)  # 等价于 torch.dot(a, b)
# 或 torch.sum(a * b)
```

### 3.4 矩阵-向量乘法（Matrix-Vector Multiplication）

```python
M = torch.randn(3, 4)
v = torch.randn(4)

# M @ v
y = torch.einsum('ij,j->i', M, v)  # 等价于 torch.mv(M, v)
```

### 3.5 矩阵乘法（Matrix-Matrix Multiplication）

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# A @ B
C = torch.einsum('ik,kj->ij', A, B)  # 等价于 torch.mm(A, B)
```

### 3.6 批量矩阵乘法（Batched Matrix Multiplication）

```python
A = torch.randn(10, 3, 4)  # 批量大小 10，每个矩阵 3x4
B = torch.randn(10, 4, 5)  # 批量大小 10，每个矩阵 4x5

# 批量矩阵乘法：每个批次独立相乘
C = torch.einsum('bij,bjk->bik', A, B)  # 等价于 torch.bmm(A, B)
```

### 3.7 外积（Outer Product）

```python
a = torch.randn(3)
b = torch.randn(4)

# 外积
O = torch.einsum('i,j->ij', a, b)  # 等价于 torch.outer(a, b)
```

### 3.8 Hadamard 乘积（逐元素乘积）

```python
A = torch.randn(3, 4)
B = torch.randn(3, 4)

# 逐元素乘积
H = torch.einsum('ij,ij->ij', A, B)  # 等价于 A * B
```

### 3.9 迹（Trace）

```python
M = torch.randn(4, 4)

# 迹：对角元素之和
trace = torch.einsum('ii->', M)  # 等价于 torch.trace(M)
```

### 3.10 对角元素提取

```python
M = torch.randn(4, 4)

# 提取对角元素
diag = torch.einsum('ii->i', M)  # 等价于 torch.diag(M)
```

### 3.11 张量收缩（Tensor Contraction）

例如，两个 3D 张量在中间两个维度上收缩：

```python
A = torch.randn(2, 3, 5)
B = torch.randn(2, 5, 4)

# 收缩最后一个维度和第一个维度
C = torch.einsum('ijk,ikl->ijl', A, B)  # 结果形状为 (2, 3, 4)
# 等价于 torch.einsum('ijk,ikm->ijm', A, B)（如果维度标记不同）
```

更复杂的例子：

```python
A = torch.randn(3, 4, 5)
B = torch.randn(5, 6)

# A 的最后一维和 B 的第一维收缩，保留 A 的前两维和 B 的第二维
C = torch.einsum('ijk,kl->ijl', A, B)  # 结果形状 (3, 4, 6)
```

### 3.12 元素相乘并求和（Reduction）

```python
A = torch.randn(3, 4, 5)
B = torch.randn(3, 4, 5)

# 沿最后一维相乘并求和（保留前两维）
S = torch.einsum('ijk,ijk->ij', A, B)  # 等价于 (A * B).sum(dim=2)
```

### 3.13 复杂操作：如注意力机制中的 Q、K、V 计算

对于自注意力，假设 `Q`、`K`、`V` 形状均为 `(batch, seq_len, dim)`，计算注意力分数：

```python
Q = torch.randn(2, 4, 8)  # batch=2, seq=4, dim=8
K = torch.randn(2, 4, 8)
V = torch.randn(2, 4, 8)

# 计算注意力分数 (batch, seq_q, seq_k)
scores = torch.einsum('bik,bjk->bij', Q, K)  # 等价于 torch.bmm(Q, K.transpose(1,2))
# 然后 softmax 后与 V 相乘
attn = torch.einsum('bij,bjk->bik', softmax_scores, V)
```

---

## 4. Einsum 与常规函数的性能比较

在大多数情况下，Einsum 的性能与专用的函数（如 `matmul`、`sum`）相当，因为底层会调用优化的 BLAS 库。但在某些情况下，Einsum 可能稍慢，因为它需要解析字符串并生成执行计划，但对于大多数中型到大型张量，差异可以忽略。

**优势**：
- 代码简洁，无需记忆多种函数名。
- 易于表达复杂的张量操作，尤其是涉及多维收缩的情况。
- 可读性好（对于熟悉 Einstein 记法的人来说）。

**劣势**：
- 对初学者可能不易理解。
- 对于简单的操作（如转置、求和），专用函数更直观。
- 在某些极端情况下，可能不如手写优化的循环或专用函数快（但很少见）。

---

## 5. 高级用法：Einsum 的可选参数

Einsum 还支持一些可选参数，例如控制求和方式或优化策略。

### 5.1 PyTorch 中的 `torch.einsum`

```python
# 基本用法
result = torch.einsum('ij,jk->ik', A, B)

# 使用 optimize 参数（自动选择最优路径）
result = torch.einsum('ij,jk->ik', A, B, optimize=True)
```

### 5.2 NumPy 中的 `np.einsum`

NumPy 提供了 `optimize` 参数，可以显著加速张量收缩。

```python
import numpy as np

A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# 自动优化
C = np.einsum('ij,jk->ik', A, B, optimize=True)
```

### 5.3 TensorFlow 中的 `tf.einsum`

TensorFlow 的用法类似：

```python
import tensorflow as tf

A = tf.random.normal([3, 4])
B = tf.random.normal([4, 5])
C = tf.einsum('ij,jk->ik', A, B)
```

---

## 6. 常见错误与注意事项

### 6.1 维度标签混淆

确保相同字母表示相同的维度大小。例如，`'ij,ik->ik'` 要求 A 和 B 的第一维大小相同（`i`），而第二维可以不同。

### 6.2 输出维度标签的顺序决定输出形状

输出的标签顺序决定了张量的形状。例如，`'ij,kl->ikjl'` 会输出形状为 `(i, k, j, l)`。

### 6.3 未出现在输出中的标签自动求和

如果在输入中出现了但在输出中被省略的标签，Einsum 会自动对这些维度求和。

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# 输出为 'ij'，省略 'k'，所以自动对 k 求和
C = torch.einsum('ik,kj->ij', A, B)  # 正常矩阵乘法
```

### 6.4 不支持修改输入张量

Einsum 总是返回新的张量，不会修改原始数据。

### 6.5 对于标量结果，输出为空

```python
dot = torch.einsum('i,i->', a, b)  # 输出是标量，'->' 后为空
```

### 6.6 性能提示

对于复杂的收缩，使用 `optimize=True` 可以在 NumPy 和 PyTorch 中显著提升性能，因为它会寻找最优的收缩顺序。

---

## 7. 实际应用示例

### 7.1 双线性层（Bilinear Layer）

```python
# 输入 x: (batch, in_features)
# 权重 W: (in_features, out_features)
# 偏置 b: (out_features)
y = torch.einsum('bi,io->bo', x, W) + b  # 等价于 x @ W + b
```

### 7.2 注意力机制中的缩放点积

```python
# Q, K, V: (batch, heads, seq_len, dim)
scores = torch.einsum('bhld,bhmd->bhlm', Q, K) / (dim ** 0.5)
attn_weights = torch.softmax(scores, dim=-1)
output = torch.einsum('bhlm,bhmd->bhld', attn_weights, V)
```

### 7.3 计算协方差矩阵

```python
# 数据矩阵 X: (batch, features)
X_centered = X - X.mean(dim=0, keepdim=True)
cov = torch.einsum('bi,bj->ij', X_centered, X_centered) / (X.size(0) - 1)
```

### 7.4 图像卷积的简化表示

虽然实际卷积通常用专用函数，但 Einsum 可以表达一些卷积的变体：

```python
# 假设输入 I: (batch, in_ch, h, w)
# 卷积核 K: (out_ch, in_ch, k, k)
# 输出 O: (batch, out_ch, h_out, w_out)
# 这里只是为了演示，实际应使用 conv2d
```

---

## 8. 与广播（Broadcasting）的关系

Einsum 自动处理广播，但要求维度标签一致。例如：

```python
A = torch.randn(3, 1, 5)  # 第2维为1，可广播
B = torch.randn(4, 5)
# 想得到 (3, 4, 5) 的逐元素乘积
# 可以使用 'ijk,ik->ijk'，但需要确保 A 和 B 的对应维度匹配或可广播
# 注意：einsum 不会自动广播，所以必须显式匹配维度
```

通常，如果涉及广播，使用常规的广播乘法 `*` 更简单。

---

## 9. 进阶：Einsum 的优化路径

在 PyTorch 1.8+ 中，`torch.einsum` 支持 `optimize` 参数，可以选择默认优化或手动指定。对于包含多个输入、多个求和维度的张量收缩，优化路径可以减少内存使用和计算量。

```python
# 自动选择最优路径
result = torch.einsum(equation, *operands, optimize=True)
```

在 NumPy 中，`np.einsum` 也有 `optimize` 参数，可以设为 `'greedy'` 或 `'optimal'`。

---

## 10. 总结

**Einsum 的核心优势**：

```
一个函数，统一所有张量操作
```

**适合使用 Einsum 的场景**：

```
1. 复杂的张量收缩（如注意力机制）
2. 需要同时进行多个操作的场合（如求和与乘积）
3. 代码简洁性优先，且团队成员熟悉 Einstein 记法
```

**不适合的场景**：

```
1. 简单的元素级操作（如加、乘、激活函数）
2. 对性能极度敏感且可用专用函数替代时
3. 项目中有大量对 Einsum 不熟悉的同事
```

**核心公式记忆法**：

```
输出 = 对重复索引求和，保留未重复的索引
```


# PyTorch Seq2Seq 机器翻译与注意力机制完整笔记

机器翻译（Machine Translation）是 NLP 中经典的序列到序列（Seq2Seq）任务，目标是将一种语言的句子翻译成另一种语言。本笔记涵盖基础 Seq2Seq 模型和带注意力机制的 Seq2Seq 模型，从零实现一个英译法的翻译系统。

---

## 1. Seq2Seq 核心思想：编码器-解码器架构

Seq2Seq 模型采用编码器-解码器（Encoder-Decoder）架构，最早由 Sutskever 等人于 2014 年提出。

```
输入序列（英文）: "Hello how are you"
    ↓
编码器（Encoder）：RNN/LSTM/GRU 逐词读取，生成上下文向量
    ↓
上下文向量（Context Vector）：压缩了整个输入序列的信息
    ↓
解码器（Decoder）：RNN/LSTM/GRU 逐词生成输出序列
    ↓
输出序列（法语）: "Bonjour comment allez-vous"
```

### 1.1 编码器（Encoder）

编码器读取输入序列，输出一个**上下文向量**（Context Vector）：

```
输入: "Hello" → "how" → "are" → "you"
    ↓
RNN 逐步处理，最终输出一个向量
    ↓
上下文向量（Encoder 的最终隐藏状态）
```

这个上下文向量是整个输入序列的“语义摘要”，传递给解码器作为初始状态。

### 1.2 解码器（Decoder）

解码器接收上下文向量，自回归地生成输出序列：

```
上下文向量 → 生成第一个词 → 生成第二个词 → ... → 生成 <end> 停止
```

- 解码器每一步输出一个词的概率分布
- 选择概率最大的词作为输出
- 将输出的词作为下一步的输入（自回归）
- 直到生成 `<end>` 标记或达到最大长度

### 1.3 训练：Teacher Forcing

训练时，解码器的每一步输入是**真实的目标词**，而不是模型自己的预测。这样可以加速收敛，避免错误累积。

```
输入: "<start>" → "Bonjour" → "comment" → "allez-vous"
目标: "Bonjour" → "comment" → "allez-vous" → "<end>"
```

---

## 2. 数据准备与预处理

### 2.1 数据加载

使用 TorchText 加载英法翻译数据集：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets
import spacy

# 加载分词器
spacy_en = spacy.load('en_core_web_sm')
spacy_fr = spacy.load('fr_core_news_sm')

# 分词函数
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

# 定义 Field
SRC = data.Field(
    tokenize=tokenize_en,
    init_token='<sos>',
    eos_token='<eos>',
    lower=True,
    batch_first=True,
)

TRG = data.Field(
    tokenize=tokenize_fr,
    init_token='<sos>',
    eos_token='<eos>',
    lower=True,
    batch_first=True,
)

# 加载 Multi30k 数据集（英德翻译）
# 实际可以替换为其他翻译数据集
train_data, valid_data, test_data = datasets.Multi30k.splits(
    exts=('.en', '.de'),
    fields=(SRC, TRG),
)

# 构建词汇表
SRC.build_vocab(train_data, min_freq=2, max_size=10000)
TRG.build_vocab(train_data, min_freq=2, max_size=10000)
```

### 2.2 创建迭代器

```python
from torchtext.legacy.data import BucketIterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=64,
    sort_key=lambda x: len(x.src),
    shuffle=True,
    device=device,
)

# 查看一个批次
batch = next(iter(train_iter))
print(f"源序列形状: {batch.src.shape}")  # (batch, src_len)
print(f"目标序列形状: {batch.trg.shape}")  # (batch, trg_len)
```

---

## 3. 基础 Seq2Seq 模型实现

### 3.1 编码器

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch_size, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, embed_dim)
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: (batch, src_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)
        # cell:   (num_layers, batch, hidden_dim)
        return hidden, cell
```

### 3.2 解码器（基础版）

基础解码器只使用编码器的**最终隐藏状态**作为初始状态：

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, cell):
        # trg: (batch_size, trg_len)
        # hidden: (num_layers, batch, hidden_dim)
        # cell:   (num_layers, batch, hidden_dim)

        embedded = self.dropout(self.embedding(trg))  # (batch, trg_len, embed_dim)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # outputs: (batch, trg_len, hidden_dim)

        predictions = self.fc(outputs)  # (batch, trg_len, vocab_size)
        return predictions, hidden, cell
```

### 3.3 完整 Seq2Seq 模型

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch, src_len)
        # trg: (batch, trg_len)

        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        # 存储所有时间步的输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 编码器输出最终隐藏状态
        hidden, cell = self.encoder(src)

        # 解码器的第一个输入是 <sos>
        input = trg[:, 0]  # (batch,)

        for t in range(1, trg_len):
            # 解码器单步前向传播
            output, hidden, cell = self.decoder(
                input.unsqueeze(1),  # (batch, 1)
                hidden,
                cell,
            )
            # output: (batch, 1, vocab_size)
            outputs[:, t, :] = output.squeeze(1)

            # Teacher Forcing 或使用预测
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)  # (batch, 1)

            if teacher_force:
                input = trg[:, t]  # 使用真实目标词
            else:
                input = top1.squeeze(1)  # 使用模型预测

        return outputs
```

---

## 4. 训练流程

### 4.1 训练循环

```python
def train(model, iterator, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        # 前向传播
        output = model(src, trg)

        # 计算损失（忽略 <pad>）
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # 去掉 <sos>
        trg = trg[:, 1:].reshape(-1)  # 去掉 <sos>

        loss = criterion(output, trg)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
```

### 4.2 评估循环

```python
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            src = batch.src
            trg = batch.trg

            # 评估时 teacher_forcing_ratio = 0
            output = model(src, trg, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
```

### 4.3 完整的训练过程

```python
# 超参数
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.5
LR = 0.001
NUM_EPOCHS = 10

# 初始化模型
encoder = Encoder(len(SRC.vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(len(TRG.vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练
best_valid_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    train_loss = train(model, train_iter, optimizer, criterion)
    valid_loss = evaluate(model, valid_iter, criterion)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'seq2seq_best.pt')

print("训练完成！")
```

---

## 5. 推理（翻译）

```python
def translate_sentence(model, sentence, src_field, trg_field, max_len=50):
    model.eval()

    # 1. 分词并转换为索引
    tokens = [tok.text.lower() for tok in spacy_en.tokenizer(sentence)]
    tokens = ['<sos>'] + tokens + ['<eos>']
    indices = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.tensor(indices).unsqueeze(0).to(device)  # (1, len)

    # 2. 编码
    hidden, cell = model.encoder(src_tensor)

    # 3. 解码
    trg_indices = [trg_field.vocab.stoi['<sos>']]

    for _ in range(max_len):
        input_tensor = torch.tensor([trg_indices[-1]]).unsqueeze(0).to(device)
        output, hidden, cell = model.decoder(input_tensor, hidden, cell)

        pred_token = output.argmax(2).item()
        trg_indices.append(pred_token)

        if pred_token == trg_field.vocab.stoi['<eos>']:
            break

    # 4. 转换为文本
    translation = [trg_field.vocab.itos[idx] for idx in trg_indices[1:-1]]
    return ' '.join(translation)

# 测试翻译
sentence = "I love programming in Python"
translation = translate_sentence(model, sentence, SRC, TRG)
print(f"英文: {sentence}")
print(f"法语: {translation}")
```

---

## 6. 注意力机制（Attention）

基础 Seq2Seq 模型将整个输入序列压缩成一个固定长度的上下文向量，当输入序列较长时，这个向量会丢失早期位置的信息。**注意力机制**通过让解码器在每一步动态地关注输入序列的不同位置来解决这个问题。

### 6.1 注意力核心思想

```
解码器生成第 t 个词时：
    1. 查看编码器所有时间步的输出
    2. 计算每个编码器输出与当前解码器状态的"相关性"（注意力权重）
    3. 用这些权重加权求和编码器输出，得到上下文向量
    4. 用上下文向量辅助当前词的预测
```

### 6.2 注意力权重的计算

对于解码器的每个时间步 t 和编码器的每个位置 i：

```
score_ti = f(h_dec_t, h_enc_i)  # 计算相关性
alpha_ti = softmax(score_t1, score_t2, ..., score_tn)  # 归一化为概率
context_t = sum_i alpha_ti * h_enc_i  # 加权求和
```

### 6.3 注意力评分函数的变体

| 方法 | 公式 | 说明 |
|------|------|------|
| Dot | `h_dec · h_enc` | 简单内积，要求维度相同 |
| General | `h_dec^T · W · h_enc` | 可学习线性变换 |
| Concat（Bahdanau） | `v^T · tanh(W₁·h_dec + W₂·h_enc)` | 最早用于 NMT |

---

## 7. 带注意力机制的 Seq2Seq

### 7.1 注意力模块

```python
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Bahdanau 风格：使用 concat 评分
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: (batch, hidden_dim) - 解码器当前时间步的隐藏状态
        # encoder_outputs: (batch, src_len, hidden_dim)

        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        # 扩展隐藏状态以匹配编码器输出的长度
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch, src_len, hidden_dim)

        # 拼接并计算注意力得分
        energy = torch.tanh(self.attn(
            torch.cat([hidden_expanded, encoder_outputs], dim=2)
        ))  # (batch, src_len, hidden_dim)

        attention = self.v(energy).squeeze(2)  # (batch, src_len)

        # 如果有 mask，将填充位置的注意力设为 -inf
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # Softmax 得到注意力权重
        attn_weights = torch.softmax(attention, dim=1)  # (batch, src_len)

        # 上下文向量：注意力权重加权求和
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        # (batch, hidden_dim)

        return context, attn_weights
```

### 7.2 带注意力的解码器

```python
class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim,  # 输入：词嵌入 + 上下文向量
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # 拼接 LSTM 输出和上下文
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, mask=None):
        # input: (batch, 1) - 当前输入的词索引
        # hidden: (num_layers, batch, hidden_dim)
        # cell: (num_layers, batch, hidden_dim)

        # 取最后一层的隐藏状态用于注意力计算
        hidden_last = hidden[-1]  # (batch, hidden_dim)

        # 计算注意力上下文向量
        context, attn_weights = self.attention(hidden_last, encoder_outputs, mask)

        # 嵌入输入词
        embedded = self.dropout(self.embedding(input))  # (batch, 1, embed_dim)

        # 将上下文向量与词嵌入拼接作为 LSTM 输入
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        # (batch, 1, embed_dim + hidden_dim)

        # LSTM 前向传播
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch, 1, hidden_dim)

        # 拼接 LSTM 输出和上下文用于预测
        prediction_input = torch.cat([output.squeeze(1), context], dim=1)
        # (batch, hidden_dim * 2)

        prediction = self.fc(prediction_input)  # (batch, vocab_size)

        return prediction, hidden, cell, attn_weights
```

### 7.3 完整的 Seq2Seq 模型（带注意力）

```python
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attn_weights_all = torch.zeros(batch_size, trg_len, src.size(1)).to(self.device)

        # 编码器
        encoder_outputs, (hidden, cell) = self.encoder(src)
        # encoder_outputs: (batch, src_len, hidden_dim)

        # 创建 mask：标记有效位置（非 <pad>）
        mask = (src != SRC.vocab.stoi['<pad>'])

        # 初始输入 <sos>
        input = trg[:, 0]  # (batch,)

        for t in range(1, trg_len):
            output, hidden, cell, attn_weights = self.decoder(
                input.unsqueeze(1),
                hidden,
                cell,
                encoder_outputs,
                mask,
            )
            # output: (batch, vocab_size)
            outputs[:, t, :] = output
            attn_weights_all[:, t, :] = attn_weights

            # Teacher Forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            if teacher_force:
                input = trg[:, t]
            else:
                input = top1

        return outputs, attn_weights_all
```

### 7.4 训练带注意力的模型

训练代码与基础 Seq2Seq 几乎相同，只需将模型替换为 `Seq2SeqWithAttention`：

```python
# 初始化带注意力的模型
encoder = Encoder(len(SRC.vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = AttnDecoder(len(TRG.vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2SeqWithAttention(encoder, decoder, device).to(device)

# 其余训练代码与之前相同
# ...
```

---

## 8. 基础 Seq2Seq vs 带注意力 Seq2Seq

| 对比方面 | 基础 Seq2Seq | 带注意力 Seq2Seq |
|----------|-------------|------------------|
| 上下文向量 | 固定长度的最终隐藏状态 | 动态计算的加权和 |
| 长序列处理 | 早期信息易丢失 | 所有位置都可访问 |
| 解码器输入 | 仅词嵌入 | 词嵌入 + 上下文向量 |
| 可解释性 | 黑盒 | 注意力权重可可视化 |
| 翻译质量 | 短句较好，长句下降 | 长句翻译质量更好 |

---

## 9. 注意力可视化

```python
import matplotlib.pyplot as plt

def visualize_attention(src_sentence, trg_sentence, model, attn_weights):
    # attn_weights: (trg_len, src_len)

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(attn_weights, cmap='Blues')

    # 设置刻度标签
    ax.set_xticks(range(len(src_sentence)))
    ax.set_yticks(range(len(trg_sentence)))
    ax.set_xticklabels(src_sentence, rotation=90)
    ax.set_yticklabels(trg_sentence)

    ax.set_xlabel('Source (English)')
    ax.set_ylabel('Target (French)')
    ax.set_title('Attention Weights')
    plt.tight_layout()
    plt.show()

# 翻译并可视化
sentence = "I love programming in Python"
tokens = [tok.text.lower() for tok in spacy_en.tokenizer(sentence)]

# 获取翻译结果和注意力权重
model.eval()
# ... 执行翻译，保存注意力权重 ...

# 可视化
visualize_attention(tokens, translation_tokens, model, attn_weights)
```

---

## 10. 常见错误

### 10.1 编码器输出未传递给解码器

基础 Seq2Seq 只传递隐藏状态，而注意力机制需要完整的编码器输出序列。

```python
# 正确：传递编码器所有时间步的输出
encoder_outputs, (hidden, cell) = self.encoder(src)
output, hidden, cell, attn_weights = self.decoder(
    input, hidden, cell, encoder_outputs, mask
)

# 错误：只传递隐藏状态（无法计算注意力）
output, hidden, cell = self.decoder(input, hidden, cell)
```

### 10.2 Mask 处理不当

在计算注意力时，必须屏蔽 `<pad>` 位置，避免模型把注意力放到填充字符上。

```python
# 正确：创建 mask 并传递给注意力模块
mask = (src != SRC.vocab.stoi['<pad>'])
context, attn_weights = self.attention(hidden, encoder_outputs, mask)

# 错误：没有屏蔽填充位置
context, attn_weights = self.attention(hidden, encoder_outputs)
```

### 10.3 注意力输入维度错误

注意力的 `hidden` 应该是解码器当前时间步的隐藏状态（最后一层），而不是所有层。

```python
# 正确：取最后一层
hidden_last = hidden[-1]  # (batch, hidden_dim)

# 错误：使用所有层
hidden_all = hidden  # (num_layers, batch, hidden_dim) - 维度不匹配
```

### 10.4 Teacher Forcing 比例设置不当

- 比例过高（如 1.0）：训练稳定，但模型不擅长自回归生成
- 比例过低（如 0.0）：训练不稳定，难以收敛
- 常用值：0.5 或从 1.0 逐渐降低到 0.0

### 10.5 损失函数忽略 `<pad>` 标记

```python
# 正确：忽略填充
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])

# 错误：没有忽略填充
criterion = nn.CrossEntropyLoss()
```

### 10.6 推理时使用正确的起始标记

```python
# 正确：以 <sos> 开始
trg_indices = [TRG.vocab.stoi['<sos>']]

# 错误：从空列表开始，第一个输入未知
trg_indices = []
```

---

## 11. 重点总结

**Seq2Seq 核心架构**：

```
编码器（读取源序列）→ 上下文向量 → 解码器（生成目标序列）
```

**基础 Seq2Seq**：

```
使用最终隐藏状态作为上下文向量
短句效果好，长句易丢失信息
```

**注意力机制**：

```
解码器动态关注编码器不同位置
上下文向量 = 加权和（权重由注意力评分决定）
长句翻译质量显著提升
```

**训练关键点**：

```
Teacher Forcing：训练时使用真实目标词作为输入
使用 BucketIterator 按长度分批，减少填充量
梯度裁剪防止梯度爆炸
```

**推理关键点**：

```
自回归生成：每步预测作为下一步输入
Teacher Forcing Ratio = 0
最大长度限制，防止无限循环
```

总结：

```
Seq2Seq 模型是机器翻译等序列到序列任务的基础框架。
从简单编码器-解码器到带注意力机制，模型逐步解决了固定上下文向量
的信息瓶颈问题。注意力机制不仅提升了翻译质量，还提供了模型
决策的可解释性，是 Seq2Seq 架构中最重要的改进之一。
```

# PyTorch Transformer 

## 1. Transformer 是什么？

Transformer 是一种主要依靠注意力机制处理序列数据的模型。它最初用于机器翻译，后来广泛应用于：

- 文本分类
- 机器翻译
- 文本生成
- 图像识别
- 语音处理
- 多模态任务

Transformer 的核心特点是：

```
不依赖 RNN 的逐步计算
可以同时处理整个序列
通过注意力机制建模不同位置之间的关系
```

原始 Transformer 由 Encoder 和 Decoder 组成：

```
源序列（例如英文）
        ↓
Encoder
        ↓
语义表示
        ↓
Decoder
        ↓
目标序列（例如中文）
```

PyTorch 的 `nn.Transformer` 实现了原始 Transformer 的基本结构，并支持 Encoder、Decoder、Mask 等功能。[PyTorch Transformer 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer)
![](../图片/Pasted%20image%2020260814014229.png)
---

## 2. Transformer 的整体结构

原始 Transformer 的结构：

```
输入 token
    ↓
Token Embedding
    ↓
Positional Encoding
    ↓
Encoder × N
    ↓
Decoder × N
    ↓
Linear
    ↓
Softmax
    ↓
输出 token
```

一个 Encoder Layer 包含：

```
Multi-Head Self-Attention
        ↓
残差连接 + LayerNorm
        ↓
Feed Forward Network
        ↓
残差连接 + LayerNorm
```

一个 Decoder Layer 包含：

```
Masked Multi-Head Self-Attention
        ↓
残差连接 + LayerNorm
        ↓
Cross-Attention
        ↓
残差连接 + LayerNorm
        ↓
Feed Forward Network
        ↓
残差连接 + LayerNorm
```

---

## 3. Token Embedding

模型不能直接处理单词，需要先把每个 token 转换成整数，再通过 Embedding 转换成向量：

```
"i"       → 5
"like"    → 8
"pytorch" → 20
```

然后：

```
embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=d_model,
)
```

如果输入形状为：

```
(batch_size, sequence_length)
```

经过 Embedding 后：

```
(batch_size, sequence_length, d_model)
```

其中：

- `vocab_size`：词表大小。
- `d_model`：每个 token 的向量维度。
- `sequence_length`：序列长度。

---

## 4. 为什么需要位置编码？

Transformer 的注意力机制本身没有顺序概念。

例如：

```
I love PyTorch
PyTorch love I
```

如果只看 token 集合，它们包含相同的单词，但顺序完全不同。

因此需要给每个位置加入位置信息：

```
token embedding + positional encoding
```

常见的正弦位置编码公式：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

简单实现：

```
import math  # 导入数学模块
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()  # 初始化父类

        position = torch.arange(
            max_len
        ).unsqueeze(1)  # 创建位置编号，形状为 (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )  # 创建不同维度对应的频率

        pe = torch.zeros(
            max_len,
            d_model,
        )  # 创建位置编码矩阵

        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # 偶数维使用 sin

        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # 奇数维使用 cos

        pe = pe.unsqueeze(0)  # 增加批次维度，变为 (1, max_len, d_model)

        self.register_buffer(
            "pe",
            pe,
        )  # 保存为 buffer，不作为模型参数训练

    def forward(self, x):
        sequence_length = x.size(1)  # 获取当前序列长度

        x = x + self.pe[:, :sequence_length, :]  # 将位置编码加到 token embedding 上

        return x  # 返回带有位置信息的表示
```

---

## 5. Self-Attention

Self-Attention 的作用是让序列中的每个位置关注其他位置。

例如：

```
The animal didn't cross the street because it was tired.
```

模型需要判断 `it` 指的是 `animal` 还是 `street`。注意力机制可以帮助模型建立这种远距离关系。

---

## 6. Query、Key 和 Value

对于输入序列中的每个 token，都会生成三个向量：

```
Query（Q）：我想寻找什么信息？
Key（K）：我具有什么信息？
Value（V）：如果被关注，实际传递什么信息？
```

计算过程：

```
Q = XWQ
K = XWK
V = XWV
```

注意力公式：

```
Attention(Q, K, V)
= softmax(QKᵀ / √dₖ)V
```
![](../图片/Pasted%20image%2020260814013552.png)
步骤如下：

```
Q 和 K 做点积
      ↓
除以 √dₖ，防止数值过大
      ↓
Softmax 转换为注意力权重
      ↓
与 V 加权求和
```

PyTorch 中可以直接使用：

```
attention = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    batch_first=True,
)
```

`MultiheadAttention` 的作用是让模型在多个表示子空间中同时关注不同信息。[PyTorch MultiheadAttention 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)

---

## 7. Multi-Head Attention

单个注意力头可能只能学习一种关系，因此 Transformer 使用多个注意力头。

```
输入
 ↓
分成多个 Head
 ↓
每个 Head 独立计算 Attention
 ↓
拼接所有 Head
 ↓
Linear
```

公式：

```
MultiHead(Q, K, V)
= Concat(head₁, head₂, ..., headₕ)WO
```

例如：

```
d_model = 512
num_heads = 8
```

每个注意力头的维度大约为：

```
512 / 8 = 64
```

要求：

```
d_model % num_heads == 0
```

否则无法平均分配每个 Head 的维度。

---

## 8. Feed Forward Network

注意力层之后会接一个前馈网络：

```
Linear(d_model → dim_feedforward)
        ↓
ReLU 或 GELU
        ↓
Linear(dim_feedforward → d_model)
```

公式：

```
FFN(x) = Linear₂(Activation(Linear₁(x)))
```

它对每个位置独立进行相同的非线性变换：

```
self.feed_forward = nn.Sequential(
    nn.Linear(d_model, dim_feedforward),  # 扩大特征维度
    nn.ReLU(),  # 使用非线性激活函数
    nn.Linear(dim_feedforward, d_model),  # 恢复到原始维度
)
```

注意力层负责：

```
不同 token 之间的信息交互
```

前馈网络负责：

```
对每个 token 的特征进行进一步变换
```

---

## 9. 残差连接和 LayerNorm

Transformer 中经常使用：

```
输出 = LayerNorm(x + 子层输出)
```

其中：

```
x：原始输入
子层输出：Attention 或 Feed Forward 的结果
```

代码形式：

```
x = x + attention_output  # 残差连接
x = self.norm1(x)  # LayerNorm
```

残差连接的作用：

- 保留原始信息
- 缓解深层网络训练困难
- 改善梯度传播

LayerNorm 的作用：

- 稳定特征分布
- 加快训练
- 减少训练不稳定

---

## 10. Encoder 和 Decoder 的区别

### Encoder

Encoder 的 Self-Attention 可以看到输入序列中的所有位置：

```
输入：I love PyTorch
每个 token 可以关注整句话
```

### Decoder

Decoder 通常需要使用 Mask，防止当前位置看到未来信息：

```
生成第 1 个词时，只能看到第 1 个位置
生成第 2 个词时，只能看到前 2 个位置
生成第 3 个词时，只能看到前 3 个位置
```

例如：

```
目标序列：<START> I love PyTorch
```

预测 `love` 时，不能提前看到 `PyTorch`。

---

## 11. Causal Mask

Causal Mask 通常是一个上三角 Mask：

```
允许：0
禁止：-inf
```

示例：

```
[0,    -inf, -inf, -inf]
[0,     0,   -inf, -inf]
[0,     0,    0,   -inf]
[0,     0,    0,    0  ]
```

PyTorch 可以直接生成：

```
def generate_square_subsequent_mask(size, device):
    mask = torch.triu(
        torch.ones(size, size, device=device)
        * float("-inf"),
        diagonal=1,
    )  # 创建上三角的未来信息遮挡矩阵

    return mask  # 返回 causal mask
```

也可以使用：

```
tgt_mask = nn.Transformer.generate_square_subsequent_mask(
    target_length
).to(device)  # 生成目标序列的因果 Mask
```

在 PyTorch 中，如果传入 Bool 类型的 Transformer Mask，`True` 表示对应位置不允许参与注意力，这一点需要特别注意。[PyTorch Transformer 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer)

---

## 12. 使用 `nn.Transformer`

PyTorch 提供了完整的 Transformer 模块：

```
transformer = nn.Transformer(
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    dropout=0.1,
    batch_first=True,
)
```

设置 `batch_first=True` 后：

```
输入形状：(batch_size, sequence_length, d_model)
```

如果不设置，则默认形状为：

```
(sequence_length, batch_size, d_model)
```

初学时推荐设置：

```
batch_first=True
```

这样更符合常见的 DataLoader 输出格式。

---

## 13. 一个简单的 Transformer Seq2Seq 模型

下面以一个简单的序列到序列任务为例：

```
class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        max_len=100,
    ):
        super(TransformerModel, self).__init__()  # 初始化父类

        self.d_model = d_model  # 保存 token 表示维度

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
        )  # 创建源语言 Embedding

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
        )  # 创建目标语言 Embedding

        self.src_positional_encoding = PositionalEncoding(
            d_model,
            max_len,
        )  # 创建源序列位置编码

        self.tgt_positional_encoding = PositionalEncoding(
            d_model,
            max_len,
        )  # 创建目标序列位置编码

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )  # 创建完整 Transformer

        self.output_layer = nn.Linear(
            d_model,
            tgt_vocab_size,
        )  # 将 Transformer 输出映射到目标词表

    def forward(
        self,
        src,
        tgt,
        tgt_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        src = self.src_embedding(src)  # 将源 token 转换为向量
        src = src * math.sqrt(self.d_model)  # 对 Embedding 进行缩放
        src = self.src_positional_encoding(src)  # 加入源序列位置编码

        tgt = self.tgt_embedding(tgt)  # 将目标 token 转换为向量
        tgt = tgt * math.sqrt(self.d_model)  # 对 Embedding 进行缩放
        tgt = self.tgt_positional_encoding(tgt)  # 加入目标序列位置编码

        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # 执行 Encoder 和 Decoder

        output = self.output_layer(output)  # 输出每个位置对应的词表分数

        return output  # 返回 logits
```

输出形状：

```
输入 src：
(batch_size, source_length)

输入 tgt：
(batch_size, target_length)

输出：
(batch_size, target_length, tgt_vocab_size)
```

---

## 14. 为什么要对 Embedding 进行缩放？

原始 Transformer 通常使用：

```
embedding = embedding * math.sqrt(d_model)
```

这是因为位置编码与 token embedding 相加时，需要让两者的数值尺度比较合适。

```
src = src * math.sqrt(self.d_model)
```

这不是所有现代 Transformer 都必须显式写出的步骤，但在学习原始 Transformer 结构时非常常见。

---

## 15. Transformer 训练过程

假设目标序列为：

```
<START> I love PyTorch <END>
```

训练时将目标序列错开一位：

```
Decoder 输入：
<START> I love PyTorch

预测目标：
I love PyTorch <END>
```

代码：

```
tgt_input = tgt[:, :-1]  # 去掉最后一个 token，作为 Decoder 输入
tgt_output = tgt[:, 1:]  # 去掉第一个 token，作为预测目标
```

生成 Mask：

```
target_length = tgt_input.size(1)  # 获取目标输入长度

tgt_mask = nn.Transformer.generate_square_subsequent_mask(
    target_length,
    device=tgt_input.device,
)  # 防止 Decoder 看到未来 token
```

模型训练：

```
logits = model(
    src=src,
    tgt=tgt_input,
    tgt_mask=tgt_mask,
)  # 得到每个目标位置的预测分数
```

计算损失：

```
criterion = nn.CrossEntropyLoss(
    ignore_index=pad_idx
)  # 忽略 <PAD> 位置的损失

loss = criterion(
    logits.reshape(-1, logits.size(-1)),
    tgt_output.reshape(-1),
)  # 展平所有时间步后计算交叉熵
```

完整训练步骤：

```
optimizer.zero_grad()  # 清除旧梯度
loss.backward()  # 反向传播
optimizer.step()  # 更新模型参数
```

---

## 16. Padding Mask

一个 batch 中的句子长度可能不同，需要使用 `<PAD>` 补齐：

```
[<START>, I, like, PyTorch, <END>]
[<START>, I, agree, <END>, <PAD>]
```

模型不应该关注 `<PAD>` 位置，因此需要创建 Padding Mask：

```
src_key_padding_mask = (
    src == pad_idx
)  # True 表示对应位置是 PAD，需要被忽略

tgt_key_padding_mask = (
    tgt_input == pad_idx
)  # 标记目标序列中的 PAD 位置
```

传入 Transformer：

```
logits = model(
    src=src,
    tgt=tgt_input,
    tgt_mask=tgt_mask,
    src_key_padding_mask=src_key_padding_mask,
    tgt_key_padding_mask=tgt_key_padding_mask,
)
```

需要区分两种 Mask：

```
Causal Mask：
防止 Decoder 看到未来 token

Padding Mask：
忽略句子补齐出来的 PAD token
```

---

## 17. Transformer 推理过程

训练时可以一次输入完整目标序列，但推理时没有真实目标句子，只能逐步生成。

```
输入 src
   ↓
Decoder 输入 <START>
   ↓
预测第一个 token
   ↓
把预测结果拼接到 Decoder 输入
   ↓
继续预测
   ↓
直到生成 <END>
```

简单贪心搜索代码：

```
def greedy_decode(
    model,
    src,
    start_idx,
    end_idx,
    max_length,
):
    model.eval()  # 切换到评估模式

    generated = torch.tensor(
        [[start_idx]],
        device=src.device,
    )  # 用 <START> 初始化目标序列

    with torch.inference_mode():
        for _ in range(max_length - 1):
            target_length = generated.size(1)  # 获取当前目标序列长度

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                target_length,
                device=src.device,
            )  # 创建当前长度对应的 causal mask

            logits = model(
                src=src,
                tgt=generated,
                tgt_mask=tgt_mask,
            )  # 预测目标序列每个位置的 logits

            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )  # 只取最后一个位置的预测结果

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )  # 将新 token 拼接到生成序列后面

            if next_token.item() == end_idx:
                break  # 生成 <END> 后停止

    return generated  # 返回生成的 token 序列
```

这里使用的是贪心搜索：

```
每一步都选择当前概率最高的 token
```

更复杂的生成方法还包括：

- Beam Search
- Top-k Sampling
- Top-p Sampling
- Temperature Sampling

---

## 18. Encoder-only、Decoder-only 和 Encoder-Decoder

Transformer 不只有一种结构。

### Encoder-only

例如 BERT：

```
输入文本 → Encoder → 分类或特征表示
```

常用于：

- 文本分类
- 命名实体识别
- 句子匹配

### Decoder-only

例如 GPT：

```
已有文本 → Decoder → 预测下一个 token
```

常用于：

- 文本生成
- 对话
- 代码生成

### Encoder-Decoder

例如原始 Transformer、T5：

```
源文本 → Encoder
目标文本 → Decoder
```

常用于：

- 机器翻译
- 文本摘要
- 图像描述
- 语音转文本

---

## 19. Transformer 与 RNN 的区别

### RNN

```
第 1 个 token → 第 2 个 token → 第 3 个 token
```

必须按顺序处理，难以并行。

### Transformer

```
整个序列同时输入
通过 Attention 建立不同位置之间的关系
```

优势：

- 更容易并行训练
- 更擅长处理长距离依赖
- 可以灵活关注序列中的任意位置

缺点：

- 注意力计算通常需要较多显存
- 序列越长，计算量越大
- 必须额外加入位置编码

---

## 20. 一个简化的 Transformer Encoder 分类模型

如果只是做文本分类，不需要完整的 Decoder，可以只使用 Encoder：

```
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        d_model=128,
        nhead=4,
        num_layers=2,
        max_len=256,
    ):
        super(TransformerClassifier, self).__init__()  # 初始化父类

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0,
        )  # 创建词嵌入层

        self.position = PositionalEncoding(
            d_model,
            max_len,
        )  # 创建位置编码

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )  # 创建一个 Encoder 层

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )  # 堆叠多个 Encoder 层

        self.fc = nn.Linear(
            d_model,
            num_classes,
        )  # 创建分类层

    def forward(self, token_ids):
        padding_mask = (
            token_ids == 0
        )  # 标记 PAD 位置

        x = self.embedding(token_ids)  # 将 token 转换为向量
        x = x * math.sqrt(self.embedding.embedding_dim)  # 缩放 Embedding
        x = self.position(x)  # 加入位置编码

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )  # 使用 Transformer Encoder 编码序列

        valid_tokens = ~padding_mask  # 找到非 PAD 位置
        lengths = valid_tokens.sum(dim=1).clamp(min=1)  # 获取每个句子的有效长度

        mask = valid_tokens.unsqueeze(-1)  # 扩展 Mask 维度
        x = x * mask  # 将 PAD 位置的特征置为 0

        x = x.sum(dim=1) / lengths.unsqueeze(1)  # 对有效 token 做平均池化

        logits = self.fc(x)  # 输出分类分数
        return logits  # 返回分类结果
```

---

## 21. 常见错误

### 忘记位置编码

错误：

```
x = embedding(token_ids)
output = transformer(x)
```

正确：

```
x = embedding(token_ids)
x = positional_encoding(x)
output = transformer(x)
```

### 忘记目标 Mask

Decoder 如果没有 Causal Mask，训练时可能看到未来答案，导致训练结果虚高，但推理时表现很差。

```
tgt_mask = nn.Transformer.generate_square_subsequent_mask(
    target_length
)
```

### `batch_first` 设置不一致

如果设置：

```
batch_first=True
```

输入必须是：

```
(batch_size, sequence_length, d_model)
```

如果没有设置，输入默认是：

```
(sequence_length, batch_size, d_model)
```

### `d_model` 不能被 `nhead` 整除

错误：

```
d_model=128
nhead=3
```

正确：

```
d_model=128
nhead=4
```

因为：

```
128 / 4 = 32
```

### 没有忽略 Padding

需要传入：

```
src_key_padding_mask=src == pad_idx
```

否则模型会把 `<PAD>` 当作正常 token 学习。

### 训练时直接使用预测结果

训练时一般使用 Teacher Forcing：

```
输入真实的前一个 token
```

推理时才使用模型上一步的预测结果。

---

## 22. 重点总结

Transformer 的核心组成：

```
Token Embedding
      ↓
Positional Encoding
      ↓
Multi-Head Attention
      ↓
残差连接 + LayerNorm
      ↓
Feed Forward Network
      ↓
残差连接 + LayerNorm
```

核心注意力公式：

```
Attention(Q, K, V)
= softmax(QKᵀ / √dₖ)V
```

PyTorch 中常用模块：

```
nn.Embedding()  # 将 token 编号转换为向量
nn.MultiheadAttention()  # 多头注意力
nn.TransformerEncoderLayer()  # Encoder 层
nn.TransformerEncoder()  # Encoder 堆叠
nn.Transformer()  # 完整 Encoder-Decoder Transformer
```

Transformer 的输入输出形状：

```
token_ids：
(batch_size, sequence_length)

Embedding 后：
(batch_size, sequence_length, d_model)

Transformer 输出：
(batch_size, sequence_length, d_model)

分类或词预测 logits：
(batch_size, sequence_length, vocab_size)
```

总结：

```
Transformer 通过 Self-Attention 让序列中的每个位置动态关注其他位置，
再结合位置编码、残差连接、LayerNorm 和前馈网络，
完成对序列信息的高效建模。
```




![](../图片/Pasted%20image%2020260814015237.png)
# PyTorch 从零实现 U-Net 图像分割

图像分割是将每个像素分类到特定类别的任务。U-Net 是 2015 年提出的经典架构，以 U 形结构和跳跃连接著称，能在少量数据下获得出色分割效果。[原始论文](https://arxiv.org/abs/1505.04597)

---

## 1. 核心思想：编码器-解码器 + 跳跃连接

U-Net 通过对称的 U 形结构同时解决两个问题：**语义理解**（“是什么”）和**空间定位**（“在哪里”）。


**跳跃连接的核心价值**：
- 编码器浅层特征包含边缘、纹理等**细节**信息，对精确定位边界至关重要
- 通过跳跃连接，解码器可直接访问这些细节，结合深层语义特征，实现精准分割
- 同时缓解梯度消失，加速收敛

---

## 2. 数据准备要点

- **图像**：RGB 三通道，通常归一化到 ImageNet 均值标准差。
- **掩码（Mask）** ：单通道灰度图，像素值为类别索引（0, 1, 2, ...）。**必须保持整数，不能归一化**。
- **数据增强**：对图像和掩码施加完全相同的变换（旋转、翻转等），保持对应关系。

---

## 3. 模型核心实现

### 3.1 双卷积块（DoubleConv）

两个 3×3 卷积堆叠，感受野相当于 5×5，但参数量更少。

```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
```

### 3.2 下采样块（Down）

通过最大池化降低空间维度，同时增加通道数。

```python
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),                # 尺寸减半
            DoubleConv(in_channels, out_channels)  # 通道加倍
        )

    def forward(self, x):
        return self.maxpool_conv(x)
```

### 3.3 上采样块（Up）

上采样后与编码器对应层拼接，然后双卷积。

```python
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # 上采样方式：双线性插值（无参数）或转置卷积
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: 下层特征（解码器）, x2: 编码器对应层特征
        x1 = self.up(x1)

        # 处理尺寸不一致（可能因像素偏移差1）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)  # 拼接（跳跃连接）
        return self.conv(x)
```

### 3.4 完整 U-Net

```python
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)          # 第1层
        self.down1 = Down(64, 128)                     # 第2层
        self.down2 = Down(128, 256)                    # 第3层
        self.down3 = Down(256, 512)                    # 第4层
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)         # 瓶颈层

        self.up1 = Up(1024, 512 // factor, bilinear)   # 上采样1
        self.up2 = Up(512, 256 // factor, bilinear)    # 上采样2
        self.up3 = Up(256, 128 // factor, bilinear)    # 上采样3
        self.up4 = Up(128, 64, bilinear)               # 上采样4
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)  # 1×1卷积输出

    def forward(self, x):
        x1 = self.inc(x)      # 64通道
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024

        x = self.up1(x5, x4)  # 结合第4层
        x = self.up2(x, x3)   # 结合第3层
        x = self.up3(x, x2)   # 结合第2层
        x = self.up4(x, x1)   # 结合第1层
        logits = self.outc(x)
        return logits
```

---

## 4. 损失函数与评估

### 4.1 Dice Loss

Dice 系数衡量两个集合的相似度，天然处理类别不平衡：

```python
def dice_loss(pred, target, smooth=1e-6):
    # pred: (batch, n_classes, H, W) 概率（softmax后）
    # target: (batch, H, W) 整数标签
    pred = torch.softmax(pred, dim=1)
    n_classes = pred.size(1)
    target_one_hot = F.one_hot(target, n_classes).permute(0,3,1,2).float()
    intersection = (pred * target_one_hot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
    dice = (2.*intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
```

### 4.2 常用组合：BCE + Dice（二分类）

```python
class CombinedLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        # pred: (batch, 1, H, W), target: (batch, H, W) 0/1
        bce = self.bce(pred.squeeze(1), target.float())
        pred_sigmoid = torch.sigmoid(pred.squeeze(1))
        dice = 1 - (2.*(pred_sigmoid * target.float()).sum() + 1e-6) / (pred_sigmoid.sum() + target.float().sum() + 1e-6)
        return self.weight_bce * bce + self.weight_dice * dice
```

### 4.3 评估指标：IoU

```python
def iou_score(pred, target, n_classes):
    pred = torch.softmax(pred, dim=1).argmax(dim=1)  # (batch, H, W)
    ious = []
    for cls in range(n_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        ious.append((intersection + 1e-6) / (union + 1e-6))
    return torch.tensor(ious).mean()
```

---

## 5. 训练循环要点

```python
model = UNet(n_channels=3, n_classes=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = CombinedLoss()

for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:   # masks: (batch, H, W) 整数
        images, masks = images.to(device), masks.to(device)
        preds = model(images)            # (batch, 1, H, W)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # 验证逻辑类似，但 model.eval() + torch.no_grad()
```

**关键点**：
- 掩码类型为 `torch.long`，值在 [0, num_classes-1]
- 梯度裁剪（`clip_grad_norm_`）可防止梯度爆炸
- 使用 `BucketIterator` 按图像大小分批可减少填充量

---

## 6. 常见陷阱

| 陷阱 | 正确做法 |
|------|---------|
| 掩码被归一化 | 保持整数，不用 ToTensor 或归一化 |
| 输出通道与损失不匹配 | 二分类：1通道+sigmoid+BCE 或 2通道+softmax+CrossEntropy |
| 跳跃连接尺寸差1 | 在 Up 块中裁剪填充对齐 |
| 数据增强不一致 | 对图像和掩码施加相同变换（用固定种子） |
| 预测时忘记归一化 | 测试时使用训练时相同的均值和标准差 |

---

## 7. 重点总结

**U-Net 核心设计**：

```
1. 编码器：下采样提取语义（通道增加，尺寸减小）
2. 解码器：上采样恢复分辨率（通道减少，尺寸增加）
3. 跳跃连接：拼接编码器特征，融合细节与语义
```

**与分类网络的区别**：

| 方面 | 分类 | 分割 |
|------|------|------|
| 输出 | 一个向量 | 全图每个像素 |
| 空间尺寸 | 不断减小 | 最终复原 |
| 关键操作 | 全连接/池化 | 上采样/跳跃连接 |
| 损失 | CrossEntropy | Dice / 组合损失 |

**本质理解**：

```
U-Net 通过跳跃连接让解码器同时看到“全局语义”和“局部细节”，
这是它能在小数据集上成功的关键。
```

# PyTorch 目标检测与 YOLO 

## 1. 什么是目标检测？

目标检测不仅要判断图片中有什么，还要找到每个目标的位置。

```
图像分类：
这张图片中有一只猫

目标检测：
这张图片中有一只猫
猫的位置是 [x1, y1, x2, y2]
```

目标检测通常需要同时完成：

```
分类：目标属于什么类别？
定位：目标在图片中的什么位置？
数量：图片中有多少个目标？
```

一个检测结果通常包含：

```
Bounding Box：目标框
Class：目标类别
Confidence：模型对检测结果的置信度
```

例如：

```
类别：dog
边界框：[100, 50, 300, 280]
置信度：0.92
```

---

## 2. 边界框的表示方式

最常见的边界框格式有两种。

### `xyxy` 格式

```
[x_min, y_min, x_max, y_max]
```

例如：

```
box = [100, 50, 300, 280]
```

表示：

```
左上角：(100, 50)
右下角：(300, 280)
```

### `xywh` 格式

```
[x_center, y_center, width, height]
```

例如：

```
box = [200, 165, 200, 230]
```

表示：

```
中心点：(200, 165)
宽度：200
高度：230
```

YOLO 数据集标注通常使用归一化后的 `xywh` 格式：

```
class_id x_center y_center width height
```

所有坐标都除以图片宽度或高度，因此范围通常是：

```
x_center, y_center, width, height ∈ [0, 1]
```

---

## 3. IoU：交并比

IoU（Intersection over Union）用于衡量两个边界框的重叠程度：

```
IoU = 交集面积 / 并集面积
```

示意：

```
两个框重叠越多 → IoU 越接近 1
两个框完全不重叠 → IoU = 0
```

IoU 常用于：

- 判断预测框是否正确
- 计算检测损失
- 执行 NMS
- 计算 mAP

一个简单的 PyTorch 实现：

```
def intersection_over_union(boxes_preds, boxes_labels):
    """
    boxes_preds 和 boxes_labels 使用 xyxy 格式：
    [x_min, y_min, x_max, y_max]
    """

    x1 = torch.max(
        boxes_preds[..., 0],
        boxes_labels[..., 0],
    )  # 计算交集左上角的 x 坐标

    y1 = torch.max(
        boxes_preds[..., 1],
        boxes_labels[..., 1],
    )  # 计算交集左上角的 y 坐标

    x2 = torch.min(
        boxes_preds[..., 2],
        boxes_labels[..., 2],
    )  # 计算交集右下角的 x 坐标

    y2 = torch.min(
        boxes_preds[..., 3],
        boxes_labels[..., 3],
    )  # 计算交集右下角的 y 坐标

    intersection = (
        (x2 - x1).clamp(min=0)
        * (y2 - y1).clamp(min=0)
    )  # 计算交集面积

    pred_area = (
        (boxes_preds[..., 2] - boxes_preds[..., 0]).clamp(min=0)
        * (boxes_preds[..., 3] - boxes_preds[..., 1]).clamp(min=0)
    )  # 计算预测框面积

    label_area = (
        (boxes_labels[..., 2] - boxes_labels[..., 0]).clamp(min=0)
        * (boxes_labels[..., 3] - boxes_labels[..., 1]).clamp(min=0)
    )  # 计算真实框面积

    union = pred_area + label_area - intersection  # 计算并集面积

    return intersection / (union + 1e-6)  # 返回 IoU，避免除以 0
```

---

## 4. YOLO 的核心思想

YOLO 的全称是：

```
You Only Look Once
```

YOLO 将目标检测看作一个整体的回归问题：一次性读取整张图片，同时预测目标的位置和类别。[YOLO 原始论文](https://arxiv.org/abs/1506.02640)

基本流程：

```
整张图片
    ↓
CNN Backbone 提取特征
    ↓
Detection Head 输出预测结果
    ↓
解码边界框和类别
    ↓
置信度筛选
    ↓
NMS 去除重复框
```

与需要先生成候选区域的两阶段检测器相比，YOLO 属于单阶段检测器：

```
两阶段方法：
候选区域 → 分类和回归

YOLO：
整张图片 → 直接预测边界框和类别
```

---

## 5. YOLO 的网格思想

以 YOLOv1 为例，模型将图片划分为 `S×S` 的网格：

```
S = 7

整张图片 → 7×7 网格
```

如果一个目标的中心落在某个网格中，那么这个网格负责预测该目标。

每个网格可以预测：

```
B 个边界框
每个框的 5 个值：
    x, y, w, h, confidence
C 个类别概率
```

因此输出大小为：

```
S × S × (B × 5 + C)
```

例如原始 YOLOv1：

```
S = 7
B = 2
C = 20
```

输出为：

```
7 × 7 × (2 × 5 + 20)
= 7 × 7 × 30
```

---

## 6. 一个预测框包含什么？

一个 YOLO 预测框通常包含：

```
x：框中心点相对于网格的位置
y：框中心点相对于网格的位置
w：框的宽度
h：框的高度
objectness：该框中是否存在目标
class scores：目标属于每个类别的分数
```

检测结果的最终置信度通常可以理解为：

```
最终类别置信度
= objectness × class probability
```

例如：

```
objectness = 0.9
cat probability = 0.8

最终 cat 置信度 = 0.9 × 0.8 = 0.72
```

---

## 7. 现代 YOLO 与 YOLOv1 的区别

YOLOv1 的网格思想非常适合学习目标检测，但现代 YOLO 版本通常还会加入：

```
多尺度检测
更复杂的 Backbone
Feature Pyramid
Anchor 或 Anchor-free 预测方式
更合理的边界框损失
更先进的数据增强
更好的标签分配方法
```

因此：

```
YOLOv1 的 S×S×(B×5+C)
```

主要是帮助理解 YOLO 的基本思想。

现代 YOLO 的输出通常是多个尺度的特征图，例如：

```
小目标检测层：80×80
中目标检测层：40×40
大目标检测层：20×20
```

这样可以同时检测不同大小的目标。

---

## 8. YOLO 模型的组成

一个 YOLO 检测器通常由三部分组成。

### Backbone

用于提取图像特征：

```
输入图片 → 多层卷积 → 特征图
```

常见 Backbone：

```
Darknet
CSPDarknet
ResNet
EfficientNet
```

### Neck

用于融合不同尺度的特征：

```
浅层特征：分辨率高，适合小目标
深层特征：语义信息强，适合大目标
```

常见结构：

```
FPN
PAN
```

### Detection Head

负责预测：

```
边界框坐标
目标置信度
类别分数
```

---

## 9. PyTorch 中的简单 YOLO Detection Head

假设 Backbone 输出：

```
(batch_size, 256, 20, 20)
```

每个网格预测：

```
3 个框
4 个坐标
1 个 objectness
num_classes 个类别分数
```

那么每个框需要：

```
5 + num_classes
```

个输出。

Detection Head 可以写成：

```
import torch
import torch.nn as nn


class YOLODetectionHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        num_anchors=3,
    ):
        super(YOLODetectionHead, self).__init__()  # 初始化父类

        self.num_classes = num_classes  # 保存类别数量
        self.num_anchors = num_anchors  # 保存每个网格的预测框数量

        self.prediction = nn.Conv2d(
            in_channels,
            num_anchors * (5 + num_classes),
            kernel_size=1,
        )  # 使用 1×1 卷积输出每个网格的检测结果

    def forward(self, x):
        output = self.prediction(x)  # 输出原始预测结果

        batch_size, _, height, width = output.shape  # 获取输出形状

        output = output.view(
            batch_size,
            self.num_anchors,
            5 + self.num_classes,
            height,
            width,
        )  # 将通道维度拆分为 anchors 和预测值

        output = output.permute(
            0,
            1,
            3,
            4,
            2,
        )  # 变为 (batch, anchors, grid_h, grid_w, prediction_dim)

        return output  # 返回 YOLO 预测结果
```

输出形状：

```
(batch_size, num_anchors, grid_height, grid_width, 5 + num_classes)
```

例如：

```
(batch, 3, 20, 20, 5 + 10)
```

---

## 10. YOLO 的训练目标

对于每个网格和每个预测框，模型需要学习：

```
边界框坐标损失
objectness 损失
类别分类损失
```

总损失可以写成：

```
Loss =
λ_box × Box Loss
+ λ_obj × Objectness Loss
+ λ_noobj × No-object Loss
+ λ_cls × Class Loss
```

### Box Loss

衡量预测框和真实框的位置差异：

```
预测框离真实框越近，损失越小
```

现代检测器常使用：

```
IoU Loss
GIoU Loss
DIoU Loss
CIoU Loss
```

### Objectness Loss

判断当前预测框是否包含目标：

```
有目标 → objectness 应该接近 1
没有目标 → objectness 应该接近 0
```

### Classification Loss

当网格中存在目标时，预测目标属于哪个类别：

```
cat、dog、car 等类别
```

背景网格通常不计算分类损失，否则背景数量太多，会影响训练。

---

## 11. YOLO 标签如何分配？

假设目标中心坐标是：

```
x_center = 0.62
y_center = 0.35
```

对于 `S=7` 的网格：

```
grid_x = int(x_center * S)  # 得到目标中心所在的列
grid_y = int(y_center * S)  # 得到目标中心所在的行
```

得到：

```
grid_x = 4
grid_y = 2
```

表示：

```
第 2 行、第 4 列的网格负责预测这个目标
```

网格内部的相对坐标：

```
x_cell = x_center * S - grid_x  # 计算目标中心在网格中的相对 x 坐标
y_cell = y_center * S - grid_y  # 计算目标中心在网格中的相对 y 坐标
```

这些坐标通常位于：

```
[0, 1]
```

---

## 12. 一个简单的数据集格式

目标检测中，一张图片可能对应多个目标，因此 Dataset 通常返回：

```
image, target
```

其中 `target` 可以包含：

```
target = {
    "boxes": boxes,  # 形状为 (num_objects, 4)
    "labels": labels,  # 形状为 (num_objects,)
}
```

示例：

```
target = {
    "boxes": torch.tensor([
        [50, 40, 180, 220],
        [250, 100, 420, 300],
    ], dtype=torch.float32),  # 两个目标框

    "labels": torch.tensor([
        0,
        1,
    ], dtype=torch.int64),  # 两个目标的类别
}
```

由于每张图片中的目标数量可能不同，默认 `DataLoader` 不能直接把所有 target 堆叠起来。

可以使用：

```
def detection_collate_fn(batch):
    images = []  # 保存图片
    targets = []  # 保存目标信息

    for image, target in batch:
        images.append(image)  # 添加图片
        targets.append(target)  # 添加对应 target

    images = torch.stack(images)  # 图片通常可以堆叠为统一批次

    return images, targets  # targets 保持为 list
```

---

## 13. PyTorch 目标检测数据格式

以 Torchvision 的检测模型为例，训练时通常传入：

```
images = [
    image_1,
    image_2,
]

targets = [
    {
        "boxes": boxes_1,
        "labels": labels_1,
    },
    {
        "boxes": boxes_2,
        "labels": labels_2,
    },
]
```

示例：

```
for images, targets in train_loader:
    images = [
        image.to(device)
        for image in images
    ]  # 将每张图片移动到设备

    targets = [
        {
            key: value.to(device)
            for key, value in target.items()
        }
        for target in targets
    ]  # 将 boxes 和 labels 移动到设备

    loss_dict = model(
        images,
        targets,
    )  # 训练模式下返回各项损失
```

Torchvision 提供了 Faster R-CNN、RetinaNet、SSD、FCOS 等目标检测模型，但并不等同于 YOLO。[Torchvision 检测模型文档](https://docs.pytorch.org/vision/stable/models.html)

---

## 14. NMS：非极大值抑制

模型可能对同一个目标预测出很多重叠框：

```
框 A：0.92
框 B：0.86
框 C：0.74
```

NMS 的步骤：

```
1. 按置信度从高到低排序
2. 保留置信度最高的框
3. 删除与它 IoU 过高的框
4. 对剩余框重复以上过程
```

PyTorch 可以直接使用：

```
from torchvision.ops import nms  # 导入 NMS


boxes = torch.tensor([
    [50, 50, 200, 200],
    [55, 55, 205, 205],
    [300, 300, 450, 450],
], dtype=torch.float32)  # 创建预测框

scores = torch.tensor([
    0.95,
    0.80,
    0.88,
])  # 创建每个框的置信度

keep_indices = nms(
    boxes,
    scores,
    iou_threshold=0.5,
)  # 删除重叠程度过高的低分框

final_boxes = boxes[keep_indices]  # 获取 NMS 后保留的框
final_scores = scores[keep_indices]  # 获取保留框对应的分数
```

通常要对每个类别分别进行 NMS：

```
cat 的框单独 NMS
dog 的框单独 NMS
car 的框单独 NMS
```

否则不同类别的目标可能互相错误地抑制。

---

## 15. YOLO 推理流程

完整推理过程：

```
输入图片
    ↓
调整大小和标准化
    ↓
CNN Backbone 提取特征
    ↓
YOLO Head 输出预测
    ↓
解码 x、y、w、h
    ↓
计算 objectness × class score
    ↓
过滤低置信度框
    ↓
按类别执行 NMS
    ↓
返回最终检测结果
```

伪代码：

```
model.eval()  # 切换到评估模式

with torch.inference_mode():
    raw_predictions = model(images)  # 获取 YOLO 原始输出

boxes, objectness, class_scores = decode_predictions(
    raw_predictions
)  # 将模型输出解码为边界框和类别分数

confidence = objectness * class_scores  # 计算最终类别置信度

keep = confidence > confidence_threshold  # 过滤低置信度预测框

final_boxes = boxes[keep]  # 保留高置信度框
final_scores = confidence[keep]  # 保留对应置信度

keep_indices = nms(
    final_boxes,
    final_scores,
    iou_threshold=0.5,
)  # 执行 NMS

final_boxes = final_boxes[keep_indices]  # 获取最终边界框
final_scores = final_scores[keep_indices]  # 获取最终分数
```

---

## 16. 目标检测中的数据增强

目标检测的数据增强必须同时修改：

```
图片
边界框
```

例如图片水平翻转时：

```
图片被翻转
目标框的 x 坐标也必须翻转
```

不能只修改图片而不修改边界框，否则图片和标签会错位。

常见增强：

```
随机水平翻转
随机缩放
随机裁剪
颜色变化
Mosaic
MixUp
```

使用 Albumentations 时，它可以同步变换图片和边界框：

```
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
    ),
)  # 指定边界框格式和类别标签字段
```

---

## 17. 目标检测的评价指标

### IoU

用于衡量预测框与真实框的重叠程度。

### Precision

在所有预测为目标的框中，有多少是真正正确的：

```
Precision = TP / (TP + FP)
```

### Recall

在所有真实目标中，有多少被检测出来：

```
Recall = TP / (TP + FN)
```

### AP

对单个类别，在不同置信度阈值下计算 Precision-Recall 曲线，并计算曲线下面积。

### mAP

对所有类别的 AP 取平均：

```
mAP = 所有类别 AP 的平均值
```

常见指标：

```
mAP@0.5
mAP@0.5:0.95
```

其中：

```
mAP@0.5：
IoU ≥ 0.5 时认为预测正确

mAP@0.5:0.95：
在多个 IoU 阈值下取平均，评价更严格
```

---

## 18. 目标检测和图像分类的区别

### 图像分类

```
输入：一张图片
输出：一个类别
```

输出形状：

```
(batch_size, num_classes)
```

### 目标检测

```
输入：一张图片
输出：多个框、多个类别和多个置信度
```

输出可以理解为：

```
每张图片：
[
    [x1, y1, x2, y2, score, class_id],
    [x1, y1, x2, y2, score, class_id],
    ...
]
```

因此目标检测的数据集比分类数据集更复杂：

```
一张图片 → 一个标签
```

变成：

```
一张图片 → 多个边界框 + 多个类别
```

---

## 19. 常见错误

### 边界框格式混用

需要明确使用的是：

```
xyxy
xywh
归一化坐标
像素坐标
```

如果格式混用，模型可能完全无法训练。

### 图像增强后没有同步修改边界框

错误：

```
只翻转图片
```

正确：

```
同时翻转图片和边界框
```

### 忘记处理多个目标

一张图片可能包含多个目标，不能把标签写成一个整数：

```
label = 1  # 只适合图像分类
```

检测任务需要：

```
boxes = [...]
labels = [...]
```

### 置信度筛选过早

如果置信度阈值设置过高，可能过滤掉很多正确但分数较低的框。

通常需要结合：

```
confidence threshold
NMS IoU threshold
```

一起调整。

### 没有执行 NMS

同一个目标可能保留很多重复框，最终检测结果会很混乱。

### 只看准确率

目标检测不能只看分类准确率，需要关注：

```
IoU
Precision
Recall
AP
mAP
```

---

## 20. 目标检测完整流程总结

```
准备图片和边界框标签
          ↓
Dataset 读取 image、boxes、labels
          ↓
同步执行数据增强
          ↓
DataLoader 组成 batch
          ↓
YOLO Backbone 提取特征
          ↓
YOLO Head 预测框、objectness 和类别
          ↓
计算 Box、Objectness、Classification Loss
          ↓
反向传播更新模型
          ↓
推理时解码预测结果
          ↓
置信度筛选
          ↓
NMS 去除重复框
          ↓
计算 IoU、AP 和 mAP
```

## 21. 重点总结

YOLO 的核心思想：

```
整张图片只经过一次网络，
直接同时预测多个目标的边界框和类别。
```

一个预测结果通常包括：

```
边界框坐标
objectness
类别分数
```

目标检测最需要理解的几个概念：

```
Bounding Box：目标的位置
IoU：预测框和真实框的重叠程度
Objectness：框中是否存在目标
NMS：删除重复预测框
mAP：综合评价检测效果
```

总结：

```
YOLO 将目标检测转化为一个端到端的回归问题：
模型直接从整张图片预测目标的位置、类别和置信度，
再通过置信度筛选和 NMS 得到最终检测结果。
```