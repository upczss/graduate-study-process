虚拟环境1：
名称：pytorch_env
pytorch版本：2.13.0stable+CUDA12.6
python版本：3.11
路径：C:\Users\zss\anaconda3\envs\pytorch_env

虚拟环境2：
名称：mllearning
pytorch版本：2.7.1stable+CUDA12.8
python版本：3.11
路径：C:\Users\zss\anaconda3\envs\pytorch_env
# 两大函数
```python
dir():打开列表
help():说明
```
# 加载数据
原始数据 → Dataset → DataLoader → 一个个 batch → 模型训练
```python
Dataset负责获取数据集里面的数据及其lable，一共有多少个样本
Dataloader为后面的网络提供不同的数据形式
```
## 如何使用Dataset

```python
# 从 PyTorch 中导入 Dataset 类
# 我们自己定义的数据集需要继承 Dataset
from torch.utils.data import Dataset
from PIL import Image
# os 用来处理文件夹路径和文件名
import os
# 定义自己的图片数据集
# MyData 继承了 PyTorch 提供的 Dataset
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        """
        初始化数据集。
        创建 MyData 对象时，这个方法会自动执行。
        参数：
        root_dir：
            训练集的根目录，例如：
            .../hymenoptera_data/train
        label_dir：
            类别文件夹名称，例如：
            ants 或 bees
        """
        # 保存训练集根目录
        self.root_dir = root_dir
        # 保存类别文件夹名称
        self.label_dir = label_dir
        # 把训练集根目录和类别目录拼接起来
        self.path = os.path.join(
            self.root_dir,
            self.label_dir
        )

        # 获取类别文件夹中的所有文件名
        # ants 文件夹中有：
        # 0013035.jpg
        # 1030023514_aad5c608f9.jpg
        # 1095476100_3906d8afde.jpg
        # 那么 self.img_names 大致是：
        # [
        #     "0013035.jpg",
        #     "1030023514_aad5c608f9.jpg",
        #     "1095476100_3906d8afde.jpg"
        # ]
        #
        # 注意：这里得到的只是文件名，并没有真正读取图片
        self.img_names = os.listdir(self.path)

    def __getitem__(self, index):
        """
        根据索引读取一个样本。
        当我们执行：
            dataset[0]
        Python 就会自动调用：
            dataset.__getitem__(0)
        参数：
        index：
            图片的索引，也就是要读取第几张图片
        返回：
        img：
            读取到的图片
        label：
            图片对应的类别，例如 ants 或 bees
        """

        # 根据 index 从图片名称列表中获取一个文件名
        img_name = self.img_names[index]
        # 把图片所在的文件夹路径和图片名称拼接起来
        img_path = os.path.join(
            self.path,
            img_name
        )

        # 使用 PIL 打开图片
        # convert("RGB") 的作用是统一图片的颜色格式：
        # 在后续处理中造成通道数量不一致的问题
        img = Image.open(img_path).convert("RGB")
        # 使用图片所在文件夹的名字作为标签
        # 如果当前数据集读取的是 ants 文件夹：
        # label = "ants"
        # 如果当前数据集读取的是 bees 文件夹：
        # label = "bees"
        label = self.label_dir
        # 返回一张图片和它对应的标签
        #
        # 返回结果的形式是：
        # (图片, 标签)
        return img, label

    def __len__(self):
        """
        返回数据集中的样本数量。

        当我们执行：

            len(dataset)

        Python 就会自动调用：

            dataset.__len__()
        """

        # self.img_names 中有多少个文件名，
        # 就认为这个数据集有多少张图片
        return len(self.img_names)


# 只有直接运行 dataset.py 时，下面的代码才会执行
# 如果以后在其他 Python 文件中导入 MyData：
# from dataset import MyData
# 那么下面的测试代码不会自动执行
if __name__ == "__main__":

    # __file__ 表示当前 Python 文件 dataset.py 的路径
    # os.path.abspath(__file__)
    # 得到 dataset.py 的绝对路径
    # os.path.dirname(...)
    # 得到 dataset.py 所在的文件夹
    # dataset.py 位于：
    # obsidian/code/pytorch learning/dataset.py
    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )
    # 根据 dataset.py 所在位置找到训练集
    # current_dir：
    # obsidian/code/pytorch learning
    # ".."：
    # 返回上一级，即 obsidian/code
    # 后面再进入：
    # data/hymenoptera_data/train
    root_dir = os.path.join(
        current_dir,
        "..",
        "data",
        "hymenoptera_data",
        "train"
    )

    root_dir = os.path.abspath(root_dir)

    # 在终端中打印实际读取的数据集路径
    # 如果出现路径错误，可以根据这里的输出进行检查
    print("训练集路径：", root_dir)

    # 判断训练集目录是否真的存在
    #
    # 如果目录不存在，就主动抛出一个容易理解的错误
    if not os.path.exists(root_dir):
        raise FileNotFoundError(
            f"没有找到训练集目录：{root_dir}"
        )

    ants_dataset = MyData(
        root_dir=root_dir,
        label_dir="ants"
    )


    bees_dataset = MyData(
        root_dir=root_dir,
        label_dir="bees"
    )

    # 注意：
    # 这里不是把图片内容相加，
    # 而是把两个数据集首尾连接起来
    train_dataset = ants_dataset + bees_dataset

    print("蚂蚁图片数量：", len(ants_dataset))

    print("蜜蜂图片数量：", len(bees_dataset))

    print("训练集图片总数：", len(train_dataset))

    img, label = ants_dataset[0]

    print("\n第一张蚂蚁图片的信息：")
    print("图片标签：", label)
    print("图片尺寸：", img.size)
    print("图片模式：", img.mode)
    print("图片对象类型：", type(img))

    # 显示读取到的图片
    img.show()


    bee_img, bee_label = bees_dataset[0]

    print("\n第一张蜜蜂图片的信息：")
    print("图片标签：", bee_label)
    print("图片尺寸：", bee_img.size)
    print("图片模式：", bee_img.mode)

    first_img, first_label = train_dataset[0]

    print("\n合并数据集中的第一个样本：")
    print("标签：", first_label)

    # 第一张蜜蜂图片在合并数据集中的索引，
    # 等于蚂蚁数据集的长度
    first_bee_index = len(ants_dataset)
    first_bee_img, first_bee_label = train_dataset[
        first_bee_index
    ]
    print("\n合并数据集中的第一张蜜蜂图片：")
    print("索引：", first_bee_index)
    print("标签：", first_bee_label)
```
# Tensorboard的使用

Python训练代码
      ↓
SummaryWriter记录数据
      ↓
生成logs日志文件
      ↓
TensorBoard读取日志
      ↓
浏览器显示图表


```python
from torch.utils.tensorboard import SummaryWriter
# 创建日志记录器，日志保存到 logs 文件夹
writer = SummaryWriter("logs")
# 生成并记录 y = 2x 的数据
for i in range(100):
    writer.add_scalar(
        "y=2x",  # 图表名称
        2 * i,   # 纵坐标 y
        i        # 横坐标 x
    )
# 关闭日志记录器
writer.close()
```
## 常用 TensorBoard 函数

### 1. `add_scalar()`

记录一个标量：

```
writer.add_scalar(
    "Loss/train",
    loss_value,
    step
)
```

适合记录：

- 损失
- 正确率
- 学习率

### 2. `add_scalars()`

在同一个图表中记录多个标量：

```
writer.add_scalars(
    "Loss",
    {
        "train": train_loss,
        "validation": val_loss
    },
    epoch
)
```

### 3. `add_image()`

记录一张图片：

```
writer.add_image(
    "image",
    image_tensor,
    step
)
```

### 4. `add_images()`

一次记录多张图片：

```
writer.add_images(
    "batch_images",
    image_batch,
    step
)
```

### 5. `add_graph()`

显示模型计算图：

```
writer.add_graph(
    model,
    input_tensor
)
```

### 6. `add_histogram()`

记录参数或梯度的分布：

```
writer.add_histogram(
    "weights",
    model.weight,
    epoch
)
```

### 7. `flush()`

立即把缓存中的数据写入日志：

```
writer.flush()
```

程序长时间训练时可以定期调用。

### 8. `close()`

保存剩余数据并关闭记录器：

```
writer.close()
```

# Transforms
# PyTorch 学习笔记：Transforms

## 1. Transforms 是什么

`transforms` 是 `torchvision` 中用于处理图像的工具。

原始图片通常不能直接送入神经网络，需要先经过一些预处理，例如：

- 调整图片尺寸
- 转换成 Tensor
- 裁剪图片
- 翻转图片
- 修改亮度和颜色
- 标准化像素值

基本流程：

```
原始图片
   ↓
Transforms 图像处理
   ↓
符合要求的 Tensor
   ↓
送入神经网络
```

导入方式：

```
from torchvision import transforms
```

## 2. `Compose`：组合多个操作

`transforms.Compose()` 可以把多个图像处理操作组合起来，并按照从上到下的顺序执行。

```
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
```

使用：

```
image = transform(image)
```

执行顺序：

```
原始图片
   ↓
Resize((224, 224))
   ↓
ToTensor()
   ↓
处理后的 Tensor
```

顺序很重要，因为不同 transform 对输入类型有不同要求。

## 3. `ToTensor`：转换成 Tensor

```
transform = transforms.ToTensor()
tensor_image = transform(image)
```

`ToTensor()` 通常完成两件事。

### 改变维度顺序

PIL 图片或 NumPy 图片通常采用：

```
H × W × C
```

PyTorch 图片 Tensor 通常采用：

```
C × H × W
```

其中：

- `H`：高度
- `W`：宽度
- `C`：颜色通道

例如：

```
转换前：(400, 500, 3)
转换后：(3, 400, 500)
```

### 缩放像素值

普通图片的像素值通常为：

```
0～255
```

转换成 Tensor 后通常变为：

```
0～1
```

例如：

```
0   → 0.0
128 → 大约0.502
255 → 1.0
```
Tensor（张量）是 PyTorch 中存储数据和进行计算的基本数据类型，可以理解为支持任意维度的数字容器。
### Tensor 的维度
0 维：一个数，也叫标量
1 维：一组数，也叫向量
2 维：表格或矩阵
3 维：可以表示一张彩色图片
4 维：可以表示一批图片
更高维：可以表示视频、序列等复杂数据
### 图片 Tensor
PyTorch 中，一张彩色图片通常使用：
C × H × W
其中：
C：颜色通道，RGB 图片通常是 3
H：图片高度
W：图片宽度
一批图片通常使用：
N × C × H × W
其中 N 是一个批次中的图片数量。
### Tensor 的重要属性
shape：Tensor 的形状
ndim：Tensor 有多少个维度
dtype：元素的数据类型
device：Tensor 位于 CPU 还是 GPU
numel：Tensor 包含的元素总数
### 常见数据类型
float32：模型输入和模型参数常用
float64：精度更高的浮点数
int64：分类标签常用
bool：布尔值
### Tensor 的主要能力
Tensor 不仅可以保存数据，还支持：
加减乘除
矩阵运算
形状变换
CPU 和 GPU 之间的数据移动
自动求导
神经网络的前向传播和反向传播
### Tensor 与其他数据类型的区别
Python 列表：适合保存普通数据，不适合大规模数值计算
NumPy 数组：适合数值计算，但通常不能直接使用 GPU，也不支持 PyTorch 自动求导
PyTorch Tensor：支持高效计算、GPU 加速和自动求导
## 4. `Resize`：调整图片尺寸

固定宽度和高度：

```
transforms.Resize((224, 224))
```

结果：

```
高度 = 224
宽度 = 224
```

如果只传入一个整数：

```
transforms.Resize(224)
```

它会把图片的较短边调整为 `224`，同时保持宽高比例，因此最后不一定是 `224 × 224`。

神经网络和批量加载通常需要图片尺寸一致，所以经常使用：

```
transforms.Resize((224, 224))
```

## 5. `Normalize`：标准化

```
transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)
```

它对每个通道执行：

```
标准化结果 = (原数值 - mean) / std
```

RGB 图片有三个通道，因此通常需要三个均值和三个标准差。

使用 `mean=0.5`、`std=0.5` 时，会将大致位于 `[0, 1]` 的像素转换到 `[-1, 1]`。

`Normalize` 要处理 Tensor，所以一般放在 `ToTensor()` 后面：

```
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
```

## 6. 常见裁剪操作

随机裁剪：

```
transforms.RandomCrop((224, 224))
```

从图片中随机裁剪指定大小的区域，常用于训练集的数据增强。

中心裁剪：

```
transforms.CenterCrop((224, 224))
```

从图片中心裁剪指定大小的区域，结果比较稳定，常用于验证集和测试集。

随机尺寸裁剪：

```
transforms.RandomResizedCrop((224, 224))
```

先随机选择图片区域，再调整到指定尺寸，是图像分类训练中常见的数据增强方式。

## 7. 常见翻转操作

随机水平翻转：

```
transforms.RandomHorizontalFlip(p=0.5)
```

表示图片有 `50%` 的概率水平翻转。

随机垂直翻转：

```
transforms.RandomVerticalFlip(p=0.5)
```

表示图片有 `50%` 的概率垂直翻转。

翻转是否合理取决于任务。例如自然图片通常可以水平翻转，但数字、文字和部分医学图片可能不适合随意翻转。

## 8. `ColorJitter`：改变颜色

```
transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.1
)
```

可以随机调整：

- `brightness`：亮度
- `contrast`：对比度
- `saturation`：饱和度
- `hue`：色调

它可以让模型适应不同的光照和颜色变化。

## 9. 数据增强

数据增强是指对训练图片进行随机变换，产生不同版本的训练样本。

例如：

```
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])
```

同一张图片每次读取时，可能产生不同结果。

数据增强的作用包括：

- 增加训练数据的多样性
- 降低模型死记训练图片的风险
- 提高模型的泛化能力
- 减少过拟合

## 10. 训练集与验证集的区别

训练集通常使用随机数据增强：

```
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
```

验证集和测试集通常只进行固定处理：

```
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
```

原因是验证和测试需要结果稳定、可重复，不应该每次随机改变图片。

## 11. 在 Dataset 中使用

创建数据集时传入 transform：

```
dataset = MyDataset(
    root_dir="data/train",
    transform=train_transform
)
```

在 `__getitem__()` 中应用：

```
if self.transform is not None:
    image = self.transform(image)
```

这样每次执行：

```
image, label = dataset[index]
```

都会自动对图片进行处理。

## 12. 常见注意事项

### 操作顺序

下面的顺序比较常见：

```
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

一般先进行尺寸、裁剪、翻转等图像操作，再转换成 Tensor，最后标准化。

### 输入类型

不同 transform 接受的输入类型可能不同：

- PIL 图片
- NumPy 数组
- PyTorch Tensor

使用前需要确认当前图片的类型：

```
print(type(image))
```

### 图片通道

RGB 图片通常是三个通道：

```
R、G、B
```

灰度图片通常只有一个通道。可以用 PIL 统一转换：

```
image = image.convert("RGB")
```

### Batch 尺寸

同一个 batch 中的图片形状必须一致，否则默认的 `DataLoader` 无法将它们堆叠起来。

因此常使用：

```
transforms.Resize((224, 224))
```

## 总结

`transforms` 的核心作用是：

```
预处理图片 + 数据增强 + 转换为模型需要的 Tensor
```

入门阶段重点掌握：

```
transforms.Compose()
transforms.Resize()
transforms.ToTensor()
transforms.Normalize()
transforms.RandomCrop()
transforms.RandomHorizontalFlip()
```

典型处理流程：

```
读取图片
   ↓
调整尺寸或随机增强
   ↓
转换为 Tensor
   ↓
标准化
   ↓
交给 DataLoader
   ↓
输入神经网络
```

最常用的基础写法：

```
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
```

# torchvision中数据集的使用
torchvision是处理视觉用的
```python
import torchvision

from torch.utils.tensorboard import SummaryWriter

  

dataset_transform = torchvision.transforms.Compose([

    torchvision.transforms.ToTensor()

])#转化为张量，以用于transform

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)#从训练集下载
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)


# print(test_set[0])

# print(test_set.classes)

#

# img, target = test_set[0]

# print(img)

# print(target)

# print(test_set.classes[target])

# img.show()

#

# print(test_set[0])

writer = SummaryWriter("p10")

for i in range(10):

    img, target = test_set[i]

    writer.add_image("test_set", img, i)
    
writer.close()#窗口要关闭
```