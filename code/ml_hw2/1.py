"""引入一些模块"""
# 数据操作
import math
import numpy as np
# 读取、写入数据
import pandas as pd
import os
import csv
# 进度条
from tqdm import tqdm
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
# 绘制图像、Tensorboard可视化
from torch.utils.tensorboard import SummaryWriter
import random
import gc #回收内存节约资源

# 定义操作
def same_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
#读取pt文件（加载特征）
def load_feat(patch):
    feat = torch.load(patch)
    return feat
#拼接frame
def shift(x,n):
    if n<0:
        left = x[0].repeat(-n,1)
        right = x[:n]
    elif n>0:
        right = x[-1].repeat(n,1)
        left = x[n:]
    else:
        return x
    return torch.cat((left,right),dim=0)

def concat_feat(x,concat_n):
    assert concat_n%2==1 #必须是奇数
    if concat_n<2:
        return x
    seq_len = x.size(0)#frame的个数
    feat_dim = x.size(1)#特征维度39
    x = x.repeat(1,concat_n)#重复concat_n次
    x = x.view(seq_len,concat_n,feat_dim).permute(1,0,2)#permute变换维度的顺序
    #平移操作
    mid = concat_n//2
    for r_idx in range(1,mid+1):
        x[mid+r_idx, :] = shift(x[mid+r_idx, :], r_idx)
        x[mid-r_idx, :] = shift(x[mid-r_idx, :], -r_idx)
    return x.permute(1,0,2).view(seq_len,concat_n*feat_dim)#permute变换维度的顺序

# 预处理数据
def preprocess_data(split,feat_dir,phone_path,concat_nframes,train_ratio=0.8,train_val_seed=1337):
    class_num = 41 #一共有41个类别
    mode = 'train' if split in ('train', 'val') else 'test'

    label_dict = {}#索引字典

    if mode != 'test':
        # 拼接标签文件完整路径并读取所有行
        label_file_path = os.path.join(phone_path, f'{mode}_labels.txt')
        phone_file = open(label_file_path).readlines()
        # 逐行解析标签
        for line in phone_file:
            # 去除换行符，按空格切分
            line = line.strip('\n').split(' ')
            # line[0]为音频文件名，后面数字是对应音素编号
            label_dict[line[0]] = [int(p) for p in line[1:]]


    # 划分数据集
    # 训练集和验证集共用 train_split.txt
    if split == "train" or split == "val":
        usage_list = open(
            os.path.join(phone_path, "train_split.txt")
        ).readlines()

        # 固定随机种子，使每次划分结果一致
        random.seed(train_val_seed)
        random.shuffle(usage_list)

        # 计算训练集样本数量
        percent = int(len(usage_list) * train_ratio)

        if split == "train":
            # 前 train_ratio 部分作为训练集
            usage_list = usage_list[:percent]

        elif split == "val":
            # 剩余部分作为验证集
            usage_list = usage_list[percent:]


    # 测试集使用单独的文件列表
    elif split == "test":
        usage_list = open(
            os.path.join(phone_path, "test_split.txt")
        ).readlines()


    # split 参数不合法时抛出错误
    else:
        raise ValueError("Invalid 'split' argument for dataset: PhoneDataset!")


    # 去掉每一行末尾的换行符
    usage_list = [line.strip("\n") for line in usage_list]

    # 输出当前数据集的信息
    print(
        "[Dataset] - # Phone Classes: " + str(class_num)
        + ", number of utterance for " + split
        + ": " + str(len(usage_list))
    )
    max_len = 3000000
    x = torch.empty(max_len,39*concat_nframes)
    if mode != 'test':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in enumerate(tqdm(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f"{fname}.pt"))
        cur_len = len(feat)
        feat = concat_feat(feat,concat_nframes)
        if mode != 'test':
            label = torch.LongTensor(label_dict[fname])
           
        x[idx:idx+cur_len] = feat
        if mode != 'test':
            y[idx:idx+cur_len] = label 

        idx += cur_len

    x = x[:idx, :]
    if mode != 'test':
        y = y[:idx]

    print(f'[INFO]{split}set')
    print(x.shape)
    if mode != 'test':
        print(y.shape)
        return x,y
    else:
        return x


# dataset
class LibriDataset(Dataset):
    def __init__(self, x, y=None):
        # 保存输入特征
        self.feature = x

        # y 存在时，表示训练/验证集
        if y is not None:
            # 分类标签必须是整数类型
            self.label = torch.LongTensor(y)
        else:
            # 测试集通常没有标签
            self.label = None

    def __getitem__(self, idx):
        # 有标签时返回：特征、标签
        if self.label is not None:
            return self.feature[idx], self.label[idx]

        # 无标签时只返回特征
        else:
            return self.feature[idx]

    def __len__(self):
        # 返回数据集中的样本数
        return len(self.feature)

    
# #Model
# class BasicBlock(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(BasicBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         x = self.block(x)
#         return x
    
# class Classifier(nn.Module):
#     def __init__(self, input_dim,output_dim=41,hidden_layers=1,hidden_dim=256):
#         super(Classifier,self).__init__()
#         self.fc == nn.Sequential(
#             BasicBlock(input_dim,hidden_dim),#输入层
#             *[BasicBlock(input_dim,hidden_dim) for_in range(hidden_layers)],#隐含层
#             nn.Linear(hidden_dim,output_dim)#输出层
#         )
    
#     def forward(self,x):
#         x = self.fc(x)
#         return x


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        # 初始化父类 nn.Module
        super(BasicBlock, self).__init__()

        # 线性层后接 ReLU 激活函数
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # 数据经过当前模块
        return self.block(x)


# 分类模型：输入特征 -> 多层全连接网络 -> 41 类预测结果
class Classifier(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=41,
        hidden_layers=1,
        hidden_dim=256
    ):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            # 输入层：input_dim -> hidden_dim
            BasicBlock(input_dim, hidden_dim),

            # 隐藏层：hidden_dim -> hidden_dim
            *[
                BasicBlock(hidden_dim, hidden_dim)
                for _ in range(hidden_layers)
            ],

            # 输出层：输出 41 个类别的分数（logits）
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # 输入 x 的形状通常为：[batch_size, input_dim]
        # 输出形状为：[batch_size, 41]
        return self.fc(x)

# 参数定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
concat_nframes = 3
train_ratio = 0.8
config = {
    "seed": 0,
    "batch_size": 512,
    "learning_rate": 1e-4,
    "n_epochs": 1,
    "model_path": r"C:\Users\zss\OneDrive\Desktop\obsidian\code\ml_hw2\model.ckpt"
}#用字典的形式来存储参数

input_dim = 39*concat_nframes#拼接了三帧
hidden_layers= 3
hidden_dim = 256

# 训练过程
# 训练函数
def trainer(train_set, val_set, train_loader, val_loader, config, model, device):
    # 多分类任务的损失函数
    criterion = nn.CrossEntropyLoss()

    # AdamW 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"]
    )

    # 记录最佳验证集准确率
    best_acc = 0.0

    # 总训练轮数
    n_epochs = config["n_epochs"]

    # 开始训练
    for epoch in range(n_epochs):
        train_acc = 0.0
        train_loss = 0.0

        # 切换为训练模式
        model.train()

        # 遍历训练集的每个 batch
        for i, batch in enumerate(tqdm(train_loader)):
            # 取出特征和标签
            features, labels = batch

            # 将数据移动到 CPU 或 GPU
            features = features.to(device)
            labels = labels.to(device)

            # 清空上一轮的梯度
            optimizer.zero_grad()

            # 模型预测，输出每个类别的分数
            outputs = model(features)

            # 计算预测结果与真实标签之间的损失
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, train_pred = torch.max(outputs,dim=1)



            # 对比预测与真实标签，相等为True(1)、不等为False(0)；求和得到本batch正确样本总数，累加至总正确数
            train_acc += (train_pred.detach() == labels.detach()).sum().item()

            # 累加当前批次损失（loss是tensor，.item()转为普通数字）
            train_loss += loss.item()

        #验证环节
        # validation 验证阶段代码
        if len(val_set) > 0:
            val_acc = 0.0
            val_loss = 0.0
            # 切换模型为评估模式（关闭Dropout等训练专属层）
            model.eval()
            # 关闭梯度计算，节省显存、加速推理
            with torch.no_grad():
                # 遍历验证集，带进度条
                for i, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    # 数据迁移到GPU/CPU设备
                    features = features.to(device)
                    labels = labels.to(device)
                    # 模型前向传播得到各类别得分
                    outputs = model(features)
                    # 计算当前批次损失
                    loss = criterion(outputs, labels)
                    # dim=1取概率最大的类别下标，丢弃概率值
                    _, val_pred = torch.max(outputs, 1)
                    # 预测、标签转CPU后对比，统计正确样本数量并累加总正确数
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                    # 累加批次损失
                    val_loss += loss.item()
                # 当验证集准确率优于历史最佳结果时，保存模型
                if val_acc > best_acc:
                    # 更新最佳验证集正确预测数量
                    best_acc = val_acc

                    # 只保存模型参数
                    torch.save(model.state_dict(), config["model_path"])

                    # 输出当前最佳验证集准确率
                    print(
                        "saving model with acc: {:.3f}".format(
                            best_acc / len(val_set)
                        )
                    )

    if len(val_set) == 0:
        torch.save(model.state_dict(), config["model_path"])
        print("saving model at last epoch")

    # 释放 DataLoader 占用的内存
    del train_loader, val_loader
    gc.collect()
    return 


# 训练准备

same_seeds(config["seed"])

# LibriPhone 数据集根目录
data_dir = r"C:\Users\zss\OneDrive\Desktop\obsidian\code\data\hw2\libriphone\libriphone"

train_x, train_y = preprocess_data(
    split="train",
    feat_dir=os.path.join(data_dir, "feat"),
    phone_path=data_dir,
    concat_nframes=concat_nframes,
    train_ratio=0.8,
    train_val_seed=config["seed"]
)

val_x, val_y = preprocess_data(
    split="val",
    feat_dir=os.path.join(data_dir, "feat"),
    phone_path=data_dir,
    concat_nframes=concat_nframes,
    train_ratio=0.8,
    train_val_seed=config["seed"]
)
train_set = LibriDataset(train_x, train_y)
val_set = LibriDataset(val_x, val_y)
        
del train_x, train_y, val_x, val_y
gc.collect()

#dataloader
train_loader = DataLoader(
    train_set,
    batch_size=config["batch_size"],
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=config["batch_size"],
    shuffle=False,
    pin_memory=True
)

#开始训练


# 开始训练
print(f"DEVICE: {device}")

# 建立分类模型并移动到指定设备
model = Classifier(
    input_dim=input_dim,
    hidden_layers=hidden_layers,
    hidden_dim=hidden_dim
).to(device)

# 训练模型
trainer(
    train_set,
    val_set,
    train_loader,
    val_loader,
    config,
    model,
    device
)

def predict(model, test_loader, device):
    # 存放所有测试样本的预测类别
    pred = np.array([], dtype=np.int32)

    # 切换到评估模式
    model.eval()

    # 预测时不计算梯度，节省显存和时间
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # 测试集没有标签，batch 就是特征
            features = batch.to(device)

            # 输出每个类别的分数
            outputs = model(features)

            # 取得分数最高的类别编号
            _, test_pred = torch.max(outputs, dim=1)

            # 将当前 batch 的预测结果拼接起来
            pred = np.concatenate(
                [pred, test_pred.cpu().numpy()],
                axis=0
            )

    return pred

def save_pred(preds, file):
    # 创建 CSV 文件并写入表头
    with open(file, "w") as f:
        f.write("Id,Class\n")

        # 逐行写入：样本编号、预测类别
        for i, y in enumerate(preds):
            f.write("{},{}\n".format(i, y))

# 测试集预测准备

# 读取并预处理测试集特征
test_x = preprocess_data(
    split="test",
    feat_dir=os.path.join(data_dir, "feat"),
    phone_path=data_dir,
    concat_nframes=concat_nframes
)

# 测试集没有标签
test_set = LibriDataset(test_x, y=None)

# 测试时不打乱数据顺序
test_loader = DataLoader(
    test_set,
    batch_size=config["batch_size"],
    shuffle=False,
    pin_memory=True
)

# 重新建立模型，并读取训练期间保存的最佳权重
model = Classifier(
    input_dim=input_dim,
    hidden_layers=hidden_layers,
    hidden_dim=hidden_dim
).to(device)

model.load_state_dict(torch.load(config["model_path"]))
pred = predict(model, test_loader, device)
save_pred(pred, "prediction.csv")
