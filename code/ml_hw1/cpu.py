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

def same_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# 划分数据集
def train_valid_split(data_set, valid_ratio,seed):
    valid_size = int(len(data_set) * valid_ratio)
    train_size = len(data_set) - valid_size
    train_data,valid_data = random_split(data_set,[train_size,valid_size],generator=torch.Generator().manual_seed(seed))
    return np.array(train_data),np.array(valid_data)
# 选择特征
def select_features(train_data,valid_data,test_data,select_all=True):
    y_train = train_data[:,-1]#行选择全部，列选择最后一列
    y_valid = valid_data[:,-1]



    raw_x_train = train_data[:,:-1]#行选择全部，列选择除了最后一列之外的所有列
    raw_x_valid = valid_data[:,:-1]
    raw_x_test = test_data#训练集由于最后一列2没有标签所以直接全部选择即可

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))#选择所有特征
    else:
        feat_idx = [0,1,2,3,4,5,6,7,8,9]#选择前10个特征

    return raw_x_train[:,feat_idx],raw_x_valid[:,feat_idx],raw_x_test[:,feat_idx],y_train,y_valid

# 数据集
class COVID19Dataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.FloatTensor(features)
        if targets is None:
            self.targets = None
        else:
            self.targets = torch.FloatTensor(targets).reshape(-1, 1)
    def __getitem__(self, index):
        if self.targets is None:
            return self.features[index]
        else:
            return self.features[index], self.targets[index]
    def __len__(self):
        return len(self.features)


#神经网络
class My_Model(nn.Module):
    def __init__(self, input_dim):
        # 继承父类nn.Module的初始化方法，必须写
        super(My_Model, self).__init__()
        # 搭建串行网络层
        self.layers = nn.Sequential(
            # 第一层线性层：输入维度input_dim，输出16维隐藏特征
            nn.Linear(input_dim, 16),
            # ReLU激活函数，引入非线性，提升模型拟合能力
            nn.ReLU(),
            # 第二层线性层：输入16维，输出8维隐藏特征
            nn.Linear(16, 8),
            nn.ReLU(),
            # 输出层：输入8维，输出1个预测值（回归任务）
            nn.Linear(8, 1)
        )
    # 前向传播函数，数据流过网络的逻辑，必须实现
    def forward(self, x):
        # 将输入x送入搭建好的网络层，返回预测结果
        return self.layers(x)
    
# 参数设置
device = torch.device("cpu")
config = {
    "seed": 5201314,
    "select_all": True,
    "valid_ratio": 0.2,
    "n_epochs": 1 if os.getenv("CPU_SMOKE_TEST") == "1" else 3000,
    "batch_size": 256,
    "learning_rate": 1e-5,
    "early_stop": 400,
    "save_path": "./models/model.ckpt"
}#用字典的形式来存储参数

# 训练过程
def trainer(train_loader, valid_loader, model, config, device):
    # 回归任务损失函数：均方误差MSE，reduction='mean'对一个batch求平均loss
    criterion = nn.MSELoss(reduction='mean')
    # 随机梯度下降优化器，带动量momentum=0.9，学习率从配置读取
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    # Tensorboard可视化日志写入器，默认生成runs文件夹记录loss曲线
    writer = SummaryWriter()

    # 判断./models文件夹是否存在，不存在则新建，用于保存最优模型权重
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    # 从配置读取总训练轮数
    n_epochs = config['n_epochs']
    # 记录全局最优验证loss，初始设为无穷大，后续更小值才更新
    best_loss = math.inf
    # tensorboard绘图累计步数，每训练一个batch自增
    step = 0
    # 早停计数器：连续多轮验证loss不下降则停止训练
    early_stop_count = 0

    # 外层循环：遍历每一轮训练
    for epoch in range(n_epochs):
        # 切换模型至训练模式（启用梯度计算）
        model.train()
        # 列表记录本epoch所有batch的loss，后续求平均训练损失
        loss_record = []
        # tqdm生成训练进度条，position=0单条进度，leave=True训练完成保留进度条
        train_pbar = tqdm(train_loader, position=0, leave=True)

        # 遍历训练集每个batch的数据
        for x, y in train_pbar:
            # 梯度清零，防止上一轮梯度累积
            optimizer.zero_grad()
            # 将特征、标签迁移至指定设备（CPU/GPU）
            x, y = x.to(device), y.to(device)
            # 模型前向传播，得到预测值
            pred = model(x)
            # 计算当前batch的损失（MSE）
            loss = criterion(pred, y)
            # 反向传播，自动计算各参数梯度
            loss.backward()
            # 优化器根据梯度更新模型权重
            optimizer.step()
            # Tensorboard绘图全局步数自增
            step += 1
            # detach()断开计算图，item()取出张量loss的纯数值，存入记录列表
            loss_record.append(loss.detach().item())

            #显示训练过程
            train_pbar.set_description(f"Epoch [{epoch + 1}/{n_epochs}]")
            train_pbar.set_postfix(loss=loss.detach().item())
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/Train', mean_train_loss, step)

        #验证过程
        model.eval()  # 切换模型至评估模式（关闭梯度计算）
        loss_record = []  # 记录验证集每个batch的loss

        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():  # 禁用梯度计算，节省显存和计算
                pred = model(x)
                loss = criterion(pred, y)# 计算当前批次的损失值
                loss_record.append(loss.detach().item())# detach()脱离计算图，item()提取张量的纯数字，存入损失记录列表


        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {mean_train_loss:.4f}, Valid Loss: {mean_valid_loss:.4f}")
        writer.add_scalar('Loss/Valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])#保存最好的模型权重
            print(f"验证集损失下降，保存模型权重至 {config['save_path']}")
            early_stop_count = 0  # 重置早停计数器
        else:
            early_stop_count += 1
            if early_stop_count >= config['early_stop']:
                print("验证集损失连续多轮未下降，停止训练")
                return
            
#准备工作
same_seeds(config['seed'])  # 设置随机种子，保证实验可复现
print(f"当前使用设备：{device}")
train_path = r"C:\Users\zss\OneDrive\Desktop\obsidian\code\data\hw1\train.csv"
test_path = r"C:\Users\zss\OneDrive\Desktop\obsidian\code\data\hw1\test.csv"

train_data = pd.read_csv(train_path).values
test_data = pd.read_csv(test_path).values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'],config['seed'])  # 划分训练集和验证集
print(f"训练集样本数：{len(train_data)}, 验证集样本数：{len(valid_data)}, 测试集样本数：{len(test_data)}")
x_train, x_valid, x_test, y_train, y_valid = select_features(train_data, valid_data, test_data, config['select_all'])  # 选择特征
print(f"训练集特征维度：{x_train.shape[1]}, 验证集特征维度：{x_valid.shape[1]}, 测试集特征维度：{x_test.shape[1]}")
train_dataset = COVID19Dataset(x_train, y_train)  # 创建训练集数据集对象
valid_dataset = COVID19Dataset(x_valid, y_valid)  # 创建验证集数据集
test_dataset = COVID19Dataset(x_test)  # 创建测试集数据集对象（无标签）
# 准备data loader，批量加载数据，shuffle=True表示每个epoch打乱数据顺序
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=(device.type == "cuda"))  # 创建训练集数据加载器
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=(device.type == "cuda"))  # 创建验证集数据加载器
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=(device.type == "cuda"))  # 创建测试集数据加载器

#开始训练
model = My_Model(input_dim=x_train.shape[1]).to(device)  # 创建模型实例，并迁移至指定设备
trainer(train_loader, valid_loader, model, config, device)  # 调用训练函数

# 预测

def predict(model, test_loader, device):
    model.eval()  # 切换模型至评估模式（关闭梯度计算）
    preds = []  # 用于存储所有预测结果
    for x in tqdm(test_loader, position=0, leave=True):
        x = x.to(device)  # 将特征迁移至指定设备
        with torch.no_grad():  # 禁用梯度计算，节省显存和计算
            pred = model(x)  # 模型前向传播，得到预测值
            preds.append(pred.detach().cpu())  # 将预测结果迁移回CPU
    preds = torch.cat(preds, dim=0).numpy().reshape(-1)  # 将所有批次的预测结果拼接成一个整体，并转换为NumPy数组
    return preds


def save_pred(preds, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'value'])  # 写入表头
        for i, value in enumerate(preds):
            writer.writerow([i, float(value)])  # 写入每一行数据


# 加载最优模型权重并生成预测结果
if os.path.exists(config['save_path']):
    model.load_state_dict(torch.load(config['save_path'], map_location=device))
    predictions = predict(model, test_loader, device)
    output_path = os.path.join(os.path.dirname(config['save_path']), 'predictions.csv')
    save_pred(predictions, output_path)
    print(f"预测结果已保存至 {output_path}")
else:
    print("未找到训练好的模型权重，无法生成预测结果")