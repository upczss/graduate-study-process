# 李宏毅机器学习 HW7

本项目使用 Hugging Face BERT 完成中文抽取式问答。代码会自动读取：

```text
C:\Users\zss\OneDrive\Desktop\obsidian\code\data\hw7
```

该目录中应包含：

```text
hw7_train.json
hw7_dev.json
hw7_test.json
```

## 文件说明

- `data.py`：滑动窗口、答案下标转换和答案后处理
- `train.py`：训练、验证、线性学习率调整和保存最佳模型
- `predict.py`：预测测试集并生成 Kaggle 的 `result.csv`
- `check_data.py`：检查数据数量及训练答案下标

## 安装

建议使用带 NVIDIA GPU 的 Colab 或本地 PyTorch 环境：

```bash
pip install -r requirements.txt
```

## 先做快速测试

以下命令只使用少量题目，目的是检查整个流程能否运行：

```bash
python train.py --limit-train 100 --limit-dev 50 --epochs 1
```

如果快速测试成功，可以删除测试产生的 `outputs`，再开始正式训练。

## 正式训练

```bash
python train.py --epochs 2 --batch-size 8 --gradient-accumulation 4
```

默认模型是 `bert-base-chinese`。显存不足时可以降低 batch size：

```bash
python train.py --epochs 2 --batch-size 4 --gradient-accumulation 8
```

如果想尝试更好的 Hugging Face 中文预训练模型：

```bash
python train.py --model-name hfl/chinese-roberta-wwm-ext --epochs 2
```

训练结果保存在：

```text
outputs/
├── best_model/
├── epoch_1/
└── epoch_2/
```

`best_model` 是验证集 Exact Match 最高的模型。

## 生成 Kaggle 文件

```bash
python predict.py
```

程序会读取 `outputs/best_model`，并在当前项目中生成：

```text
result.csv
```

正常文件应包含表头和 4,957 条预测：

```csv
ID,Answer
0,羽毛
1,某个答案
```

Kaggle 只需要上传 `result.csv`，不需要上传模型、代码或数据集。

## 主要改进

与原始 Colab 示例相比，本实现包含：

- 重叠滑动窗口，减少边界答案丢失
- 线性学习率衰减和 warmup
- FP16 混合精度
- Gradient accumulation
- 只从文章 token 中选择答案
- 保证 `end_index >= start_index`
- 限制最大答案长度
- 比较多个 start/end 组合，而不是分别独立取最大值
- 使用原文字符偏移截取答案，避免中文 decode 空格问题
