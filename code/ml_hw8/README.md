# 李宏毅机器学习 HW8

本项目使用 Autoencoder 完成无监督图片异常检测。代码默认读取：

```text
C:\Users\zss\OneDrive\Desktop\obsidian\code\data\hw8
```

其中包含：

```text
trainingset.npy   # 100,000 张正常人脸
testingset.npy    # 19,636 张待检测图片
```

## 文件说明

- `data.py`：使用内存映射读取 `.npy`，并把图片标准化到 `[-1, 1]`
- `models.py`：FCN、CNN 和 VAE 三种 Autoencoder
- `train.py`：训练、验证、保存模型及重建预览
- `predict.py`：计算异常分数并生成 Kaggle CSV
- `check_data.py`：检查数据形状和类型

## 检查数据

```bash
python check_data.py
```

## 快速测试

先用少量图片确认训练流程能够运行：

```bash
python train.py --model fcn --epochs 1 --limit-train 1000
```

## 正式训练

建议先训练 FCN，它与公开复现中效果较好的基础方案接近：

```bash
python train.py --model fcn --latent-dim 32 --epochs 50 --batch-size 256
```

也可以尝试 CNN：

```bash
python train.py --model cnn --latent-dim 128 --epochs 50 --batch-size 256
```

或 VAE：

```bash
python train.py --model vae --latent-dim 128 --epochs 50 --batch-size 256
```

如果显存不足，把 batch size 改为 128 或 64。

结果结构：

```text
outputs/
└── fcn/
    ├── config.json
    ├── checkpoints/
    │   ├── best.pt
    │   └── latest.pt
    └── previews/
```

`previews` 中每张图片的第一行是原图，第二行是对应重建结果。

## 生成 Kaggle 文件

训练 FCN 后直接运行：

```bash
python predict.py
```

使用 CNN 或 VAE 时需要指定 checkpoint：

```bash
python predict.py --checkpoint outputs/cnn/checkpoints/best.pt
python predict.py --checkpoint outputs/vae/checkpoints/best.pt
```

可选的水平翻转测试增强：

```bash
python predict.py --tta
```

最终会生成：

```text
prediction.csv
```

格式为：

```csv
ID,score
0,12.345678
1,7.891234
```

Kaggle 只需要上传 `prediction.csv`。`score` 越高，表示图片越异常。
