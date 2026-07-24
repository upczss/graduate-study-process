# 李宏毅机器学习 HW6

本项目使用一个统一训练入口实现三种模式：

- `dcgan`：Sigmoid + BCE loss + Adam
- `wgan`：Wasserstein loss + weight clipping + RMSProp
- `wgan_gp`：Wasserstein loss + gradient penalty + Adam

模型默认生成 64×64 RGB 动漫头像。三种模式共享 Generator 和数据处理，
但 Discriminator 输出、损失函数、优化器及训练约束不同。

## 准备环境和数据

```bash
pip install -r requirements.txt
```

当前代码默认读取：

```text
C:\Users\zss\OneDrive\Desktop\obsidian\code\data\hw6\faces
```

该目录已经包含 71,314 张图片，因此运行时可以省略 `--data-dir`。程序也支持
通过 `--data-dir` 使用其他位置的数据。支持的目录结构如下：

```text
faces/                 或    dataset/
├── 0.jpg                    └── faces/
├── 1.jpg                        ├── 0.jpg
└── ...                          └── ...
```

## 训练

在本目录中运行，Windows 路径含空格时需要加引号：

```bash
python train.py --mode dcgan --epochs 50
python train.py --mode wgan --epochs 50
python train.py --mode wgan_gp --epochs 50
```

结果会保存在：

```text
runs/
└── wgan_gp/
    ├── config.json
    ├── samples/
    └── checkpoints/
        ├── epoch_005.pt
        └── latest.pt
```

如果显存不足，可降低 batch size：

```bash
python train.py --mode wgan_gp --batch-size 32
```

从 checkpoint 继续：

```bash
python train.py --mode wgan_gp --epochs 100 \
  --resume runs/wgan_gp/checkpoints/latest.pt
```

## 生成提交文件

选择效果最好的 checkpoint，通常优先尝试 WGAN-GP：

```bash
python generate.py \
  --checkpoint runs/wgan_gp/checkpoints/latest.pt \
  --num-images 1000 \
  --output-dir generated_images \
  --archive images.tgz
```

脚本会生成 `1.jpg` 到 `1000.jpg`，并创建内部不含文件夹的
`images.tgz`。如果压缩包超过原作业的 2 MB 限制，可降低 JPEG 质量：

```bash
python generate.py --checkpoint runs/wgan_gp/checkpoints/latest.pt \
  --jpeg-quality 70 --overwrite
```

训练前建议在 Colab 中选择 GPU Runtime。正式长时间训练前，可以先使用
少量数据和 1 个 epoch 检查流程是否正常。
