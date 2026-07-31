# 李宏毅机器学习 HW10

本作业对官方提供的 200 张 CIFAR-10 图片进行非定向迁移攻击。代码支持：

- FGSM
- I-FGSM
- MI-FGSM
- DIM-MI-FGSM
- 多代理模型 Ensemble Attack
- `dog2.png` 的 JPEG 防御实验
- 自动检查 `L-inf <= 8`
- 自动生成小于 2 MB 的 `.tgz` 提交文件

## 环境

```bash
conda activate mllearning
cd C:\Users\zss\OneDrive\Desktop\obsidian\code\ml_hw10
python -m pip install pytorchcv==0.0.74
```

安装 `pytorchcv` 不会替换现有 PyTorch。第一次运行某个代理模型时会下载相应的 CIFAR-10 预训练权重。

## 检查数据

```bash
python run.py check
```

默认数据路径为：

```text
C:\Users\zss\OneDrive\Desktop\obsidian\code\data\hw10\data
```

## 快速测试

只攻击前 8 张图片，验证代码和模型：

```bash
python run.py attack --attack fgsm --limit 8
```

使用 `--limit` 时不会生成提交压缩包。

## 生成攻击图片

FGSM：

```bash
python run.py attack --attack fgsm --overwrite
```

I-FGSM：

```bash
python run.py attack --attack ifgsm --steps 20 --alpha 0.8 --overwrite
```

MI-FGSM：

```bash
python run.py attack --attack mifgsm --steps 20 --alpha 0.8 --overwrite
```

DIM-MI-FGSM 加三模型集成：

```bash
python run.py attack --attack dim-mifgsm ^
  --models "nin_cifar10,resnet20_cifar10,preresnet20_cifar10" ^
  --steps 20 --alpha 0.8 --overwrite
```

上面的 `^` 是 Windows CMD 的换行符。在 PowerShell 中可以写成一行，或把 `^` 改成反引号；Git Bash 中使用反斜杠。

正式运行后会得到：

```text
outputs/<attack>/
├── airplane/
├── automobile/
├── ...
├── truck/
└── metrics.json

outputs/<attack>.tgz
```

`.tgz` 中只包含 10 个类别目录和 200 张 PNG，不包含 `metrics.json`。

## JPEG 防御实验

```bash
python run.py defense
```

结果保存在：

```text
outputs/defense/
├── dog2_adversarial.png
├── dog2_jpeg_defense.png
├── defense_comparison.png
└── defense_result.json
```

原作业中的 `compression rate=70%` 对应约 `JPEG quality=30`。JPEG 会丢弃一部分高频信息，因此可能减弱对抗噪声。

## 参数说明

- `epsilon` 固定为 8，程序不允许修改。
- `alpha` 和 `steps` 可以调整。
- 代理模型准确率只表示攻击在本地模型上的效果，JudgeBoi 使用未知模型，迁移效果可能不同。
- 模型越多，通常迁移性越好，但显存占用和运行时间也会增加。
