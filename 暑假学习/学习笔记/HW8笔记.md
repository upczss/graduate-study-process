# HW8：使用 Autoencoder 进行异常检测

这次作业是无监督图片异常检测。训练集只包含正常人脸，模型需要学习正常图片的
特征。测试时，重建误差较大的图片会得到更高的异常分数。

## 一、读取和处理数据

```python
images = np.load("trainingset.npy", mmap_mode="r")
```

训练集约有 1.2 GB，使用内存映射可以按需读取图片，不需要一次把整个数组复制
到内存中。

```python
image = images[index].copy()
image = torch.from_numpy(image).permute(2, 0, 1).float()
image = image / 127.5 - 1.0
```

原始图片结构是 `64×64×3`，PyTorch 使用 `3×64×64`，所以需要调整维度。
像素也会从 `[0, 255]` 标准化到 `[-1, 1]`。

```python
if augment and torch.rand(()) < 0.5:
    image = torch.flip(image, dims=(2,))
```

训练时随机水平翻转图片，可以增加数据变化，同时不会改变人脸是否正常。

## 二、Autoencoder 模型

Autoencoder 由 Encoder 和 Decoder 组成：

```text
输入图片 → Encoder → latent vector → Decoder → 重建图片
```

Encoder 把图片压缩成较小的 latent representation，Decoder 再根据 latent
vector 恢复图片。

### FCN Autoencoder

```python
self.encoder = nn.Sequential(
    nn.Linear(64 * 64 * 3, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, latent_dim),
)
```

FCN 会先把图片拉平成一维向量，再通过全连接层逐渐压缩。结构比较简单，但不会
直接保留图片中像素的空间关系。

```python
self.decoder = nn.Sequential(
    nn.Linear(latent_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 64 * 64 * 3),
    nn.Tanh(),
)
```

Decoder 的结构与 Encoder 大致相反，最后通过 `Tanh` 输出 `[-1, 1]` 范围的
重建图片。

### CNN Autoencoder

```python
self.encoder = nn.Sequential(
    nn.Conv2d(3, 32, 4, 2, 1),
    nn.LeakyReLU(0.2),
    nn.Conv2d(32, 64, 4, 2, 1),
    nn.LeakyReLU(0.2),
)
```

CNN 使用卷积层缩小图片，能够学习眼睛、鼻子和脸部轮廓等局部特征，更适合处理
图片数据。

```python
self.decoder = nn.Sequential(
    nn.ConvTranspose2d(256, 128, 4, 2, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(128, 64, 4, 2, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 3, 4, 2, 1),
    nn.Tanh(),
)
```

转置卷积会逐步放大特征图，最终恢复成 `3×64×64` 的彩色图片。

### VAE

```python
mean = self.mean(hidden)
log_variance = self.log_variance(hidden)

standard_deviation = torch.exp(0.5 * log_variance)
latent = mean + torch.randn_like(standard_deviation) * standard_deviation
```

VAE 不直接生成一个固定 latent vector，而是学习潜在分布的均值和方差，再从
分布中采样。这种方法能让 latent space 更加连续。

```python
kl_loss = -0.5 * torch.mean(
    1 + log_variance - mean.pow(2) - log_variance.exp()
)

loss = reconstruction_loss + beta * kl_loss
```

VAE 的 loss 由重建误差和 KL Divergence 组成。KL loss 用来限制潜在分布，但
权重过高可能使重建图片变模糊。

## 三、模型训练

```python
noisy_images = (
    clean_images + torch.randn_like(clean_images) * noise_std
).clamp(-1.0, 1.0)

output = model(noisy_images)
loss = F.mse_loss(output, clean_images)
```

训练时可以给输入加入少量随机噪声，但目标仍然是原始干净图片。这种方式叫
Denoising Autoencoder，可以避免模型只学习简单复制输入。

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

每个 batch 都会先清空旧梯度，再进行反向传播和参数更新，使重建误差逐渐下降。

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5,
)
```

训练使用 AdamW。`weight_decay` 可以对参数进行轻微限制，减少模型过度拟合训练
图片。

```python
with torch.amp.autocast("cuda", enabled=use_fp16):
    output = model(images)
    loss = calculate_loss(output, images)
```

有 GPU 时使用 FP16，可以减少显存占用并提高训练速度。

```python
if validation_loss < best_validation_loss:
    torch.save(checkpoint, "best.pt")
```

训练集会划分出一小部分正常图片作为验证集。验证重建误差更低时，保存新的最佳
模型。

## 四、异常分数

```python
reconstructed = model.reconstruct(images)

score = torch.sqrt(
    (reconstructed - images).pow(2).flatten(1).sum(dim=1)
)
```

异常分数来自输入图片和重建图片之间的误差。模型熟悉正常人脸，因此通常能很好
地重建正常图片；没有见过的异常图片更难重建，分数应该更高。

```text
正常图片 → 重建效果好 → 分数低
异常图片 → 重建效果差 → 分数高
```

VAE 测试时使用 latent distribution 的均值进行重建，不进行随机采样，从而让
同一张图片每次得到相同的异常分数。

```python
flipped = torch.flip(images, dims=(3,))
score = (original_score + flipped_score) / 2
```

可选的 TTA 会分别计算原图和水平翻转图片的重建误差，再取平均值，使异常分数
更稳定。

## 五、评价和提交

作业使用 ROC AUC 评价模型：

```text
AUC = 1.0 → 能完美区分正常和异常
AUC = 0.5 → 接近随机排序
```

ROC AUC 关注异常图片的分数是否普遍高于正常图片，不需要提前设置一个固定分类
阈值。

```python
with open("prediction.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "score"])

    for image_id, score in enumerate(scores):
        writer.writerow([image_id, score])
```

最终需要为19,636张测试图片分别输出一个异常分数。Kaggle 提交文件格式为：

```csv
ID,score
0,25.3821
1,61.7945
```

Kaggle 只需要上传 `prediction.csv`，不需要上传训练模型或数据集。

## 六、总结

HW8 的核心是让 Autoencoder 学习正常人脸的分布，再利用重建误差检测未知分布。
模型不能只追求很低的训练 loss：如果重建能力过强，异常图片也可能被完整还原，
反而会降低正常与异常图片之间的分数差距。

FCN、CNN 和 VAE 的主要区别在于图片的压缩方式以及 latent representation。
实际效果需要结合重建图片和 Kaggle ROC AUC 进行比较。
