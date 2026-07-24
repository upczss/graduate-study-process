# HW6：使用 GAN 生成动漫头像

这次作业要让模型从随机噪声中生成动漫头像。我使用同一套代码实现
DCGAN、WGAN 和 WGAN-GP，并通过 `mode` 切换训练方式。三种模式使用相同的
数据和 Generator，主要区别在 Discriminator、损失函数和训练限制。

## 一、数据处理

```python
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3),
])
```

我先把图片调整为 `64×64`，再随机水平翻转，增加一点训练数据的变化。
最后把像素标准化到 `[-1, 1]`，因为 Generator 的 `Tanh` 输出也是这个范围。

```python
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
)
```

我使用 DataLoader 分批读取图片，并在每个 epoch 打乱顺序。这样不用一次把
全部图片放进显存。

## 二、模型部分

### Generator

```python
self.network = nn.Sequential(
    self._block(latent_dim, 512, 4, 1, 0),
    self._block(512, 256, 4, 2, 1),
    self._block(256, 128, 4, 2, 1),
    self._block(128, 64, 4, 2, 1),
    nn.ConvTranspose2d(64, 3, 4, 2, 1),
    nn.Tanh(),
)
```

Generator 接收一个随机向量，并用多层转置卷积逐渐放大。图片大小会从
`1×1` 变成 `4×4、8×8、16×16、32×32`，最后得到 `64×64` 的彩色图片。

```python
noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
fake_images = generator(noise)
```

这里的 `noise` 是随机噪声。Generator 学到的不是某一张固定图片，而是如何
把不同的随机噪声转换成不同的动漫头像。

### Discriminator / Critic

```python
layers = [
    nn.Conv2d(3, 64, 4, 2, 1),
    nn.LeakyReLU(0.2),
    # 后面继续使用卷积缩小图片
    nn.Conv2d(512, 1, 4, 1, 0),
]

if mode == "dcgan":
    layers.append(nn.Sigmoid())
```

Discriminator 使用卷积不断缩小图片，最后输出一个数。DCGAN 使用
`Sigmoid` 得到真假概率；WGAN 和 WGAN-GP 不使用 `Sigmoid`，输出的是图片的
真实性分数，所以这时一般叫它 Critic。

```python
if mode == "dcgan":
    layers.append(nn.BatchNorm2d(out_channels))
```

我的 DCGAN 判别器使用 BatchNorm。WGAN 和 WGAN-GP 的 Critic 没有使用
BatchNorm，避免一个样本的评分受到同一批其他样本的影响。

## 三、三种训练模式

### DCGAN

```python
real_targets = torch.ones_like(real_scores)
fake_targets = torch.zeros_like(fake_scores)

loss_d = criterion(real_scores, real_targets)
loss_d += criterion(fake_scores, fake_targets)
```

训练 Discriminator 时，我希望真实图片的结果接近 1，生成图片的结果接近
0，因此使用二元交叉熵损失。

```python
loss_g = criterion(fake_scores, torch.ones_like(fake_scores))
```

训练 Generator 时，我把生成图片的目标设为 1。也就是让 Generator 尽量生成
能够被 Discriminator 判断为真实的图片。

### WGAN

```python
loss_d = fake_scores.mean() - real_scores.mean()
loss_g = -fake_scores.mean()
```

WGAN 不再进行普通的真假二分类，而是让真实图片获得较高分，让生成图片获得
较低分。Generator 的目标则是提高生成图片的分数。

```python
optimizer_d.step()

with torch.no_grad():
    for parameter in discriminator.parameters():
        parameter.clamp_(-0.01, 0.01)
```

原始 WGAN 会把 Critic 的参数限制在一个很小的范围中，这一步叫 weight
clipping。它可以满足 WGAN 需要的限制，但也可能让梯度变得太小。

### WGAN-GP

```python
alpha = torch.rand(batch_size, 1, 1, 1, device=device)
interpolated = alpha * real_images + (1 - alpha) * fake_images
interpolated.requires_grad_(True)
```

WGAN-GP 不裁剪参数，而是在真实图片和生成图片之间制造一批插值图片，然后
检查 Critic 对这些图片的梯度。

```python
gradients = autograd.grad(
    outputs=critic(interpolated),
    inputs=interpolated,
    grad_outputs=torch.ones(batch_size, device=device),
    create_graph=True,
)[0]

penalty = ((gradients.reshape(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
loss_d = fake_scores.mean() - real_scores.mean() + 10 * penalty
```

Gradient Penalty 希望梯度范数接近 1。如果偏离 1，就把惩罚加到 Critic 的
loss 中。相比 weight clipping，这种做法通常能让训练更加稳定。

## 四、训练流程

```python
optimizer_d.zero_grad()
fake_images = generator(noise).detach()
loss_d = calculate_discriminator_loss(real_images, fake_images)
loss_d.backward()
optimizer_d.step()
```

我先训练 Discriminator/Critic。这里对生成图片调用 `detach()`，是因为这一步
只更新判别器，不需要计算 Generator 的梯度。

```python
optimizer_g.zero_grad()
generated = generator(noise)
generated_scores = discriminator(generated)
loss_g = calculate_generator_loss(generated_scores)
loss_g.backward()
optimizer_g.step()
```

接着训练 Generator。这次不使用 `detach()`，因为梯度需要从 Discriminator
的输出一直传回 Generator。

```python
update_generator = (
    mode == "dcgan"
    or global_step % critic_iterations == 0
)
```

DCGAN 每批都会更新一次 Generator。WGAN 和 WGAN-GP 通常先多训练几次
Critic，再训练一次 Generator，让 Critic 先提供比较可靠的评分。

```python
if mode == "dcgan":
    optimizer = optim.Adam(parameters, lr=2e-4, betas=(0.5, 0.999))
elif mode == "wgan":
    optimizer = optim.RMSprop(parameters, lr=5e-5)
else:
    optimizer = optim.Adam(parameters, lr=1e-4, betas=(0.0, 0.9))
```

三种模式使用的优化器设置不同。DCGAN 和 WGAN-GP 使用 Adam，原始 WGAN
使用 RMSProp。学习率太大容易让 GAN 训练不稳定。

## 五、保存与生成结果

```python
torch.save({
    "mode": mode,
    "generator": generator.state_dict(),
    "discriminator": discriminator.state_dict(),
    "optimizer_g": optimizer_g.state_dict(),
    "optimizer_d": optimizer_d.state_dict(),
}, "latest.pt")
```

我会同时保存两个模型和优化器，这样训练中断后可以继续。最终生成图片时，
其实只需要加载 Generator。

```python
generator.eval()

with torch.inference_mode():
    noise = torch.randn(1000, latent_dim, 1, 1, device=device)
    images = generator(noise)
```

生成阶段使用 `eval()` 和 `inference_mode()`，因为这时不需要计算梯度。我给
Generator 1000 个随机向量，就可以得到 1000 张不同的动漫头像。

```python
for index, image in enumerate(images, start=1):
    image = image.add(1).div(2).clamp(0, 1)
    to_pil_image(image.cpu()).save(f"{index}.jpg")
```

我先把图片从 `[-1, 1]` 转回 `[0, 1]`，再保存为 JPG。作业要求图片命名为
`1.jpg` 到 `1000.jpg`，最后直接压缩成 `images.tgz`，内部不能多套文件夹。

## 六、我的总结

DCGAN 是最基础的版本，容易理解，但训练可能不稳定。WGAN 改用
Wasserstein loss，并通过 weight clipping 限制 Critic。WGAN-GP 又用
Gradient Penalty 替代 weight clipping，一般能得到更稳定的训练过程。

这次作业中最重要的是理解 Generator 和 Discriminator 如何交替更新，以及
三种模式在输出层、loss 和训练限制上的区别。
