# Latent-diffusion-model
# CIFAR-100 Latent Diffusion Model (LDM)

本项目实现了一个基于CIFAR-100数据集的Latent Diffusion Model (LDM)。项目包含两个主要部分：变分自编码器（VAE）和潜在扩散模型（LDM）。
在我的理解上

## 项目结构

```
VAE/
├── simple_vae.py              # VAE模型实现
├── train_cifar100_vae.py      # VAE训练脚本
├── simplified_ldm.py          # 简化版LDM实现
├── train_cifar100_ldm.py      # LDM训练脚本
├── vae_adapter.py             # VAE适配器和数据处理
├── sample_and_fid.py          # 采样和FID评估脚本
└── configs/                   # 配置文件目录
    └── latent-diffusion/
        └── cifar100-ldm.yaml  # LDM配置文件
```

## 模型架构

### 1. VAE (Variational Autoencoder)
- 编码器：将32x32x3的图像编码到128维潜在空间
- 解码器：将潜在向量重建为原始图像
- 使用KL散度权重(1e-5)来平衡重建质量和分布匹配

### 2. LDM (Latent Diffusion Model)
- 在VAE的潜在空间中进行扩散
- 使用条件生成（类别标签）
- 1000个扩散时间步
- β调度：从0.0001线性增加到0.02

## 数据处理

使用CIFAR-100数据集，包含以下预处理步骤：
1. 随机水平翻转 (p=0.5)
2. 随机裁剪 (32x32, padding=4)
3. 归一化到[-1, 1]范围
4. 类别标签条件化

## 训练过程

1. VAE训练：
```bash
python train_cifar100_vae.py --batch_size 64 --epochs 10 --lr 0.001
```

2. LDM训练：
```bash
python train_cifar100_ldm.py --vae_ckpt checkpoints/simple_cifar100_vae/last.ckpt --batch_size 32 --epochs 50
```

## 生成样本

使用训练好的模型生成样本：
```bash
python sample_and_fid.py --vae_ckpt path/to/vae.ckpt --ldm_ckpt path/to/ldm.ckpt --num_samples 512
```

## 实验结果

1. VAE训练：
- 最佳验证损失：4245.6528（epoch 2）
- 最终验证损失：5380.2090（epoch 9）
- KL散度保持在较小水平（<0.001）

2. LDM训练：
- 训练了15个epoch
- 最终验证损失：0.0822
- FID分数：157.5509

## 环境要求

```
python >= 3.8
pytorch >= 1.7.0
torchvision
pytorch-lightning
numpy
pillow
tqdm
```

## 注意事项

1. 本项目提供了两个版本的LDM实现：
   - 完整版：使用UNet和交叉注意力
   - 简化版：使用MLP处理潜在向量

2. 建议使用GPU进行训练，典型训练时间：
   - VAE：约2小时（10个epoch）
   - LDM：约6小时（15个epoch）

3. 生成的样本和评估结果保存在：
   - 生成样本：`generated_samples/`
   - 网格图：`grid_samples.png`
   - 对比图：`comparison.png`

## 未来改进

1. 实现更复杂的条件生成策略
2. 优化采样速度
3. 改进FID分数
4. 添加更多数据增强方法

## 引用

如果您使用了本项目的代码，请引用：

```bibtex
@misc{cifar100-ldm,
  author = {Your Name},
  title = {CIFAR-100 Latent Diffusion Model},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/VAE}
}
``` 
