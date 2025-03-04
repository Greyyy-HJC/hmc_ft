import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from hmc_u1 import action
from utils import plaq_mean_from_field, topo_from_field
import math
import os
from lcnn import LCNN, LCNNRegressor

# --------------------
# GAN 结构定义
# --------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, lattice_size, beta, use_lcnn=True):
        super(Generator, self).__init__()
        self.lattice_size = lattice_size
        self.beta = beta  # 存储beta参数
        self.use_lcnn = use_lcnn  # 控制是否使用LCNN
        
        # 初始全连接层
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 2 * lattice_size * lattice_size)
        )
        
        # L-CNN层用于确保规范等变性
        if use_lcnn:
            self.lcnn = LCNN(
                lattice_size=lattice_size,
                in_channels=1,
                hidden_channels=16,
                out_channels=1
            )
        else:
            # 常规CNN层，用于替代LCNN
            self.conv = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 2, kernel_size=3, padding=1),
                nn.Tanh()
            )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        # 生成初始场构型
        x = self.fc(z)
        x = x.view(-1, 2, self.lattice_size, self.lattice_size)
        
        if self.use_lcnn:
            # 应用L-CNN确保规范等变性
            x = self.lcnn(x)
        else:
            # 应用常规CNN
            x = self.conv(x)
        
        # 确保输出在[-π, π]范围内
        x = torch.tanh(x) * math.pi
        
        return x


class Discriminator(nn.Module):
    def __init__(self, lattice_size, use_lcnn=True):
        super(Discriminator, self).__init__()
        self.use_lcnn = use_lcnn
        
        if use_lcnn:
            # L-CNN特征提取器
            self.lcnn = LCNN(
                lattice_size=lattice_size,
                in_channels=1,
                hidden_channels=16,
                out_channels=4
            )
            # 判别器头部
            input_size = 4 * 2 * lattice_size * lattice_size  # 4个通道，2个方向
        else:
            # 常规CNN特征提取器
            self.cnn = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
            # 判别器头部
            input_size = 64 * lattice_size * lattice_size
        
        self.discriminator_head = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.use_lcnn:
            # 提取规范等变特征
            features = self.lcnn(x)
            
            # 展平特征
            features = features.view(features.size(0), -1)
        else:
            # 使用常规CNN提取特征
            features = self.cnn(x)
            
            # 展平特征
            features = features.view(features.size(0), -1)
        
        # 判别器输出
        return self.discriminator_head(features)


# --------------------
# 训练 GAN
# --------------------
def train_gan(generator, discriminator, dataloader, latent_dim, beta, epochs=100, device='cpu', output_dir=None, use_lcnn=True):
    """
    训练带有L-CNN的GAN，使用规范等变特征和物理约束。
    
    参数:
    generator: 生成器模型
    discriminator: 判别器模型
    dataloader: 数据加载器
    latent_dim: 潜在空间维度
    beta: 反耦合常数
    epochs: 训练轮数
    device: 计算设备
    output_dir: 输出目录
    use_lcnn: 是否使用LCNN（若为False则使用常规CNN）
    """
    # 使用二元交叉熵损失
    criterion = nn.BCELoss()
    
    # 创建拓扑电荷预测器
    if use_lcnn:
        # 使用L-CNN回归器用于拓扑电荷预测
        topo_regressor = LCNNRegressor(generator.lattice_size).to(device)
    else:
        # 使用常规CNN回归器
        topo_regressor = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        ).to(device)
    
    # 设置学习率
    lr_G = 0.0002
    lr_D = 0.0002
    
    # 使用Adam优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))
    optimizer_T = optim.Adam(topo_regressor.parameters(), lr=0.001)
    
    # 设置为训练模式
    generator.train()
    discriminator.train()
    topo_regressor.train()
    
    # 用于记录每个epoch的loss
    epoch_losses_G = []
    epoch_losses_D = []
    epoch_losses_T = []
    
    # 记录最佳模型
    best_g_loss = float('inf')
    best_d_loss = float('inf')
    
    print(f"\n开始GAN训练... (使用{'LCNN' if use_lcnn else '常规CNN'})")
    for epoch in range(epochs):
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_T = 0.0
        batch_count = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            real_samples = batch[0].to(device)
            batch_size = real_samples.shape[0]
            batch_count += 1
            
            # 计算真实样本的拓扑电荷
            real_topo = topo_from_field(real_samples)
            
            # 标签平滑化
            real_labels = 0.9 + 0.1 * torch.rand(batch_size, 1, device=device)
            fake_labels = 0.0 + 0.1 * torch.rand(batch_size, 1, device=device)
            
            # ----------------------
            # 训练拓扑电荷回归器
            # ----------------------
            optimizer_T.zero_grad()
            pred_topo = topo_regressor(real_samples)
            loss_T = F.mse_loss(pred_topo, real_topo)
            loss_T.backward()
            optimizer_T.step()
            
            # ----------------------
            # 训练判别器
            # ----------------------
            optimizer_D.zero_grad()
            
            # 真实样本的损失
            d_real = discriminator(real_samples)
            loss_real = criterion(d_real, real_labels)
            
            # 生成假样本
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_samples = generator(z)
            
            # 假样本的损失
            d_fake = discriminator(fake_samples.detach())
            loss_fake = criterion(d_fake, fake_labels)
            
            # 判别器总损失
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            
            # ----------------------
            # 训练生成器
            # ----------------------
            optimizer_G.zero_grad()
            
            # 生成新的假样本
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_samples = generator(z)
            
            # 判别器对假样本的预测
            d_fake = discriminator(fake_samples)
            
            # 计算生成样本的拓扑电荷
            fake_topo = topo_regressor(fake_samples)
            
            # 生成器的对抗损失
            adv_loss = criterion(d_fake, torch.ones(batch_size, 1, device=device))
            
            # 物理约束损失
            action_loss = torch.mean((action(fake_samples, beta) - action(real_samples, beta))**2)
            topo_loss = F.mse_loss(fake_topo, real_topo)
            
            # 总损失
            loss_G = adv_loss + 0.1 * action_loss + 0.1 * topo_loss
            loss_G.backward()
            optimizer_G.step()
            
            # 记录损失
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            running_loss_T += loss_T.item()
            
            # 每10个批次打印一次当前loss
            if batch_count % 10 == 0:
                print(f"\n  Batch {batch_count}: G_loss: {loss_G.item():.4f}, D_loss: {loss_D.item():.4f}, T_loss: {loss_T.item():.4f}")
        
        # 计算并存储当前epoch的平均loss
        avg_loss_G = running_loss_G / batch_count
        avg_loss_D = running_loss_D / batch_count
        avg_loss_T = running_loss_T / batch_count
        
        epoch_losses_G.append(avg_loss_G)
        epoch_losses_D.append(avg_loss_D)
        epoch_losses_T.append(avg_loss_T)
        
        print(f"\nEpoch {epoch+1}/{epochs} - Avg G_loss: {avg_loss_G:.4f}, Avg D_loss: {avg_loss_D:.4f}, Avg T_loss: {avg_loss_T:.4f}")
        
        # 保存最佳模型
        if output_dir:
            if avg_loss_G < best_g_loss:
                best_g_loss = avg_loss_G
                torch.save(generator.state_dict(), f"{output_dir}/best_generator.pt")
                print(f"\n保存最佳生成器模型，损失: {best_g_loss:.4f}")
            
            if avg_loss_D < best_d_loss:
                best_d_loss = avg_loss_D
                torch.save(discriminator.state_dict(), f"{output_dir}/best_discriminator.pt")
                print(f"\n保存最佳判别器模型，损失: {best_d_loss:.4f}")
    
    # 保存最终模型
    if output_dir:
        torch.save(generator.state_dict(), f"{output_dir}/final_generator.pt")
        torch.save(discriminator.state_dict(), f"{output_dir}/final_discriminator.pt")
        torch.save(topo_regressor.state_dict(), f"{output_dir}/final_topo_regressor.pt")
    
    # 绘制loss曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs+1), epoch_losses_G, label='Generator Loss')
        plt.plot(range(1, epochs+1), epoch_losses_D, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs+1), epoch_losses_T, label='Topology Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Topology Regression Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/training_loss.png")
            print(f"\nLoss曲线已保存")
            
            # 保存loss数据
            import numpy as np
            np.save(f"{output_dir}/gan_losses_G.npy", np.array(epoch_losses_G))
            np.save(f"{output_dir}/gan_losses_D.npy", np.array(epoch_losses_D))
            np.save(f"{output_dir}/gan_losses_T.npy", np.array(epoch_losses_T))
            
    except Exception as e:
        print(f"\n绘制loss曲线时出错: {e}")
    
    # 如果有保存最佳生成器，则加载它
    if output_dir and os.path.exists(f"{output_dir}/best_generator.pt"):
        generator.load_state_dict(torch.load(f"{output_dir}/best_generator.pt"))
        print("\n已加载最佳生成器模型用于返回")
    
    return generator


# --------------------
# GAN 过松弛更新
# --------------------
def gan_overrelaxation(generator, theta, beta, latent_dim, threshold=0.5, max_iters=100, device='cpu', n_attempts=5, use_lcnn=True):
    """
    使用带有L-CNN的生成器进行过松弛更新，确保生成的配置满足规范对称性和拓扑约束。
    
    参数:
    generator: 生成器模型
    theta: 当前场构型
    beta: 反耦合常数
    latent_dim: 潜在空间维度
    threshold: 接受阈值
    max_iters: 最大迭代次数
    device: 计算设备
    n_attempts: 尝试次数
    use_lcnn: 是否使用LCNN（若为False则使用常规CNN）
    """
    generator.eval()  # 设置为评估模式
    
    # 计算当前构型的作用量和拓扑电荷
    target_S = action(theta, beta).item()
    # 对单个样本，topo_from_field现在返回形状为[1, 1]的张量
    target_Q = topo_from_field(theta.unsqueeze(0)).item() if theta.dim() == 3 else topo_from_field(theta).item()
    print(f"\n目标作用量: {target_S:.4f}, 目标拓扑电荷: {target_Q:.4f} (使用{'LCNN' if use_lcnn else '常规CNN'})")
    
    best_theta = theta
    best_S_diff = float('inf')
    best_Q_diff = float('inf')
    
    # 首先尝试直接采样方法
    n_direct_samples = 50
    z_samples = torch.randn(n_direct_samples, latent_dim, device=device)
    with torch.no_grad():
        gen_samples = generator(z_samples)
    
    # 计算每个样本的作用量和拓扑电荷，找到最接近的
    for i in range(n_direct_samples):
        sample = gen_samples[i]
        S_sample = action(sample, beta).item()
        # 对单个样本，需要先扩展维度
        Q_sample = topo_from_field(sample.unsqueeze(0)).item()
        
        S_diff = abs(S_sample - target_S)
        Q_diff = abs(Q_sample - target_Q)
        
        # 使用加权和来评估样本质量
        total_diff = S_diff + 0.5 * Q_diff
        
        if total_diff < best_S_diff + 0.5 * best_Q_diff:
            best_S_diff = S_diff
            best_Q_diff = Q_diff
            best_theta = sample
    
    # 如果直接采样找到了足够好的样本，直接返回
    if best_S_diff < threshold and best_Q_diff < threshold:
        print(f"\n直接采样找到合适构型，作用量差: {best_S_diff:.6f}, 拓扑电荷差: {best_Q_diff:.6f}")
        return best_theta
    
    # 否则使用梯度流优化方法
    for attempt in range(n_attempts):
        z = torch.randn(1, latent_dim, device=device, requires_grad=True)
        optimizer = optim.Adam([z], lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)
        
        patience = 20
        best_loss = float('inf')
        patience_counter = 0
        best_z = z.clone()
        
        for i in range(max_iters):
            # 生成新构型
            new_theta = generator(z).squeeze(0)
            
            # 计算作用量和拓扑电荷
            S_new = action(new_theta, beta)
            # 对单个样本，需要先扩展维度
            Q_new = topo_from_field(new_theta.unsqueeze(0)).squeeze()
            
            # 计算损失（加权和）
            loss = (S_new - target_S)**2 + 0.5 * (Q_new - target_Q)**2
            
            if loss.item() < threshold**2:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_z = z.clone()
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z.clone()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # 使用最佳z生成最终结果
        with torch.no_grad():
            final_theta = generator(best_z).squeeze(0)
            final_S = action(final_theta, beta).item()
            # 对单个样本，需要先扩展维度
            final_Q = topo_from_field(final_theta.unsqueeze(0)).item()
            
            S_diff = abs(final_S - target_S)
            Q_diff = abs(final_Q - target_Q)
            
            if S_diff < best_S_diff and Q_diff < best_Q_diff:
                best_S_diff = S_diff
                best_Q_diff = Q_diff
                best_theta = final_theta
                print(f"\n尝试 {attempt+1}: 找到更好的构型，作用量差: {S_diff:.6f}, 拓扑电荷差: {Q_diff:.6f}")
    
    print(f"\n最终构型 - 作用量差: {best_S_diff:.6f}, 拓扑电荷差: {best_Q_diff:.6f}")
    return best_theta


# --------------------
# 结合 GAN 的 HMC 采样
# --------------------
def run_hmc_with_gan(hmc, generator, latent_dim, n_iterations, theta, store_interval=1, nH=1, device='cpu', gan_attempts=3, gan_threshold=0.5, gan_frequency=5, use_lcnn=True):
    """
    结合带有L-CNN的GAN过松弛的HMC采样方法。
    
    参数:
    hmc: HMC采样器
    generator: 生成器模型
    latent_dim: 潜在空间维度
    n_iterations: 迭代次数
    theta: 初始场构型
    store_interval: 存储间隔
    nH: 每次迭代的HMC步数
    device: 计算设备
    gan_attempts: GAN过松弛尝试次数
    gan_threshold: GAN过松弛接受阈值
    gan_frequency: GAN过松弛应用频率
    use_lcnn: 是否使用LCNN（若为False则使用常规CNN）
    """
    generator.to(device)
    theta_ls = []
    plaq_ls = []
    hamiltonians = []
    topological_charges = []
    acceptance_count = 0
    
    for i in tqdm(range(n_iterations), desc=f"Running HMC with GAN (使用{'LCNN' if use_lcnn else '常规CNN'})"):
        # 标准HMC更新
        for _ in range(nH):
            theta, accepted, H_val = hmc.metropolis_step(theta)
            if accepted:
                acceptance_count += 1
        
        # 每gan_frequency次迭代使用一次GAN过松弛
        if i % gan_frequency == 0:
            print(f"\n\nIteration {i}: Applying GAN overrelaxation (使用{'LCNN' if use_lcnn else '常规CNN'})")
            theta_before_gan = theta.clone()
            
            # 应用GAN过松弛
            theta_gan = gan_overrelaxation(
                generator=generator, 
                theta=theta, 
                beta=hmc.beta, 
                latent_dim=latent_dim, 
                threshold=gan_threshold, 
                device=device, 
                n_attempts=gan_attempts,
                use_lcnn=use_lcnn
            )
            
            # 计算GAN过松弛前后的作用量和拓扑电荷差异
            S_before = action(theta, hmc.beta).item()
            S_after = action(theta_gan, hmc.beta).item()
            # 对单个样本，需要先扩展维度
            Q_before = topo_from_field(theta.unsqueeze(0)).item()
            Q_after = topo_from_field(theta_gan.unsqueeze(0)).item()
            
            S_diff = abs(S_after - S_before)
            Q_diff = abs(Q_after - Q_before)
            
            # 如果作用量和拓扑电荷差异都在阈值内，接受GAN过松弛的结果
            if S_diff <= gan_threshold and Q_diff <= gan_threshold:
                theta = theta_gan
                print(f"\nGAN overrelaxation accepted: Action diff = {S_diff:.6f}, Topo diff = {Q_diff:.6f}")
            else:
                print(f"\nGAN overrelaxation rejected: Action diff = {S_diff:.6f}, Topo diff = {Q_diff:.6f}")
        
        # 记录数据
        if i % store_interval == 0:
            theta_ls.append(theta.clone())
            plaq_ls.append(plaq_mean_from_field(theta).item())
            hamiltonians.append(H_val)
            # 对单个样本，需要先扩展维度
            topological_charges.append(topo_from_field(theta.unsqueeze(0)).item())
    
    acceptance_rate = acceptance_count / (n_iterations * nH)
    return theta_ls, plaq_ls, acceptance_rate, topological_charges, hamiltonians
