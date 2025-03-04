import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import plaq_from_field, topo_from_field, regularize

class LConv(nn.Module):
    """
    L-Convolution层：执行规范等变变换，使用Wilson环作为基本构建块。
    
    在U(1)格点规范理论中，这个层通过卷积操作处理规范链接变量，
    并确保输出保持规范等变性。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 创建可学习的权重，用于Wilson环的组合
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # 初始化权重
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        """
        输入x的形状: [batch_size, in_channels, 2, lattice_size, lattice_size]
        其中2表示两个方向的规范链接
        
        输出形状: [batch_size, out_channels, 2, lattice_size, lattice_size]
        """
        batch_size, in_channels, dirs, h, w = x.shape
        
        # 将角度转换为复数表示
        x_complex = torch.exp(1j * x)
        
        # 分离实部和虚部
        x_real = x_complex.real
        x_imag = x_complex.imag
        
        # 分别处理每个方向
        out_real_list = []
        out_imag_list = []
        
        for d in range(dirs):
            # 对每个方向分别进行卷积
            real_d = F.conv2d(
                x_real[:, :, d],  # [batch_size, in_channels, h, w]
                self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding
            )
            
            imag_d = F.conv2d(
                x_imag[:, :, d],
                self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding
            )
            
            out_real_list.append(real_d)
            out_imag_list.append(imag_d)
        
        # 堆叠方向维度
        out_real = torch.stack(out_real_list, dim=2)  # [batch_size, out_channels, dirs, h, w]
        out_imag = torch.stack(out_imag_list, dim=2)
        
        # 重建复数
        out_complex = torch.complex(out_real, out_imag)
        
        # 转换回角度表示
        out = torch.angle(out_complex)
        
        # 添加偏置
        out = out + self.bias.view(1, self.out_channels, 1, 1, 1)
        
        return out


class LBilinear(nn.Module):
    """
    L-Bilinear层：形成更高阶的规范协变可观测量。
    
    这个层组合两个规范链接场，形成更复杂的Wilson环和其他规范不变量。
    """
    def __init__(self, in_channels, out_channels):
        super(LBilinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 创建可学习的权重矩阵
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, in_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # 初始化权重
        nn.init.xavier_normal_(self.weight)
    
    def forward(self, x):
        """
        输入x的形状: [batch_size, in_channels, 2, lattice_size, lattice_size]
        输出形状: [batch_size, out_channels, 2, lattice_size, lattice_size]
        """
        batch_size, in_channels, dirs, h, w = x.shape
        
        # 将角度转换为复数表示
        x_complex = torch.exp(1j * x)
        
        # 分别处理每个方向
        out_list = []
        
        for d in range(dirs):
            # 提取当前方向的数据
            x_d = x_complex[:, :, d]  # [batch_size, in_channels, h, w]
            
            # 重塑以进行双线性操作
            x_flat = x_d.reshape(batch_size, in_channels, h * w)  # [batch_size, in_channels, h*w]
            
            # 为每个输出通道执行双线性操作
            out_channels_d = []
            for i in range(self.out_channels):
                # 获取当前输出通道的权重并转换为复数
                bilinear_weights = self.weight[i].to(torch.complex64)  # [in_channels, in_channels]
                
                # 使用矩阵乘法代替einsum，避免类型不匹配问题
                # 计算 W * x
                wx = torch.matmul(bilinear_weights, x_flat)  # [batch_size, in_channels, h*w]
                
                # 计算 x^T * (W * x)
                # 首先转置x_flat: [batch_size, h*w, in_channels]
                x_flat_t = x_flat.transpose(1, 2)
                
                # 然后计算矩阵乘法: [batch_size, h*w, in_channels] x [batch_size, in_channels, h*w]
                out_i = torch.matmul(x_flat_t, wx)  # [batch_size, h*w, h*w]
                
                # 提取对角线元素（我们只关心自相关）
                out_i = torch.diagonal(out_i, dim1=1, dim2=2)  # [batch_size, h*w]
                
                # 重塑回原始空间维度
                out_i = out_i.reshape(batch_size, h, w)
                out_channels_d.append(out_i)
            
            # 堆叠所有输出通道
            out_d = torch.stack(out_channels_d, dim=1)  # [batch_size, out_channels, h, w]
            out_list.append(out_d)
        
        # 堆叠所有方向
        out_complex = torch.stack(out_list, dim=2)  # [batch_size, out_channels, dirs, h, w]
        
        # 转换回角度表示
        out = torch.angle(out_complex)
        
        # 添加偏置
        out = out + self.bias.view(1, self.out_channels, 1, 1, 1)
        
        return out


class LActivation(nn.Module):
    """
    L-Activation层：应用规范不变的激活函数。
    
    这个层确保激活函数不破坏规范等变性。
    """
    def __init__(self, activation='relu'):
        super(LActivation, self).__init__()
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = lambda x: F.leaky_relu(x, negative_slope=0.2)
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        """
        输入和输出形状相同: [batch_size, channels, 2, lattice_size, lattice_size]
        
        对角度值应用激活函数，但保持规范等变性
        """
        # 将角度转换为复数
        x_complex = torch.exp(1j * x)
        
        # 计算幅度（规范不变量）
        magnitude = torch.abs(x_complex)
        
        # 对幅度应用激活函数
        activated_magnitude = self.activation(magnitude)
        
        # 保持相位不变，调整幅度
        phase = torch.angle(x_complex)
        new_complex = activated_magnitude * torch.exp(1j * phase)
        
        # 转换回角度表示
        return torch.angle(new_complex)


class LExp(nn.Module):
    """
    L-Exponentiation层：更新规范链接，同时保持SU(N)约束。
    
    在U(1)情况下，这个层确保输出角度在[-π, π]范围内。
    """
    def __init__(self):
        super(LExp, self).__init__()
    
    def forward(self, x):
        """
        输入和输出形状相同: [batch_size, channels, 2, lattice_size, lattice_size]
        
        确保输出是有效的U(1)规范场配置
        """
        # 规范化角度到[-π, π]范围
        return regularize(x)


class TraceLayer(nn.Module):
    """
    Trace层：计算规范不变的可观测量。
    
    这个层用于验证和监控生成的配置。
    """
    def __init__(self, lattice_size):
        super(TraceLayer, self).__init__()
        self.lattice_size = lattice_size
    
    def forward(self, x):
        """
        输入形状: [batch_size, channels, 2, lattice_size, lattice_size]
        输出形状: [batch_size, channels]
        
        计算每个通道的平均plaquette值作为规范不变量
        """
        batch_size, channels, _, _, _ = x.shape
        plaq_values = torch.zeros(batch_size, channels, device=x.device)
        
        for b in range(batch_size):
            for c in range(channels):
                # 提取单个配置
                config = x[b, c]
                
                # 计算plaquette
                plaq = plaq_from_field(config)
                
                # 计算平均值
                plaq_values[b, c] = torch.mean(torch.cos(plaq))
        
        return plaq_values


class LCNN(nn.Module):
    """
    完整的L-CNN模型，用于处理格点规范场配置。
    """
    def __init__(self, lattice_size, in_channels=1, hidden_channels=16, out_channels=1):
        super(LCNN, self).__init__()
        
        self.lattice_size = lattice_size
        
        # L-CNN层序列
        self.layers = nn.Sequential(
            LConv(in_channels, hidden_channels, kernel_size=3, padding=1),
            LActivation('relu'),
            LConv(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            LActivation('relu'),
            LBilinear(hidden_channels, hidden_channels // 2),
            LActivation('relu'),
            LConv(hidden_channels // 2, out_channels, kernel_size=3, padding=1),
            LExp()
        )
        
        # 用于验证的Trace层
        self.trace = TraceLayer(lattice_size)
    
    def forward(self, x):
        """
        输入形状: [batch_size, 2, lattice_size, lattice_size]
        输出形状: [batch_size, 2, lattice_size, lattice_size]
        """
        # 添加通道维度
        x = x.unsqueeze(1)
        
        # 应用L-CNN层
        x = self.layers(x)
        
        # 移除通道维度
        x = x.squeeze(1)
        
        return x
    
    def compute_observables(self, x):
        """
        计算物理可观测量，用于验证生成的配置。
        
        输入形状: [batch_size, 2, lattice_size, lattice_size]
        输出: 字典，包含平均plaquette和拓扑电荷
        """
        batch_size = x.shape[0]
        plaq_values = torch.zeros(batch_size, device=x.device)
        topo_charges = torch.zeros(batch_size, device=x.device)
        
        for b in range(batch_size):
            config = x[b]
            plaq_values[b] = torch.mean(torch.cos(plaq_from_field(config)))
            topo_charges[b] = topo_from_field(config)
        
        return {
            'plaquette': plaq_values,
            'topological_charge': topo_charges
        }


class LCNNRegressor(nn.Module):
    """
    L-CNN回归模型，用于预测拓扑电荷密度。
    """
    def __init__(self, lattice_size, hidden_channels=16):
        super(LCNNRegressor, self).__init__()
        
        self.lattice_size = lattice_size
        self.hidden_channels = hidden_channels
        
        # L-CNN特征提取器
        self.feature_extractor = nn.Sequential(
            LConv(1, hidden_channels, kernel_size=3, padding=1),
            LActivation('relu'),
            LConv(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            LActivation('relu'),
            LBilinear(hidden_channels, hidden_channels // 2),
            LActivation('relu')
        )
        
        # 回归头 - 直接使用线性层，不使用AdaptiveAvgPool3d
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_channels // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        输入形状: [batch_size, 2, lattice_size, lattice_size]
        输出: 预测的拓扑电荷 [batch_size, 1]
        """
        # 添加通道维度
        x = x.unsqueeze(1)
        
        # 提取特征
        features = self.feature_extractor(x)
        
        # 使用全局平均池化来减少空间维度
        # 对每个通道和方向进行平均池化
        pooled_features = torch.mean(features, dim=(3, 4))  # [batch_size, hidden_channels//2, 2]
        
        # 再次平均方向维度
        pooled_features = torch.mean(pooled_features, dim=2)  # [batch_size, hidden_channels//2]
        
        # 回归
        topo_charge = self.regression_head(pooled_features)
        
        return topo_charge 