# %%
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

class jointCNN_simple(nn.Module):
    def __init__(self, plaq_input_channels=2, rect_input_channels=4, plaq_output_channels=4, rect_output_channels=8, kernel_size=(3, 3)):
        super().__init__()
        # Combined input channels for plaq and rect features
        combined_input_channels = plaq_input_channels + rect_input_channels

        # First conv layer to process combined features
        self.conv1 = nn.Conv2d(
            combined_input_channels,
            combined_input_channels * 2,  # Double the channels
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation1 = nn.GELU()
        
        # Second conv layer to generate final outputs
        self.conv2 = nn.Conv2d(
            combined_input_channels * 2,
            plaq_output_channels + rect_output_channels,  # Combined output channels
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation2 = nn.GELU()

    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer
        x = self.conv1(x)
        x = self.activation1(x)
        
        # Second conv layer
        x = self.conv2(x)
        x = self.activation2(x)
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs

class ResBlock(nn.Module):
    """Residual block with two convolutional layers and a skip connection"""
    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, 
            channels, 
            kernel_size, 
            padding='same', 
            padding_mode='circular'
        )
        self.activation1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(
            channels, 
            channels, 
            kernel_size, 
            padding='same', 
            padding_mode='circular'
        )
        self.activation2 = nn.GELU()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.activation1(out)
        
        out = self.conv2(out)
        
        out += identity  # Skip connection
        out = self.activation2(out)
        
        return out
    
class ResBlock_norm(nn.Module):
    """Residual block with group norm, depthwise separable conv, and dropout"""
    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        
        # First norm + depthwise + pointwise conv
        self.norm1 = nn.GroupNorm(4, channels)
        self.depthwise1 = nn.Conv2d(
            channels, channels, kernel_size,
            padding='same', padding_mode='circular',
            groups=channels
        )
        self.pointwise1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation1 = nn.GELU()

        # Second norm + depthwise + pointwise conv
        self.norm2 = nn.GroupNorm(4, channels)
        self.depthwise2 = nn.Conv2d(
            channels, channels, kernel_size,
            padding='same', padding_mode='circular',
            groups=channels
        )
        self.pointwise2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation2 = nn.GELU()

    def forward(self, x):
        identity = x

        out = self.norm1(x)
        out = self.depthwise1(out)
        out = self.pointwise1(out)
        out = self.activation1(out)

        out = self.norm2(out)
        out = self.depthwise2(out)
        out = self.pointwise2(out)

        out = identity + out
        out = self.activation2(out)

        return out
    
class jointCNN_rnet1(nn.Module):
    def __init__(self, plaq_input_channels=2, rect_input_channels=4, plaq_output_channels=4, rect_output_channels=8, kernel_size=(3, 3), num_res_blocks=1):
        super().__init__()
        # Combined input channels for plaq and rect features
        combined_input_channels = plaq_input_channels + rect_input_channels
        intermediate_channels = combined_input_channels * 2  # Reduced from 4x to 2x
        
        # Initial convolution to increase channels
        self.initial_conv = nn.Conv2d(
            combined_input_channels,
            intermediate_channels,
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.initial_activation = nn.GELU()
        
        # ResNet blocks (reduced from 3 to 2 by default)
        self.res_blocks = nn.ModuleList([
            ResBlock(intermediate_channels, kernel_size)
            for _ in range(num_res_blocks)
        ])
        
        # Final convolution to get output channels
        self.final_conv = nn.Conv2d(
            intermediate_channels,
            plaq_output_channels + rect_output_channels,
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.final_activation = nn.GELU()

    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_activation(x)
        
        # Apply ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Final convolution
        x = self.final_conv(x)
        x = self.final_activation(x)
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs

class jointCNN_rnet3(nn.Module):
    def __init__(self, plaq_input_channels=2, rect_input_channels=4, plaq_output_channels=4, rect_output_channels=8, kernel_size=(3, 3), num_res_blocks=3):
        super().__init__()
        # Combined input channels for plaq and rect features
        combined_input_channels = plaq_input_channels + rect_input_channels
        intermediate_channels = combined_input_channels * 2  # Reduced from 4x to 2x
        
        # Initial convolution to increase channels
        self.initial_conv = nn.Conv2d(
            combined_input_channels,
            intermediate_channels,
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.initial_activation = nn.GELU()
        
        # ResNet blocks (reduced from 3 to 2 by default)
        self.res_blocks = nn.ModuleList([
            ResBlock(intermediate_channels, kernel_size)
            for _ in range(num_res_blocks)
        ])
        
        # Final convolution to get output channels
        self.final_conv = nn.Conv2d(
            intermediate_channels,
            plaq_output_channels + rect_output_channels,
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.final_activation = nn.GELU()

    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_activation(x)
        
        # Apply ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Final convolution
        x = self.final_conv(x)
        x = self.final_activation(x)
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs


class jointCNN_rnet_norm(nn.Module):
    def __init__(self, plaq_input_channels=2, rect_input_channels=4, plaq_output_channels=4, rect_output_channels=8, kernel_size=(3, 3), num_res_blocks=1, dropout_prob=0.05):
        super().__init__()
        # Combined input channels for plaq and rect features
        combined_input_channels = plaq_input_channels + rect_input_channels
        intermediate_channels = combined_input_channels * 2  # Reduced from 4x to 2x
        
        # Initial convolution to increase channels
        self.initial_conv = nn.Conv2d(
            combined_input_channels,
            intermediate_channels,
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.initial_activation = nn.GELU()
        
        # ResNet blocks (reduced from 3 to 2 by default)
        self.res_blocks = nn.ModuleList([
            ResBlock_norm(intermediate_channels, kernel_size)
            for _ in range(num_res_blocks)
        ])
        
        # Final convolution to get output channels
        self.final_conv = nn.Conv2d(
            intermediate_channels,
            plaq_output_channels + rect_output_channels,
            kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.final_activation = nn.GELU()
        
        # Single dropout at the end
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_activation(x)
        
        # Apply ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Final convolution
        x = self.final_conv(x)
        x = self.final_activation(x)
        x = self.dropout(x)
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs



def choose_cnn_model(model_tag):
    if model_tag == 'simple':
        return jointCNN_simple
    elif model_tag == 'rnet1':
        return jointCNN_rnet1
    elif model_tag == 'rnet3':
        return jointCNN_rnet3
    elif model_tag == 'rnet_norm':
        return jointCNN_rnet_norm
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")