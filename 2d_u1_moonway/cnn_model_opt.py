# %%
import torch
import torch.nn as nn

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

class jointCNN(nn.Module):
    def __init__(self, plaq_input_channels=2, rect_input_channels=4, plaq_output_channels=4, rect_output_channels=8, kernel_size=(3, 3), num_res_blocks=2):
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
