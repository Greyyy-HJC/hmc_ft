# %%
import torch
import torch.nn as nn

class jointCNN(nn.Module):
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
