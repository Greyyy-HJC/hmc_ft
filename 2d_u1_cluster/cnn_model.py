# %%
import torch
import torch.nn as nn

class plaqCNN(nn.Module):
    def __init__(self, input_channels=2, output_channels=4, kernel_size=(3, 3)):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, 
            output_channels, 
            kernel_size, 
            padding='same',  # Use 'same' padding to maintain input size
            padding_mode='circular'  # Use circular padding to maintain periodic boundary conditions
        )
        self.activation = nn.GELU()

    def forward(self, x):
        # input shape: [batch_size, 2, L, L]
        x = self.conv(x)
        x = self.activation(x)
        x = torch.arctan(x) / torch.pi / 2 # range [-1/4, 1/4]
        # output shape: [batch_size, 4, L, L]
        return x 
    

class rectCNN(nn.Module):
    def __init__(self, input_channels=4, output_channels=8, kernel_size=(3, 3)):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, 
            output_channels, 
            kernel_size, 
            padding='same',  # Use 'same' padding to maintain input size
            padding_mode='circular'  # Use circular padding to maintain periodic boundary conditions
        )
        self.activation = nn.GELU()

    def forward(self, x):
        # input shape: [batch_size, 4, L, L]
        x = self.conv(x)
        x = self.activation(x)
        x = torch.arctan(x) / torch.pi / 2 # range [-1/4, 1/4]
        # output shape: [batch_size, 8, L, L]
        return x 


class combineCNN(nn.Module):
    def __init__(self, plaq_channels=4, rect_channels=8, output_channels=12, kernel_size=(3, 3)):
        super().__init__()
        # Combined input channels for plaq and rect features
        combined_input_channels = plaq_channels + rect_channels
        
        self.conv = nn.Conv2d(
            combined_input_channels, 
            output_channels, 
            kernel_size, 
            padding='same',
            padding_mode='circular'
        )
        self.activation = nn.GELU()

    def forward(self, plaq_features, rect_features):
        # plaq_features shape: [batch_size, plaq_channels, L, L]
        # rect_features shape: [batch_size, rect_channels, L, L]
        
        # Combine features
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        x = self.conv(x)
        x = self.activation(x)
        x = torch.arctan(x) / torch.pi / 2  # range [-1/4, 1/4]
        
        # Output first 4 channels as plaq coefficients, remaining 8 channels as rect coefficients
        plaq_coeffs = x[:, :4, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, 4:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs

