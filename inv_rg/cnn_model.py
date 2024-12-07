# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from utils import regularize, topo_from_field
from lametlat.utils.plot_settings import *
    
    
# Simplified CNN Model
class RGTransformerCNN(nn.Module):
    def __init__(self):
        super(RGTransformerCNN, self).__init__()
        self.model = nn.Sequential(
            # Initial upsampling
            nn.ConvTranspose2d(2, 128, kernel_size=2, stride=2),  # Upsampling
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Additional convolutional block
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Downscaling feature dimension
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # Another convolutional block for more features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # Final output
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)  # Output layer
        )

    def forward(self, x):
        return self.model(x)
    


# Alternative CNN Model with residual connections and different structure
class RGTransformerCNNAlt(nn.Module):
    def __init__(self):
        super(RGTransformerCNNAlt, self).__init__()
        
        # Initial upsampling
        self.upsample = nn.ConvTranspose2d(2, 64, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        
        # First residual block
        self.conv1a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv1b = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Second residual block
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Final output layers
        self.conv_final = nn.Conv2d(64, 32, kernel_size=1)
        self.bn_final = nn.BatchNorm2d(32)
        self.output = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Initial upsampling
        x = self.activation(self.bn1(self.upsample(x)))
        
        # First residual block
        identity = x
        x = self.activation(self.bn2(self.conv1a(x)))
        x = self.activation(self.bn3(self.conv1b(x)))
        x = x + identity
        
        # Second residual block  
        identity = x
        x = self.activation(self.bn4(self.conv2a(x)))
        x = self.activation(self.bn5(self.conv2b(x)))
        x = x + identity
        
        # Final output
        x = self.activation(self.bn_final(self.conv_final(x)))
        x = self.output(x)
        
        return x






# Data preparation
def read_data():
    conf_rg_list = torch.load("configs/2DU1_L32_RG.pt")
    conf_target_list = torch.load("configs/2DU1_L64.pt")
    conf_rg = torch.stack(conf_rg_list)
    conf_target = torch.stack(conf_target_list)
    
    # Split into train/test
    train_conf_rg = conf_rg[:500]
    train_conf_target = conf_target[:500]
    test_conf_rg = conf_rg[500:]
    test_conf_target = conf_target[500:]
    
    return train_conf_rg, train_conf_target, test_conf_rg, test_conf_target

# Training function
def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=50):
    scaler = torch.cuda.amp.GradScaler()  # Enable mixed precision
    best_loss = float('inf')
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for conf_rg_batch, conf_target_batch in train_loader:
            conf_rg_batch, conf_target_batch = conf_rg_batch.to(device), conf_target_batch.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Mixed precision context
                conf_irg_batch = model(conf_rg_batch)
                # Regularize each batch
                conf_irg_batch = regularize(conf_irg_batch)
                loss = criterion(conf_irg_batch, conf_target_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'models/best_model.pt')
    
    fig, ax = default_plot()
    ax.plot(loss_history, 'b-')
    ax.set_title('Training Loss Over Time', **fs_p)
    ax.set_xlabel('Epoch', **fs_p)
    ax.set_ylabel('Loss', **fs_p)
    plt.tight_layout()
    plt.savefig('plots/cnn_loss.pdf', transparent=True)
    plt.show()
    
    return loss_history

# Testing function
def test_model(model, test_conf_rg, test_conf_target, criterion, device):
    model.eval()
    test_losses = []
    with torch.no_grad():
        for i in range(len(test_conf_rg)):
            single_conf = test_conf_rg[i:i+1].to(device)
            single_target = test_conf_target[i:i+1].to(device)
            conf_ori = model(single_conf)
            conf_ori = regularize(conf_ori)
            loss = criterion(conf_ori[0], single_target[0]).item()
            test_losses.append(loss)
    
    test_losses = torch.tensor(test_losses)
    print("\nTest Statistics:")
    print(f"Mean Loss: {test_losses.mean():.6f}")
    print(f"Std Loss: {test_losses.std():.6f}")
    print(f"Min Loss: {test_losses.min():.6f}")
    print(f"Max Loss: {test_losses.max():.6f}")

# Main script
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Read data
    train_conf_rg, train_conf_target, test_conf_rg, test_conf_target = read_data()
    train_loader = DataLoader(
        TensorDataset(train_conf_rg, train_conf_target), 
        batch_size=50,  # Smaller batch size
        shuffle=True
    )
    
    def topology_loss(output, target, lambda_weight=0.1):
        mse_loss = nn.MSELoss()(output, target)
        topo_output = topo_from_field(output)  
        topo_target = topo_from_field(target) 
        topo_diff = torch.abs(topo_output - topo_target).mean()
        return mse_loss + lambda_weight * topo_diff
    
    # Initialize model
    # model = RGTransformerCNN().to(device) #todo
    model = RGTransformerCNNAlt().to(device)
    # criterion = nn.MSELoss()
    criterion = topology_loss
    
    # Use AdamW with better parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=2e-5,  # Lower initial learning rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,  # Smaller reduction factor
        patience=10,  # Longer patience
    )
    
    # Train model
    loss_history = train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=100)
    
    # Test model
    test_model(model, test_conf_rg, test_conf_target, criterion, device)

# %%
if __name__ == "__main__":
    # Test with random matrix
    random_matrix = torch.randn(1, 2, 64, 64).to(device)
    target_config = test_conf_target[0:1].to(device)

    model.eval()
    with torch.no_grad():
        random_loss = criterion(random_matrix, target_config).item()
        print(f"\nLoss with random input: {random_loss:.6f}")


# %%
