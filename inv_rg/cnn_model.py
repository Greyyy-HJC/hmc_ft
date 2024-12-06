# %%
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Simplified CNN Model based on the PDF description
class RGTransformerCNN(nn.Module):
    def __init__(self):
        super(RGTransformerCNN, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(2, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),  # Add BatchNorm
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),   # Add BatchNorm
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Data preparation
def read_data():
    conf_rg_list = torch.load("configs/2DU1_L32_RG.pt")
    conf_target_list = torch.load("configs/2DU1_L64.pt")
    conf_rg = torch.stack(conf_rg_list)
    conf_target = torch.stack(conf_target_list)
    
    # Split into train/test
    train_conf_rg = conf_rg[:50]
    train_conf_target = conf_target[:50]
    test_conf_rg = conf_rg[50:]
    test_conf_target = conf_target[50:]
    
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
                conf_ori_batch = model(conf_rg_batch)
                loss = criterion(conf_ori_batch, conf_target_batch)
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
            torch.save(model.state_dict(), 'best_model.pt')
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
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
            loss = criterion(conf_ori, single_target).item()
            test_losses.append(loss)
            print(f"Test Config {i+1} Loss: {loss:.6f}")
    
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
    train_loader = DataLoader(TensorDataset(train_conf_rg, train_conf_target), batch_size=50, shuffle=True)
    
    # Initialize model, loss, optimizer, scheduler
    model = RGTransformerCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)
    
    # Train model
    loss_history = train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=50)
    
    # Test model
    test_model(model, test_conf_rg, test_conf_target, criterion, device)

# %%
# Test with random matrix
random_matrix = torch.randn(1, 2, 64, 64).to(device)
target_config = test_conf_target[0:1].to(device)

model.eval()
with torch.no_grad():
    random_loss = criterion(random_matrix, target_config).item()
    print(f"\nLoss with random input: {random_loss:.6f}")


# %%
