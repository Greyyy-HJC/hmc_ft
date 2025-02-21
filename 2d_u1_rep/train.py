# %%
import torch
import numpy as np
from cnn_model import FieldTransformation


# Parameters
lattice_size = 32

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Set default type
torch.set_default_dtype(torch.float32)

# %%
# initialize the field transformation
nn_ft = FieldTransformation(lattice_size, device=device, n_subsets=8, if_check_jac=False) #todo

for train_beta in range(3, 8):
    # load the data
    data = np.load(f'dump/theta_ori_L{lattice_size}_beta{train_beta}.npy')
    tensor_data = torch.from_numpy(data).float().to(device)
    print(f"Loaded data shape: {tensor_data.shape}")

    # split the data into training and testing
    train_size = int(0.8 * len(tensor_data))
    train_data = tensor_data[:train_size]
    test_data = tensor_data[train_size:]
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")

    # train the model
    print("\n>>> Training the model at beta = ", train_beta)
    nn_ft.train(train_data, test_data, train_beta, n_epochs=16, batch_size=32)



