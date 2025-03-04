# %%
import torch
import numpy as np
import argparse
from cnn_model import FieldTransformation

# Parse command line arguments
parser = argparse.ArgumentParser(description='Training parameters for Field Transformation')
parser.add_argument('--lattice_size', type=int, default=32,
                    help='Size of the lattice (default: 32)')
parser.add_argument('--min_beta', type=float, required=True,
                    help='Minimum beta value for training')
parser.add_argument('--max_beta', type=float, required=True,
                    help='Maximum beta value for training (exclusive)')
parser.add_argument('--n_epochs', type=int, default=16,
                    help='Number of training epochs (default: 16)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training (default: 32)')

args = parser.parse_args()

# Parameters
lattice_size = args.lattice_size

# Initialize device
device = 'cpu'

# Set default type
torch.set_default_dtype(torch.float32)

# %%
# initialize the field transformation
nn_ft = FieldTransformation(lattice_size, device=device, n_subsets=8, if_check_jac=False) #todo

for train_beta in range(int(args.min_beta), int(args.max_beta)):
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
    nn_ft.train(train_data, test_data, train_beta, n_epochs=args.n_epochs, batch_size=args.batch_size)



