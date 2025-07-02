# %%
import os
import torch
import numpy as np
import argparse
import time
import datetime
from utils import set_seed
from field_trans import FieldTransformation
from torch.nn.parallel import DataParallel

# Record program start time
start_time = time.time()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Training parameters for Field Transformation')
parser.add_argument('--lattice_size', type=int, default=32,
                    help='Size of the lattice (default: 32)')
parser.add_argument('--min_beta', type=float, required=True,
                    help='Minimum beta value for training')
parser.add_argument('--max_beta', type=float, required=True,
                    help='Maximum beta value for training (exclusive)')
parser.add_argument('--beta_gap', type=float, required=True,
                    help='Beta gap for training')
parser.add_argument('--n_epochs', type=int, default=16,
                    help='Number of training epochs (default: 16)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training (default: 32)')
parser.add_argument('--n_subsets', type=int, default=8,
                    help='Number of subsets for training (default: 8)')
parser.add_argument('--n_workers', type=int, default=0,
                    help='Number of workers for training (default: 0)')
parser.add_argument('--rand_seed', type=int, default=1331,
                    help='Random seed for training (default: 1331)')
parser.add_argument('--if_identity_init', action='store_true',
                    help='Initialize models to produce identity transformation (default: False)')
parser.add_argument('--if_check_jac', action='store_true',
                    help='Check Jacobian for training (default: False)')
parser.add_argument('--if_continue', action='store_true',
                    help='Continue training from the best model (default: False)')


args = parser.parse_args()

# Print all arguments
print("="*60)
print(">>> Arguments:")
print(f"Lattice size: {args.lattice_size}")
print(f"Minimum beta: {args.min_beta}")
print(f"Maximum beta: {args.max_beta}")
print(f"Beta gap: {args.beta_gap}")
print(f"Number of epochs: {args.n_epochs}")
print(f"Batch size: {args.batch_size}")
print(f"Number of subsets: {args.n_subsets}")
print(f"Number of workers: {args.n_workers}")
print(f"Random seed: {args.rand_seed}")
print(f"Identity initialization: {args.if_identity_init}")
print(f"Check Jacobian: {args.if_check_jac}")
print(f"Continue training: {args.if_continue}")
print("="*60)


# Print info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This program requires GPU to run.")

print(f"CUDA devices: {torch.cuda.device_count()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
device = 'cuda'
    
# Set random seed
set_seed(args.rand_seed)
save_tag = f"seed{args.rand_seed}"

# Parameters
lattice_size = args.lattice_size
print(f"Lattice size: {lattice_size}x{lattice_size}")
print(f"Number of subsets: {args.n_subsets}")
print(f"Check Jacobian: {args.if_check_jac}")
print(f"Identity initialization: {args.if_identity_init}")
print(f"Continue training: {args.if_continue}")

# Create output directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Set default type
torch.set_default_dtype(torch.float32)

# %%
# initialize the field transformation
nn_ft = FieldTransformation(lattice_size, device=device, n_subsets=args.n_subsets, if_check_jac=args.if_check_jac, num_workers=args.n_workers, identity_init=args.if_identity_init, save_tag=save_tag)

if args.if_continue:
    start_beta = args.min_beta - args.beta_gap
    nn_ft._load_best_model(train_beta=start_beta)
    print(f">>> Loaded the best model at beta = {start_beta} to continue training")
else:
    print(">>> Training from scratch")

# Parallelize the models
for i in range(len(nn_ft.models)):
    nn_ft.models[i] = DataParallel(nn_ft.models[i])

for train_beta in np.arange(args.min_beta, args.max_beta + args.beta_gap, args.beta_gap):
    beta_start_time = time.time()
    
    # load the data
    data = np.load(f'gauges/theta_ori_L{lattice_size}_beta{train_beta}.npy')
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
    
    # Calculate and print timing information
    beta_time = time.time() - beta_start_time
    total_time = time.time() - start_time
    
    # Format times as HH:MM:SS
    beta_time_formatted = str(datetime.timedelta(seconds=int(beta_time)))
    total_time_formatted = str(datetime.timedelta(seconds=int(total_time)))
    
    print(f"\n>>> Completed beta = {train_beta}")
    print(f">>> Time for this beta: {beta_time_formatted}")
    print(f">>> Total elapsed time: {total_time_formatted}")
    print("="*50)

