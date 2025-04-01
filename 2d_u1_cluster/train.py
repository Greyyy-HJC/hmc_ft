# %%
import torch
import numpy as np
import argparse
import os
import sys
from field_trans_opt import FieldTransformation
from torch.nn.parallel import DataParallel

# Make sure we're using reasonable defaults
torch.set_default_dtype(torch.float32)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Training parameters for Field Transformation')
parser.add_argument('--lattice_size', type=int, default=8,
                    help='Size of the lattice (default: 8)')
parser.add_argument('--min_beta', type=float, required=True,
                    help='Minimum beta value for training')
parser.add_argument('--max_beta', type=float, required=True,
                    help='Maximum beta value for training (exclusive)')
parser.add_argument('--n_epochs', type=int, default=4,
                    help='Number of training epochs (default: 4)')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for training (default: 16)')
parser.add_argument('--n_subsets', type=int, default=8,
                    help='Number of subsets for training (default: 8)')
parser.add_argument('--n_workers', type=int, default=0,
                    help='Number of workers for training (default: 0)')
parser.add_argument('--if_check_jac', action='store_true',
                    help='Check Jacobian during training')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug mode with more outputs')

args = parser.parse_args()

# Print system info
print(">>> PBS_NODEFILE content:")
if os.path.exists(os.environ.get('PBS_NODEFILE', '')):
    with open(os.environ['PBS_NODEFILE'], 'r') as f:
        print(f.read().strip())
else:
    print("PBS_NODEFILE not found")

print(f"Start time: {np.datetime_as_string(np.datetime64('now'), unit='s')}")
print(f"Python {sys.version.split()[0]}")
print(f"Python path: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
else:
    print("Using CPU only")
    device = 'cpu'

# Parameters
lattice_size = args.lattice_size
print(f"Lattice size: {lattice_size}x{lattice_size}")
print(f"Number of subsets: {args.n_subsets}")
print(f"Check Jacobian: {args.if_check_jac}")

# Create output directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

try:
    # Initialize field transformation
    print("Initializing field transformation...")
    nn_ft = FieldTransformation(
        lattice_size, 
        device=device, 
        n_subsets=args.n_subsets, 
        if_check_jac=args.if_check_jac, 
        num_workers=args.n_workers
    )

    # Script and parallelize the models
    for i in range(len(nn_ft.models)):
        # Skip JIT scripting to avoid issues
        print(f"Using standard PyTorch model {i}")
        
        # Apply data parallelism if CUDA is available with multiple GPUs
        if torch.cuda.device_count() > 1:
            nn_ft.models[i] = DataParallel(nn_ft.models[i])
            print(f"Using {torch.cuda.device_count()} GPUs for model {i}")

    # Train for each beta value
    for train_beta in range(int(args.min_beta), int(args.max_beta) + 1):
        try:
            # Load the data
            data_path = f'dump/theta_ori_L{lattice_size}_beta{train_beta}.npy'
            print(f"Loading data from {data_path}")
            
            if not os.path.exists(data_path):
                print(f"Error: Data file {data_path} not found!")
                continue
                
            data = np.load(data_path)
            tensor_data = torch.from_numpy(data).float().to(device)
            print(f"Loaded data shape: {tensor_data.shape}")
            
            # Limit data size for testing if needed
            if args.debug and len(tensor_data) > 128:
                tensor_data = tensor_data[:128]
                print(f"Limited data to {len(tensor_data)} samples for debugging")

            # Split the data into training and testing
            train_size = int(0.8 * len(tensor_data))
            train_data = tensor_data[:train_size]
            test_data = tensor_data[train_size:]
            print(f"Training data shape: {train_data.shape}")
            print(f"Testing data shape: {test_data.shape}")

            # Train the model
            print(f"\n>>> Training the model at beta = {train_beta}")
            nn_ft.train(train_data, test_data, train_beta, n_epochs=args.n_epochs, batch_size=args.batch_size)
            
        except Exception as e:
            print(f"Error during training for beta = {train_beta}: {str(e)}")
            import traceback
            traceback.print_exc()
            
except Exception as e:
    print(f"Error in main script: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    print(f"End time: {np.datetime_as_string(np.datetime64('now'), unit='s')}")
    print("Training script completed.")


