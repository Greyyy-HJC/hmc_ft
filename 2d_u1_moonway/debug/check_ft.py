# %%
import torch
import os
import numpy as np
import torch.nn as nn
from field_trans import FieldTransformation

# Parameters
lattice_size = 16
train_beta = 6.0

# Initialize device
device = 'cpu'

# Change to parent directory
os.chdir('/home/jinchen/git/anl/hmc_ft/2d_u1_moonway')

data = np.load(f'dump/theta_ori_L{lattice_size}_beta{train_beta}.npy')
tensor_data = torch.from_numpy(data).float().to(device)
print(f"Loaded data shape: {tensor_data.shape}")

theta_ori = tensor_data[10:12]

# %%
#! Field transformation
print(">>> Neural Network Field Transformation HMC Simulation: ")

# initialize the field transformation
n_subsets = 8
n_workers = 8
nn_ft = FieldTransformation(lattice_size, device=device, n_subsets=n_subsets, if_check_jac=False, num_workers=n_workers, identity_init=True, save_tag=None)


# Load the trained model
nn_ft._load_best_model(train_beta) # ! None means using the best model after training


#! Check with FT
theta_new = nn_ft.inverse(theta_ori)
force_ori_ft = nn_ft.compute_force(theta_new, beta=1)
force_new_ft = nn_ft.compute_force(theta_new, train_beta, transformed=True)
diff_with_ft = torch.norm(force_new_ft, p=2)
print(f"Force difference with FT: {diff_with_ft}")


#! Reset models
for model in nn_ft.models:
    for param in model.parameters():
        nn.init.zeros_(param)


#! Check without FT
theta_new = nn_ft.inverse(theta_ori)
force_ori_noft = nn_ft.compute_force(theta_new, beta=1)
force_new_noft = nn_ft.compute_force(theta_new, train_beta, transformed=False)
diff_without_ft = torch.norm(force_new_noft, p=2)
print(f"Force difference without FT: {diff_without_ft}")


# %%
