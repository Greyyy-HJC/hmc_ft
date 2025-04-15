# %%
import torch
import os
from hmc_u1 import HMC_U1
from hmc_u1_ft import HMC_U1_FT
from field_trans import FieldTransformation
from utils import hmc_summary


# Parameters
lattice_size = 16
volume = lattice_size ** 2
beta = 4.0
n_thermalization_steps = 200
n_steps = 50
step_size = 0.15
store_interval = 1
n_iterations = store_interval * 1024

# Initialize device
device = 'cpu'

# Set default type
torch.set_default_dtype(torch.float32)


# %%
#! No field transformation
print(">>> No Field Transformation HMC Simulation: ")

# Initialize HMC
hmc = HMC_U1(lattice_size, beta, n_thermalization_steps, n_steps, step_size, device=device, if_tune_step_size=False)

# Thermalize the system
theta_thermalized, therm_plaq_ls, therm_acceptance_rate = hmc.thermalize()

# Run HMC without field transformation
config_ls, plaq_ls, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized, store_interval)

print(">>> Simulation completed")


# Compute autocorrelation of topological charges
max_lag = 20
hmc_fig = hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate)
hmc_fig.savefig(f'plots/comparison_hmc_L{lattice_size}_beta{beta}.pdf', transparent=True)


# %%
#! Field transformation
print(">>> Neural Network Field Transformation HMC Simulation: ")

# initialize the field transformation
n_subsets = 8
n_workers = 8
nn_ft = FieldTransformation(lattice_size, device=device, n_subsets=n_subsets, if_check_jac=False, num_workers=n_workers)

# Check if model file exists before loading
model_path = f'models/best_model_L{lattice_size}.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the trained model using the _load_best_model method
print(">>> Loading trained model")
try:
    nn_ft._load_best_model(train_beta=None) # * None means using the best model after training
    print(">>> Model loaded successfully")
except Exception as e:
    print(f">>> Error loading model: {e}")
    raise

field_transformation = nn_ft.field_transformation
compute_jac_logdet = nn_ft.compute_jac_logdet_compiled

step_size = 0.1 # todo
hmc = HMC_U1_FT(lattice_size, beta, n_thermalization_steps, n_steps, step_size, field_transformation=field_transformation, compute_jac_logdet=compute_jac_logdet, device=device, if_tune_step_size=False)

# Thermalize the system
theta_thermalized, therm_plaq_ls, therm_acceptance_rate = hmc.thermalize()

# Run HMC without field transformation
final_config, plaq_ls, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized, store_interval)

print(">>> Simulation completed")

# Compute autocorrelation of topological charges
max_lag = 20
hmc_fig = hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate)
hmc_fig.savefig(f'plots/comparison_hmc_ft_L{lattice_size}_beta{beta}.pdf', transparent=True)



# %%
