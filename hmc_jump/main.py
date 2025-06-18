# %%
import torch
import time
from hmc_u1 import HMC_U1
from utils import hmc_summary, set_seed



# random seed
set_seed(1331)

# Parameters
lattice_size = 16
volume = lattice_size ** 2
beta = 6
n_thermalization_steps = 200
n_steps = 50
step_size = 0.1
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
hmc = HMC_U1(lattice_size, beta, n_thermalization_steps, n_steps, step_size, device=device, if_tune_step_size=False) # todo

# Thermalize the system
print(">>> Starting thermalization...")
therm_start_time = time.time()
theta_thermalized, therm_plaq_ls, therm_acceptance_rate = hmc.thermalize()
therm_end_time = time.time()
therm_time = therm_end_time - therm_start_time
print(f">>> Thermalization completed in {therm_time:.2f} seconds")

# Run HMC without field transformation
print(">>> Starting simulation...")
run_start_time = time.time()
config_ls, plaq_ls, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized, store_interval)
run_end_time = time.time()
run_time = run_end_time - run_start_time
print(f">>> Simulation completed in {run_time:.2f} seconds")
print(f">>> Total time (Standard HMC): {therm_time + run_time:.2f} seconds")


# Compute autocorrelation of topological charges
max_lag = 20
hmc_fig = hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate)
hmc_fig.savefig(f'plots/comparison_hmc_L{lattice_size}_beta{beta:.1f}.pdf', transparent=True)


# %%
from hmc_u1_jump import HMC_U1_AuxJump  # 替换为你保存的文件名

print("\n>>> Auxiliary Topological Jump HMC Simulation: ")

# Initialize HMC with auxiliary jump
hmc_aux = HMC_U1_AuxJump(
    lattice_size=lattice_size,
    beta=beta,
    n_thermalization_steps=n_thermalization_steps,
    n_steps=n_steps,
    step_size=step_size,
    aux_jump_prob=0.1,
    aux_jump_strength=3.14,
    device=device
)

# Thermalize
print(">>> Starting thermalization with aux jump...")
theta_ls_aux, therm_plaq_aux, therm_accept_aux, topo_aux_therm, H_aux_therm = hmc_aux.run(n_thermalization_steps, hmc_aux.initialize())
theta_thermalized_aux = theta_ls_aux[-1]

# Run simulation with auxiliary jumps
print(">>> Starting simulation with aux jump...")
config_ls_aux, plaq_ls_aux, accept_aux, topo_aux, H_aux = hmc_aux.run(n_iterations, theta_thermalized_aux, store_interval)

# Save result
hmc_fig_aux = hmc_summary(
    beta,
    max_lag,
    volume,
    therm_plaq_aux,
    plaq_ls_aux,
    topo_aux,
    H_aux,
    therm_accept_aux,
    accept_aux
)
hmc_fig_aux.savefig(f'plots/comparison_auxHMC_L{lattice_size}_beta{beta:.1f}.pdf', transparent=True)

# %%
