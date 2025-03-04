# HMC with GAN Overrelaxation

This project implements a Hybrid Monte Carlo (HMC) algorithm with Generative Adversarial Network (GAN) overrelaxation for U(1) lattice gauge theory in 2D. The goal is to reduce the autocorrelation time in HMC sampling by using a GAN to generate new configurations that preserve the action.

## Overview

The project consists of the following components:

1. **HMC Implementation**: Standard HMC algorithm for U(1) lattice gauge theory.
2. **GAN Implementation**: A GAN model that learns to generate field configurations from HMC samples.
3. **Overrelaxation**: A method that uses the trained GAN to generate new configurations with the same action.
4. **Comparison**: Tools to compare the autocorrelation times of standard HMC and HMC with GAN overrelaxation.

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- tqdm

## Usage

To run the experiment, use the `main.py` script:

```bash
python main.py [options]
```

### Command Line Options

- `--lattice_size`: Size of the lattice (default: 8)
- `--beta`: Inverse coupling constant (default: 2.0)
- `--n_thermalization`: Number of thermalization steps (default: 500)
- `--n_hmc_iterations`: Number of HMC iterations (default: 1000)
- `--n_steps`: Number of leapfrog steps in each HMC trajectory (default: 10)
- `--step_size`: Step size for each leapfrog step (default: 0.1)
- `--latent_dim`: Latent dimension for GAN (default: 64)
- `--gan_epochs`: Number of epochs for GAN training (default: 50)
- `--batch_size`: Batch size for GAN training (default: 32)
- `--device`: Device to use (cuda or cpu, default: cuda if available, otherwise cpu)
- `--max_lag`: Maximum lag for autocorrelation calculation (default: 100)
- `--store_interval`: Store interval for HMC (default: 1)

### Example

```bash
python main.py --lattice_size 16 --beta 4.0 --n_hmc_iterations 2000 --gan_epochs 100
```

## Output

The script creates a directory in `dump/` with a timestamp and the lattice size and beta value. The directory contains:

- `params.txt`: Parameters used for the run
- `generator.pt`: Saved trained generator model
- `standard_hmc_results.png`: Plot of standard HMC results
- `gan_hmc_results.png`: Plot of HMC with GAN results
- `autocorrelation_comparison.png`: Comparison of autocorrelation functions
- `results.txt`: Summary of results, including acceptance rates and autocorrelation times
- `autocor_standard.npy` and `autocor_gan.npy`: Saved autocorrelation data

## How It Works

1. The script first runs standard HMC to generate configurations.
2. These configurations are used to train a GAN.
3. The trained GAN is then used in the HMC algorithm to perform overrelaxation steps.
4. The autocorrelation times of both methods are calculated and compared.

## Files

- `main.py`: Main script to run the experiment
- `hmc_u1.py`: Implementation of the HMC algorithm
- `overrelax.py`: Implementation of the GAN and overrelaxation methods
- `utils.py`: Utility functions for analysis and visualization

## References

- Hybrid Monte Carlo algorithm: https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo
- Generative Adversarial Networks: https://arxiv.org/abs/1406.2661
- Overrelaxation in Monte Carlo: https://arxiv.org/abs/hep-lat/9309002 