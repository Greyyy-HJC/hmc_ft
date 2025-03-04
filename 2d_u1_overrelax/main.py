import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

from hmc_u1 import HMC_U1
from overrelax import Generator, Discriminator, train_gan, run_hmc_with_gan
from utils import auto_by_def, hmc_summary, plot_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HMC with GAN overrelaxation')
    parser.add_argument('--lattice_size', type=int, default=8, help='Lattice size')
    parser.add_argument('--beta', type=float, default=2.0, help='Inverse coupling constant')
    parser.add_argument('--n_thermalization', type=int, default=500, help='Number of thermalization steps')
    parser.add_argument('--n_hmc_iterations', type=int, default=1000, help='Number of HMC iterations')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of leapfrog steps')
    parser.add_argument('--step_size', type=float, default=0.1, help='Step size for leapfrog')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension for GAN')
    parser.add_argument('--gan_epochs', type=int, default=100, help='Number of epochs for GAN training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for GAN training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--max_lag', type=int, default=100, help='Maximum lag for autocorrelation')
    parser.add_argument('--store_interval', type=int, default=1, help='Store interval for HMC')
    parser.add_argument('--gan_attempts', type=int, default=5, help='Number of attempts for GAN overrelaxation')
    parser.add_argument('--gan_threshold', type=float, default=0.5, help='Threshold for GAN overrelaxation')
    parser.add_argument('--gan_frequency', type=int, default=5, help='Frequency of applying GAN overrelaxation')
    parser.add_argument('--use_lcnn', action='store_true', help='Whether to use LCNN (if False, use regular CNN)')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "LCNN" if args.use_lcnn else "CNN"
    output_dir = f"dump/run_{timestamp}_L{args.lattice_size}_beta{args.beta}_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save parameters
    with open(f"{output_dir}/params.txt", "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # Initialize HMC
    print(f"\nInitializing HMC with lattice_size={args.lattice_size}, beta={args.beta}")
    hmc = HMC_U1(
        lattice_size=args.lattice_size,
        beta=args.beta,
        n_thermalization_steps=args.n_thermalization,
        n_steps=args.n_steps,
        step_size=args.step_size,
        device=args.device
    )
    
    # Thermalize
    print("\nThermalizing...")
    theta, therm_plaq_ls, therm_acceptance_rate = hmc.thermalize()
    print(f"\nThermalization acceptance rate: {therm_acceptance_rate:.4f}")
    
    # Run standard HMC
    print(f"\nRunning standard HMC for {args.n_hmc_iterations} iterations...")
    theta_ls, plaq_ls, acceptance_rate, topological_charges, hamiltonians = hmc.run(
        n_iterations=args.n_hmc_iterations,
        theta=theta,
        store_interval=args.store_interval
    )
    print(f"\nStandard HMC acceptance rate: {acceptance_rate:.4f}")
    
    # Calculate autocorrelation for standard HMC
    print("\nCalculating autocorrelation for standard HMC...")
    autocor_standard = auto_by_def(np.array(topological_charges), args.max_lag)
    
    # Plot standard HMC results
    print("\nPlotting standard HMC results...")
    hmc_fig = plot_results(
        args.beta, 
        therm_plaq_ls, 
        plaq_ls, 
        topological_charges, 
        hamiltonians, 
        autocor_standard, 
        title_suffix="(Standard HMC)"
    )
    hmc_fig.savefig(f"{output_dir}/standard_hmc_results.png")
    
    # Prepare dataset for GAN training
    print("\nPreparing dataset for GAN training...")
    # Convert list of tensors to a single tensor
    theta_tensor = torch.stack(theta_ls)
    dataset = data.TensorDataset(theta_tensor)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize GAN
    print(f"\nInitializing GAN... (使用{'LCNN' if args.use_lcnn else '常规CNN'})")
    generator = Generator(args.latent_dim, args.lattice_size, args.beta, use_lcnn=args.use_lcnn).to(args.device)
    discriminator = Discriminator(args.lattice_size, use_lcnn=args.use_lcnn).to(args.device)
    
    # Train GAN
    print(f"\nTraining GAN for {args.gan_epochs} epochs... (使用{'LCNN' if args.use_lcnn else '常规CNN'})")
    generator = train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        latent_dim=args.latent_dim,
        beta=args.beta,
        epochs=args.gan_epochs,
        device=args.device,
        output_dir=output_dir,
        use_lcnn=args.use_lcnn
    )
    
    # Save trained generator
    torch.save(generator.state_dict(), f"{output_dir}/generator.pt")
    
    # Run HMC with GAN overrelaxation
    print(f"\nRunning HMC with GAN overrelaxation for {args.n_hmc_iterations} iterations... (使用{'LCNN' if args.use_lcnn else '常规CNN'})")
    # Start from the same thermalized configuration
    theta_gan_ls, plaq_gan_ls, acceptance_rate_gan, topological_charges_gan, hamiltonians_gan = run_hmc_with_gan(
        hmc=hmc,
        generator=generator,
        latent_dim=args.latent_dim,
        n_iterations=args.n_hmc_iterations,
        theta=theta.clone(),  # Use a copy of the thermalized configuration
        store_interval=args.store_interval,
        device=args.device,
        gan_attempts=args.gan_attempts,
        gan_threshold=args.gan_threshold,
        gan_frequency=args.gan_frequency,
        use_lcnn=args.use_lcnn
    )
    print(f"\nHMC with GAN acceptance rate: {acceptance_rate_gan:.4f}")
    
    # Calculate autocorrelation for HMC with GAN
    print("\nCalculating autocorrelation for HMC with GAN...")
    autocor_gan = auto_by_def(np.array(topological_charges_gan), args.max_lag)
    
    # Plot HMC with GAN results
    print("\nPlotting HMC with GAN results...")
    gan_fig = plot_results(
        args.beta, 
        therm_plaq_ls, 
        plaq_gan_ls, 
        topological_charges_gan, 
        hamiltonians_gan, 
        autocor_gan, 
        title_suffix=f"(HMC with {'LCNN' if args.use_lcnn else '常规CNN'} GAN)"
    )
    gan_fig.savefig(f"{output_dir}/gan_hmc_results.png")
    
    # Compare autocorrelation times
    print("\nComparing autocorrelation times...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(autocor_standard)), autocor_standard, marker='o', label='Standard HMC')
    plt.plot(range(len(autocor_gan)), autocor_gan, marker='x', label=f"HMC with {'LCNN' if args.use_lcnn else '常规CNN'} GAN")
    plt.title('Autocorrelation Comparison', fontsize=18)
    plt.xlabel('MDTU', fontsize=16)
    plt.ylabel('Autocorrelation', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(linestyle=":")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/autocorrelation_comparison.png")
    
    # Calculate integrated autocorrelation time
    def integrated_autocorrelation_time(autocor):
        # Find the first index where autocorrelation drops below 0.1
        cutoff_idx = np.argmax(autocor < 0.1)
        if cutoff_idx == 0:  # If autocorrelation never drops below 0.1
            cutoff_idx = len(autocor)
        
        # Integrate up to cutoff_idx
        return 1 + 2 * np.sum(autocor[1:cutoff_idx])
    
    tau_int_standard = integrated_autocorrelation_time(autocor_standard)
    tau_int_gan = integrated_autocorrelation_time(autocor_gan)
    
    print(f"\nIntegrated autocorrelation time (Standard HMC): {tau_int_standard:.2f}")
    print(f"Integrated autocorrelation time (HMC with {'LCNN' if args.use_lcnn else '常规CNN'} GAN): {tau_int_gan:.2f}")
    print(f"Improvement factor: {tau_int_standard / tau_int_gan:.2f}x")
    
    # Save results to file
    with open(f"{output_dir}/results.txt", "w") as f:
        f.write(f"Standard HMC acceptance rate: {acceptance_rate:.4f}\n")
        f.write(f"HMC with {'LCNN' if args.use_lcnn else '常规CNN'} GAN acceptance rate: {acceptance_rate_gan:.4f}\n")
        f.write(f"Integrated autocorrelation time (Standard HMC): {tau_int_standard:.2f}\n")
        f.write(f"Integrated autocorrelation time (HMC with {'LCNN' if args.use_lcnn else '常规CNN'} GAN): {tau_int_gan:.2f}\n")
        f.write(f"Improvement factor: {tau_int_standard / tau_int_gan:.2f}x\n")
    
    # Save autocorrelation data
    np.save(f"{output_dir}/autocor_standard.npy", autocor_standard)
    np.save(f"{output_dir}/autocor_gan.npy", autocor_gan)
    
    # Create comparison histograms for plaquette mean values
    print("\nCreating plaquette mean value comparison histogram...")
    plt.figure(figsize=(10, 6))
    
    # Calculate histogram bins based on the range of plaquette values
    all_plaq = np.concatenate([plaq_ls, plaq_gan_ls])
    min_plaq, max_plaq = np.min(all_plaq), np.max(all_plaq)
    bin_width = (max_plaq - min_plaq) / 30  # Adjust number of bins as needed
    bins = np.arange(min_plaq, max_plaq + bin_width, bin_width)
    
    plt.hist(plaq_ls, bins=bins, alpha=0.7, label='Standard HMC', density=True)
    plt.hist(plaq_gan_ls, bins=bins, alpha=0.7, label=f"HMC with {'LCNN' if args.use_lcnn else '常规CNN'} GAN", density=True)
    
    plt.axvline(x=np.mean(plaq_ls), color='blue', linestyle='--', 
                label=f'Standard HMC Mean: {np.mean(plaq_ls):.4f}')
    plt.axvline(x=np.mean(plaq_gan_ls), color='orange', linestyle='--', 
                label=f"{'LCNN' if args.use_lcnn else '常规CNN'} GAN Mean: {np.mean(plaq_gan_ls):.4f}")
    
    plt.title('Plaquette Mean Value Distribution Comparison', fontsize=18)
    plt.xlabel('Plaquette Mean Value', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(linestyle=":")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plaquette_comparison_histogram.png")
    
    # Create comparison histograms for topological charges
    print("\nCreating topological charge comparison histogram...")
    plt.figure(figsize=(10, 6))
    
    # Determine the range of topological charges
    all_topo = np.concatenate([topological_charges, topological_charges_gan])
    min_topo, max_topo = int(np.min(all_topo)), int(np.max(all_topo))
    bins = np.arange(min_topo - 0.5, max_topo + 1.5, 1)  # Integer bins for topological charges
    
    plt.hist(topological_charges, bins=bins, alpha=0.7, label='Standard HMC', density=True)
    plt.hist(topological_charges_gan, bins=bins, alpha=0.7, label=f"HMC with {'LCNN' if args.use_lcnn else '常规CNN'} GAN", density=True)
    
    plt.axvline(x=np.mean(topological_charges), color='blue', linestyle='--', 
                label=f'Standard HMC Mean: {np.mean(topological_charges):.4f}')
    plt.axvline(x=np.mean(topological_charges_gan), color='orange', linestyle='--', 
                label=f"{'LCNN' if args.use_lcnn else '常规CNN'} GAN Mean: {np.mean(topological_charges_gan):.4f}")
    
    plt.title('Topological Charge Distribution Comparison', fontsize=18)
    plt.xlabel('Topological Charge', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(linestyle=":")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/topological_charge_comparison_histogram.png")
    
    print(f"\nAll results saved to {output_dir}")

if __name__ == "__main__":
    main() 