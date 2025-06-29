
import torch
from utils import set_seed
from utils import plaq_from_field_batch
from field_trans_opt import FieldTransformation
# from field_trans import FieldTransformation

def main():
    # Fixed parameters
    lattice_size = 64
    train_beta = 2.0
    device = 'cpu'
    save_tag = 'seed2008'
    random_seed = 2008
    
    # Set random seed for reproducibility
    set_seed(random_seed)
    
    # Initialize field transformation
    device_obj = torch.device(device)
    n_subsets = 8
    
    print(f"\n>>> Initializing Field Transformation...")
    ft = FieldTransformation(
        lattice_size=lattice_size,
        device=device,
        n_subsets=n_subsets,
        if_check_jac=False,
        num_workers=0,
        identity_init=True,
        save_tag=save_tag
    )
    
    # Load trained model
    print(f"\n>>> Loading trained model...")
    try:
        ft._load_best_model(train_beta)
        print(f"✓ Successfully loaded model trained at beta = {train_beta}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Please check if the model file exists and the path is correct")
        return
    
    print(f"\n>>> Testing Field Transformation Effects...")
    print("-" * 60)
    
    # Generate random theta configuration
    # theta has shape [2, L, L] representing U(1) gauge field
    theta = torch.randn(2, lattice_size, lattice_size, device=device_obj)
    
    print(f"Original theta statistics:")
    print(f"  Mean: {theta.mean().item():.6f}")
    print(f"  Std:  {theta.std().item():.6f}")
    print(f"  Min:  {theta.min().item():.6f}")
    print(f"  Max:  {theta.max().item():.6f}")
    
    # Apply field transformation
    with torch.no_grad():
        theta_ft = ft.field_transformation(theta)
    
    print(f"\nTransformed theta statistics:")
    print(f"  Mean: {theta_ft.mean().item():.6f}")
    print(f"  Std:  {theta_ft.std().item():.6f}")
    print(f"  Min:  {theta_ft.min().item():.6f}")
    print(f"  Max:  {theta_ft.max().item():.6f}")
    
    # Calculate plaquette values
    # Need to add batch dimension for plaq_from_field_batch
    theta_batch = theta.unsqueeze(0)  # [1, 2, L, L]
    theta_ft_batch = theta_ft.unsqueeze(0)  # [1, 2, L, L]
    
    plaq_original = plaq_from_field_batch(theta_batch)  # [1, L, L]
    plaq_transformed = plaq_from_field_batch(theta_ft_batch)  # [1, L, L]
    
    # Calculate average plaquette values
    avg_plaq_original = torch.mean(torch.cos(plaq_original)).item()
    avg_plaq_transformed = torch.mean(torch.cos(plaq_transformed)).item()
    
    print(f"\nPlaquette values:")
    print(f"  Original:    {avg_plaq_original:.8f}")
    print(f"  Transformed: {avg_plaq_transformed:.8f}")
    print(f"  Difference:  {avg_plaq_transformed - avg_plaq_original:.8f}")
    print(f"  Rel. change: {(avg_plaq_transformed - avg_plaq_original)/avg_plaq_original*100:.4f}%")
    
    # Check transformation magnitude
    transformation_magnitude = torch.norm(theta_ft - theta).item()
    theta_magnitude = torch.norm(theta).item()
    relative_change = transformation_magnitude / theta_magnitude
    
    print(f"\nTransformation magnitude:")
    print(f"  ||theta_ft - theta||: {transformation_magnitude:.6f}")
    print(f"  ||theta||:           {theta_magnitude:.6f}")
    print(f"  Relative change:     {relative_change:.6f}")
    
    # Test inverse transformation
    print(f"\n>>> Testing inverse transformation...")
    
    with torch.no_grad():
        theta_reconstructed = ft.inverse_field_transformation(theta_ft)
    
    reconstruction_error = torch.norm(theta_reconstructed - theta).item()
    original_norm = torch.norm(theta).item()
    relative_error = reconstruction_error / original_norm
    
    print(f"  Reconstruction error: {reconstruction_error:.8f}")
    print(f"  Relative error: {relative_error:.8f}")
    
    if relative_error < 1e-4:
        print(f"  ✓ Inverse transformation works well (rel. error < 1e-4)")
    elif relative_error < 1e-2:
        print(f"  ⚠ Inverse transformation has moderate error (1e-4 ≤ rel. error < 1e-2)")
    else:
        print(f"  ✗ Inverse transformation has large error (rel. error ≥ 1e-2)")
    
    print(f"\n>>> Field transformation check completed!")

if __name__ == "__main__":
    main()