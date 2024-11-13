# %%
import torch
import torch.linalg as linalg
import torch.autograd.functional as F

def compute_jacobian_log_det_wrong(theta_new, field_transformation):
    """
    Compute the log determinant of the Jacobian matrix of the transformation.
    
    Parameters:
    -----------
    theta_new : torch.Tensor
        The new field configuration after transformation.
    """
    # Compute Jacobian using torch.autograd.functional.jacobian
    jacobian = F.jacobian(field_transformation, theta_new)

    # Ensure the Jacobian is square by reshaping it correctly
    jacobian_2d = jacobian.view(jacobian.shape[0], -1)

    # Compute singular values
    s = linalg.svdvals(jacobian_2d)

    # Compute log determinant as sum of log of singular values
    log_det = torch.sum(torch.log(s))

    return log_det

def compute_jacobian_log_det(theta_new, field_transformation):
    """
    Compute the log determinant of the Jacobian matrix of the transformation.

    Parameters:
    -----------
    theta_new : torch.Tensor
        The new field configuration after transformation.
    """
    # Compute Jacobian using torch.autograd.functional.jacobian
    jacobian = F.jacobian(field_transformation, theta_new)

    # Reshape the Jacobian to a 2D matrix
    jacobian_2d = jacobian.reshape(theta_new.numel(), theta_new.numel())

    # Compute singular values
    s = linalg.svdvals(jacobian_2d)

    # Print any negative singular values
    negative_values = s[s < 0]
    if len(negative_values) > 0:
        print("Warning: Found negative singular values:", negative_values)

    # Compute log determinant as sum of log of singular values
    log_det = torch.sum(torch.log(s))

    return log_det

def identity_transformation(x):
    return x

def test_jacobian_log_det_identity():
    theta_new = torch.eye(3, requires_grad=True)
    log_det = compute_jacobian_log_det(theta_new, identity_transformation)
    expected_log_det = torch.tensor(0.0)

    print(f"Expected log det: {expected_log_det}")
    print(f"Computed log det: {log_det}")


def scaling_transformation(x, scale_factor):
    return x * scale_factor

def test_jacobian_log_det_scaling():
    theta_new = torch.eye(3, requires_grad=True)
    scale_factor = 2.0  # You can choose any non-zero scale factor
    n = theta_new.numel()  # Total number of elements

    # Define a lambda function to fix the scale factor
    transformation = lambda x: scaling_transformation(x, scale_factor)

    # Compute the log determinant using the function
    log_det = compute_jacobian_log_det(theta_new, transformation)

    # Calculate the expected log determinant analytically
    expected_log_det = n * torch.log(torch.tensor(scale_factor))

    print(f"Expected log det: {expected_log_det.item()}")
    print(f"Computed log det: {log_det.item()}") 

if __name__ == "__main__":
    test_jacobian_log_det_identity()
    test_jacobian_log_det_scaling()
# %%
