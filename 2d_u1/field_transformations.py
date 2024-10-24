import numpy as np

def manual_field_transformation(U, epsilon=0.1, cl=1.0):
    # Use sin and arctan to adjust the transformation
    delta_U = cl * np.arctan(np.sin(U))
    
    # Perform the transformation, ensuring reversibility
    U_transformed = U * np.exp(epsilon * delta_U)
    
    return U_transformed