# %%
import torch
import torch.autograd.functional as F


L = 8
original_field = 2 * torch.pi * (torch.rand(2, L, L) - 0.5) # [2, L, L]
n_index = 8


def get_mask(index):
    mask = torch.zeros_like(original_field)
    if index == 0:
        mask[0, 0::2, 0::2] = 1
    elif index == 1:
        mask[0, 0::2, 1::2] = 1
    elif index == 2:
        mask[0, 1::2, 0::2] = 1
    elif index == 3:
        mask[0, 1::2, 1::2] = 1
    elif index == 4:
        mask[1, 0::2, 0::2] = 1
    elif index == 5:
        mask[1, 0::2, 1::2] = 1
    elif index == 6:
        mask[1, 1::2, 0::2] = 1
    elif index == 7:
        mask[1, 1::2, 1::2] = 1
    return mask

def get_coef(index):
    if index == 0:
        return 0.1
    elif index == 1:
        return 0.3
    elif index == 2:
        return 0.14
    elif index == 3:
        return -0.11
    elif index == 4:
        return -0.1
    elif index == 5:
        return 0.1
    elif index == 6:
        return 0.14
    elif index == 7:
        return 0.22
        
        
def plaq_from_field(theta):
    """
    Calculate the plaquette value for a given field configuration.
    """
    theta0, theta1 = theta[0], theta[1]
    thetaP = theta0 - theta1 - torch.roll(theta0, shifts=-1, dims=1) + torch.roll(theta1, shifts=-1, dims=0)

    return thetaP


def field_trans(theta):
    theta_copy = theta.clone()
    for index in range(n_index):
        mask = get_mask(index)
        coef = get_coef(index)
        plaq = plaq_from_field(theta_copy)
        
        sin_plaq_dir0_1 = - torch.sin(plaq)
        sin_plaq_dir0_2 = torch.sin(torch.roll(plaq, shifts=1, dims=1))
        
        sin_plaq_dir1_1 = torch.sin(plaq)
        sin_plaq_dir1_2 = - torch.sin(torch.roll(plaq, shifts=1, dims=0))
        
        # sin_plaq_duplicate = torch.stack([sin_plaq_dir0_1, sin_plaq_dir1_1], dim=0) # [2, L, L]
        sin_plaq_duplicate = torch.stack([sin_plaq_dir0_2, sin_plaq_dir1_2], dim=0) # [2, L, L]
        
        theta_copy = theta_copy + sin_plaq_duplicate * mask * coef
        
    return theta_copy
    
def hand_jac(theta):
    theta_copy = theta.clone()
    
    jac_logdet = 0
    
    for index in range(n_index):
        plaq = plaq_from_field(theta_copy)
        mask = get_mask(index)
        coef = get_coef(index)
        
        
        sin_plaq_dir0_1 = - torch.sin(plaq)
        sin_plaq_dir0_2 = torch.sin(torch.roll(plaq, shifts=1, dims=1))
        
        sin_plaq_dir1_1 = torch.sin(plaq)
        sin_plaq_dir1_2 = - torch.sin(torch.roll(plaq, shifts=1, dims=0))
        
        
        cos_plaq_dir0_1 = - torch.cos(plaq)
        cos_plaq_dir0_2 = - torch.cos(torch.roll(plaq, shifts=1, dims=1))
        
        cos_plaq_dir1_1 = - torch.cos(plaq)
        cos_plaq_dir1_2 = - torch.cos(torch.roll(plaq, shifts=1, dims=0))
        
        
        # sin_plaq_duplicate = torch.stack([sin_plaq_dir0_1, sin_plaq_dir1_1], dim=0) # [2, L, L]
        sin_plaq_duplicate = torch.stack([sin_plaq_dir0_2, sin_plaq_dir1_2], dim=0) # [2, L, L]
        
        
        # cos_plaq_duplicate = torch.stack([cos_plaq_dir0_1, cos_plaq_dir1_1], dim=0) # [2, L, L]
        cos_plaq_duplicate = torch.stack([cos_plaq_dir0_2, cos_plaq_dir1_2], dim=0) # [2, L, L]
        
        jac_logdet += torch.log(1 + cos_plaq_duplicate * mask * coef).sum()
        
        theta_copy = theta_copy + sin_plaq_duplicate * mask * coef
        
    return jac_logdet


# %%
plaq = plaq_from_field(original_field) # [L, L]

auto_jac = F.jacobian(field_trans, original_field)
auto_jac_2d = auto_jac.reshape(original_field.numel(), original_field.numel())
auto_logdet = torch.logdet(auto_jac_2d)

hand_logdet = hand_jac(original_field)

diff = (auto_logdet - hand_logdet) / hand_logdet
print("auto_logdet = ", auto_logdet)
print("hand_logdet = ", hand_logdet)
print("diff = ", diff)


transformed_field = field_trans(original_field)
field_diff = (transformed_field - original_field).abs().sum()
print("field_diff = ", field_diff)



# %%
