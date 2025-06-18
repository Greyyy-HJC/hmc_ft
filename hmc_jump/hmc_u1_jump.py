import torch
from tqdm import tqdm
from utils import plaq_from_field, topo_from_field, plaq_mean_from_field, regularize

class HMC_U1_AuxJump:
    def __init__(self, lattice_size, beta, n_thermalization_steps, n_steps, step_size,
                 aux_jump_prob=0.1, aux_jump_strength=3.14, device="cpu"):
        self.lattice_size = lattice_size
        self.beta = beta
        self.n_thermalization_steps = n_thermalization_steps
        self.n_steps = n_steps
        self.dt = step_size
        self.device = torch.device(device)
        self.aux_jump_prob = aux_jump_prob
        self.aux_jump_strength = aux_jump_strength

        torch.set_default_dtype(torch.float32)
        torch.set_default_device(self.device)
        torch.manual_seed(1331)

    def initialize(self):
        return torch.zeros([2, self.lattice_size, self.lattice_size])

    def action(self, theta):
        theta_P = plaq_from_field(theta)
        thetaP_wrapped = regularize(theta_P)
        return (-self.beta) * torch.sum(torch.cos(thetaP_wrapped))

    def force(self, theta):
        theta.requires_grad_(True)
        action_val = self.action(theta)
        action_val.backward(retain_graph=True)
        ff = theta.grad
        theta.requires_grad_(False)
        return ff

    def leapfrog(self, theta, pi):
        dt = self.dt
        theta_ = theta + 0.5 * dt * pi
        pi_ = pi - dt * self.force(theta_)
        for _ in range(self.n_steps - 1):
            theta_ = theta_ + dt * pi_
            pi_ = pi_ - dt * self.force(theta_)
        theta_ = theta_ + 0.5 * dt * pi_
        return regularize(theta_), pi_

    def aux_jump(self, theta):
        """
        Add a random global topological twist to help jump across Q sectors.
        """
        twist = self.aux_jump_strength * (2 * torch.rand(1).item() - 1)
        direction = torch.randint(0, 2, (1,)).item()  # 0 or 1 for mu=0 or mu=1
        theta[direction] += twist
        return regularize(theta)

    def metropolis_step(self, theta):
        if torch.rand(1).item() < self.aux_jump_prob:
            theta = self.aux_jump(theta)

        pi = torch.randn_like(theta)
        H_old = self.action(theta) + 0.5 * torch.sum(pi**2)

        new_theta, new_pi = self.leapfrog(theta.clone(), pi.clone())
        H_new = self.action(new_theta) + 0.5 * torch.sum(new_pi**2)

        delta_H = H_new - H_old
        accept_prob = torch.exp(-delta_H)

        if torch.rand([], device=self.device) < accept_prob:
            return new_theta, True, H_new.item()
        else:
            return theta, False, H_old.item()

    def run(self, n_iterations, theta, store_interval=1):
        theta_ls, plaq_ls, hamiltonians, topological_charges = [], [], [], []
        acceptance_count = 0

        for i in tqdm(range(n_iterations), desc="Running HMC+AUX"):
            theta, accepted, H_val = self.metropolis_step(theta)
            if i % store_interval == 0:
                theta = regularize(theta)
                theta_ls.append(theta)
                plaq_ls.append(plaq_mean_from_field(theta).item())
                hamiltonians.append(H_val)
                topological_charges.append(topo_from_field(theta).item())
            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / n_iterations
        return theta_ls, plaq_ls, acceptance_rate, topological_charges, hamiltonians
