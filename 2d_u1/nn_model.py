import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.layer(x)

class NNFieldTransformation:
    def __init__(self, lattice_size):
        self.lattice_size = lattice_size
        self.model = SimpleNN(lattice_size * lattice_size, lattice_size * lattice_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def __call__(self, U):
        U_flat = U.flatten()
        U_tensor = torch.tensor(U_flat, dtype=torch.float32)
        U_transformed_tensor = self.model(U_tensor)
        U_transformed = U_transformed_tensor.detach().numpy().reshape(U.shape)
        return np.mod(U_transformed, 2 * np.pi)

    def train(self, hmc_instance, n_iterations):
        for _ in tqdm(range(n_iterations), desc="Training Neural Network"):
            U = hmc_instance.initialize()
            U_flat = U.flatten()
            U_tensor = torch.tensor(U_flat, dtype=torch.float32, requires_grad=True)

            U_transformed = self.model(U_tensor)

            action_original = hmc_instance.action(U)
            action_transformed = hmc_instance.action(U_transformed.detach().numpy().reshape(self.lattice_size, self.lattice_size))
            loss = torch.tensor(action_transformed - action_original, dtype=torch.float32, requires_grad=True)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
