# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
from cnn_model import RGTransformerCNN, RGTransformerCNNAlt
from utils import plaq_mean_from_field, plaq_mean_theory, topo_from_field
from lametlat.utils.plot_settings import *

class RGTransformerEvaluator:
    def __init__(self, model_path='models/best_model.pt'):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize and load model
        # self.model = RGTransformerCNN().to(self.device) #todo
        self.model = RGTransformerCNNAlt().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, input_tensor):
        """Make predictions using the loaded model"""
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
        return output

    def load_and_process_data(self, rg_path, ori_path):
        """Load and process both RG and original configurations"""
        self.rg_configs = torch.load(rg_path)
        self.rg_tensor = torch.stack(self.rg_configs)
        self.irg_tensor = self.predict(self.rg_tensor)
        print("IRG configurations predicted with shape:", self.irg_tensor.shape)
        
        # Check for values outside [-pi, pi]
        mask = (self.irg_tensor < -torch.pi) | (self.irg_tensor > torch.pi)
        if torch.any(mask):
            print("Found values outside [-pi, pi] range:")
            print(self.irg_tensor[mask])
            print(f"Number of values outside range: {torch.sum(mask).item()}")
        
        
        
        
        # TODO: test use random irg configs    
        # rand = torch.randn(self.rg_tensor.shape).to(self.device) * 2 * torch.pi - torch.pi
        
        # self.irg_tensor = self.predict(rand)
        
        
        
        
        
        self.ori_configs = torch.load(ori_path)
        self.ori_tensor = torch.stack(self.ori_configs)
        
        return self.rg_tensor, self.ori_tensor, self.irg_tensor

    def calculate_plaquette_means(self, beta_ori=5, beta_rg=0.55, beta_irg=60, lattice_size=32):
        """Calculate and plot plaquette means"""
        # Calculate plaquette means
        ori_plaq_means = [plaq_mean_from_field(config).item() for config in self.ori_tensor]
        rg_plaq_means = [plaq_mean_from_field(config).item() for config in self.rg_tensor]
        irg_plaq_means = [plaq_mean_from_field(config).item() for config in self.irg_tensor]
        
        # Calculate theoretical values
        theoretical_plaq_ori = plaq_mean_theory(beta=beta_ori)
        theoretical_plaq_rg = plaq_mean_theory(beta=beta_rg)
        theoretical_plaq_irg = plaq_mean_theory(beta=beta_irg)
        
        # Create plot
        fig, ax = default_plot()
        ax.plot(ori_plaq_means, label='Original')
        ax.plot(rg_plaq_means, label='RG')
        ax.plot(irg_plaq_means, label='IRG')
        ax.axhline(y=theoretical_plaq_ori, color='black', linestyle='--', 
                  label=f'beta = {beta_ori}')
        ax.axhline(y=theoretical_plaq_rg, color='black', linestyle='-.', 
                  label=f'beta = {beta_rg}')
        ax.axhline(y=theoretical_plaq_irg, color='black', linestyle=':', 
                  label=f'beta = {beta_irg}')
        
        # Configure plot
        ax.set_xlabel('Configuration', **fs_p)
        ax.set_ylabel('Plaquette Mean', **fs_p)
        ax.grid(True)
        ax.legend(loc="upper right", ncol=3, fontsize=11)
        ax.set_ylim(auto_ylim([ori_plaq_means, rg_plaq_means, irg_plaq_means], [np.zeros_like(ori_plaq_means), np.zeros_like(rg_plaq_means), np.zeros_like(irg_plaq_means)], y_range_ratio=2))
        # ax.set_ylim(0.8, 1.1)
        plt.tight_layout()
        plt.savefig(f"plots/irg_plaq_means_L{lattice_size}.pdf", transparent=True)
        plt.show()
        
    def calculate_topological_charge(self, lattice_size):
        """Calculate and plot topological charge"""
        ori_topo = [topo_from_field(config).item() for config in self.ori_tensor]
        rg_topo = [topo_from_field(config).item() for config in self.rg_tensor]
        irg_topo = [topo_from_field(config).item() for config in self.irg_tensor]
        
        fig, ax = default_plot()
        ax.plot(ori_topo, label='Original')
        ax.plot(rg_topo, label='RG')
        ax.plot(irg_topo, label='IRG')
        
        # Configure plot
        ax.set_xlabel('Configuration', **fs_p)
        ax.set_ylabel('Topological Charge', **fs_p)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=11)
        ax.set_ylim(auto_ylim([ori_topo, rg_topo, irg_topo], [np.zeros_like(ori_topo), np.zeros_like(rg_topo), np.zeros_like(irg_topo)], y_range_ratio=3))
        plt.tight_layout()
        plt.savefig(f"plots/irg_topo_L{lattice_size}.pdf", transparent=True)
        plt.show()

def main():
    lattice_size = 64
    
    # Initialize evaluator
    evaluator = RGTransformerEvaluator()
    
    # Load and process data
    rg_tensor, ori_tensor, irg_tensor = evaluator.load_and_process_data(
        rg_path=f"configs/2DU1_L{int(lattice_size / 2)}_RG.pt",
        ori_path=f"configs/2DU1_L{int(lattice_size)}.pt"
    )
    
    # Calculate and plot plaquette means
    evaluator.calculate_plaquette_means(lattice_size=lattice_size)
    
    # Calculate and plot topological charge
    evaluator.calculate_topological_charge(lattice_size=lattice_size)
    
    # print("Difference between original and IRG configurations:")
    # diff = ori_tensor - irg_tensor
    # print(diff.shape)
    # print(diff[10])
    

if __name__ == "__main__":
    main()

# %%
evaluator = RGTransformerEvaluator()

rg_configs = torch.load("configs/2DU1_L32_RG.pt")
rg_tensor = torch.stack(rg_configs)
irg_configs = evaluator.predict(rg_tensor)
print(irg_configs.shape)

irg_plaq_means = [plaq_mean_from_field(config).item() for config in irg_configs]

fig, ax = default_plot()
ax.plot(irg_plaq_means, label='IRG')
ax.set_ylabel('Plaquette Mean', **fs_p)
ax.grid(True)
ax.legend(loc="upper right", **fs_small_p)
ax.set_ylim(auto_ylim([irg_plaq_means], [np.zeros_like(irg_plaq_means)], y_range_ratio=2))
plt.tight_layout()
plt.show()



# %%
