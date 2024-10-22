import matplotlib.pyplot as plt
import numpy as np

def plot_results(actions, topological_charges, hamiltonians, title_suffix=""):
    plt.figure(figsize=(24, 10))

    plt.subplot(231)
    plt.plot(actions)
    plt.title(f'Action vs. Iteration {title_suffix}')
    plt.xlabel('Iteration')
    plt.ylabel('Action')

    plt.subplot(232)
    plt.plot(hamiltonians)
    plt.title(f'Hamiltonian vs. Iteration {title_suffix}')
    plt.xlabel('Iteration')
    plt.ylabel('Hamiltonian')
    plt.axhline(y=np.mean(hamiltonians), color='r', linestyle='--', label='Mean Hamiltonian')
    plt.legend()

    plt.subplot(233)
    plt.plot(topological_charges)
    plt.title(f'Topological Charge vs. Iteration {title_suffix}')
    plt.xlabel('Iteration')
    plt.ylabel('Topological Charge')

    plt.subplot(234)
    plt.hist(actions, bins=30, alpha=0.7, label='Action Histogram')
    plt.title(f'Action Distribution {title_suffix}')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(235)
    plt.hist(hamiltonians, bins=30, alpha=0.7, label='Hamiltonian Histogram')
    plt.title(f'Hamiltonian Distribution {title_suffix}')
    plt.xlabel('Hamiltonian')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(236)
    plt.hist(topological_charges, bins=30, alpha=0.7, label='Topological Charge Histogram')
    plt.title(f'Topological Charge Distribution {title_suffix}')
    plt.xlabel('Topological Charge')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()