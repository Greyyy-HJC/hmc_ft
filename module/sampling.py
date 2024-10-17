# %%
import numpy as np

def metropolis_hastings(target_dist, n_samples, initial_value, proposal_width=1.0):
    """
    Perform Metropolis-Hastings sampling from a target distribution, which is exp(- target_dist)

    Parameters:
    target_dist (callable): The target distribution function to sample from.
    n_samples (int): The number of samples to generate.
    initial_value (float): The initial value for the Markov chain.
    proposal_width (float): The standard deviation of the normal distribution used for proposals.

    Returns:
    list: A list of samples drawn from the target distribution.

    The function uses the Metropolis-Hastings algorithm to generate samples from
    the target distribution. It starts from the initial value and proposes new
    values using a normal distribution centered at the current value. The
    acceptance of new proposals is based on the ratio of target distribution
    values at the proposed and current points.
    """
    samples = [initial_value]
    current = initial_value
    
    for _ in range(n_samples - 1):
        proposal = current + np.random.normal(0, proposal_width)
        
        acceptance_ratio = np.exp(target_dist(current) - target_dist(proposal))
        
        if np.random.random() < acceptance_ratio:
            current = proposal
        
        samples.append(current)
    
    return samples

def sample_initial_conditions(Nsamp, p_term, r_term):
    """
    Sample Nsamp initial conditions (p0, r0) independently from -inf to inf using Markov Chain Monte Carlo,
    with probability proportional to exp(- p_term) and exp(- r_term) respectively,
    where p_term and r_term are functions in the full Hamiltonian.

    Parameters:
        Nsamp (int): Number of samples to generate.
        p_term (function): Function for p0 sampling.
        r_term (function): Function for r0 sampling.

    Returns:
        List of tuples: [(p0_1, r0_1), (p0_2, r0_2), ..., (p0_Nsamp, r0_Nsamp)]
    """

    # Sample p0 using Metropolis-Hastings
    p0_samples = metropolis_hastings(p_term, Nsamp, initial_value=0.0)
    
    # Sample r0 using Metropolis-Hastings
    r0_samples = metropolis_hastings(r_term, Nsamp, initial_value=0.0)
    
    # Combine p0 and r0 samples
    samples = list(zip(p0_samples, r0_samples))

    return samples

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from scipy.integrate import quad

    # Define the target distributions for p and r
    def p_target(p):
        return 0.5 * p**2  # Assuming p_term is p^2/2

    def r_target(r):
        return 0.5 * r**2  # Assuming r_term is r^2/2

    # Calculate normalization constants
    def normalize(func):
        Z, _ = quad(lambda x: np.exp(-func(x)), -np.inf, np.inf)
        return lambda x: np.exp(-func(x)) / Z

    p_normalized = normalize(p_target)
    r_normalized = normalize(r_target)

    # Sample initial conditions
    Nsamp = 10000
    samples = sample_initial_conditions(Nsamp, p_target, r_target)

    # Separate p0 and r0 samples
    p0_samples, r0_samples = zip(*samples)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot p0 distribution
    ax1.hist(p0_samples, bins=30, density=True, alpha=0.3, color='blue', label='Sampled')
    p_range = np.linspace(min(p0_samples), max(p0_samples), 100)
    p_kde = gaussian_kde(p0_samples)
    ax1.plot(p_range, p_kde(p_range), 'r-', label='KDE')
    ax1.plot(p_range, p_normalized(p_range), 'g--', label='Target')
    ax1.set_xlabel('p0')
    ax1.set_ylabel('Density')
    ax1.set_title('p0 Distribution')
    ax1.legend()

    # Plot r0 distribution
    ax2.hist(r0_samples, bins=30, density=True, alpha=0.3, color='blue', label='Sampled')
    r_range = np.linspace(min(r0_samples), max(r0_samples), 100)
    r_kde = gaussian_kde(r0_samples)
    ax2.plot(r_range, r_kde(r_range), 'r-', label='KDE')
    ax2.plot(r_range, r_normalized(r_range), 'g--', label='Target')
    ax2.set_xlabel('r0')
    ax2.set_ylabel('Density')
    ax2.set_title('r0 Distribution')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
# %%
