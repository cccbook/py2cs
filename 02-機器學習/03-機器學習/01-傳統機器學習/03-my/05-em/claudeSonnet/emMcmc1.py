import numpy as np
from scipy.stats import norm, invgamma

def generate_data(n_samples, means, stds):
    n_components = len(means)
    z = np.random.choice(n_components, size=n_samples)
    x = np.random.normal(np.array(means)[z], np.array(stds)[z])
    return x, z

def gibbs_sampler(x, k, mu, sigma, alpha, n_iterations):
    n = len(x)
    z = np.random.randint(k, size=n)
    
    for _ in range(n_iterations):
        # Sample z
        probs = np.array([alpha[j] * norm.pdf(x, mu[j], sigma[j]) for j in range(k)])
        probs /= probs.sum(axis=0)
        z = np.array([np.random.choice(k, p=probs[:, i]) for i in range(n)])
        
        # Sample alpha
        alpha = np.random.dirichlet(np.array([1 + (z == j).sum() for j in range(k)]))
        
        # Sample mu and sigma
        for j in range(k):
            nj = (z == j).sum()
            if nj > 0:
                xj = x[z == j]
                mu[j] = np.random.normal((xj.sum() / sigma[j]**2) / (nj / sigma[j]**2), 
                                         1 / np.sqrt(nj / sigma[j]**2))
                sigma[j] = np.sqrt(invgamma.rvs(nj/2, scale=((xj - mu[j])**2).sum()/2))
    
    return z, alpha, mu, sigma

def mcmc_em(x, k, n_iterations, n_gibbs):
    # Initialize parameters
    mu = np.random.normal(np.mean(x), np.std(x), size=k)
    sigma = np.random.uniform(0, np.std(x), size=k)
    alpha = np.ones(k) / k
    
    for _ in range(n_iterations):
        # E-step: Gibbs sampling
        z, alpha, mu, sigma = gibbs_sampler(x, k, mu, sigma, alpha, n_gibbs)
        
        # M-step: Update parameters
        for j in range(k):
            nj = (z == j).sum()
            if nj > 0:
                xj = x[z == j]
                mu[j] = xj.mean()
                sigma[j] = np.sqrt(((xj - mu[j])**2).sum() / nj)
        
        alpha = np.array([(z == j).sum() for j in range(k)]) / len(x)
    
    return alpha, mu, sigma

# Generate synthetic data
np.random.seed(42)
true_means = [-2, 2, 6]
true_stds = [0.5, 1, 0.8]
x, true_z = generate_data(1000, true_means, true_stds)

# Run MCMC-EM algorithm
k = 3  # number of components
n_iterations = 50
n_gibbs = 100
alpha, mu, sigma = mcmc_em(x, k, n_iterations, n_gibbs)

print("Estimated mixture weights (alpha):", alpha)
print("Estimated means (mu):", mu)
print("Estimated standard deviations (sigma):", sigma)

# Compare with true parameters
print("\nTrue means:", true_means)
print("True standard deviations:", true_stds)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(x, bins=50, density=True, alpha=0.7, color='skyblue')
xmin, xmax = plt.xlim()
x_range = np.linspace(xmin, xmax, 1000)
for j in range(k):
    plt.plot(x_range, alpha[j] * norm.pdf(x_range, mu[j], sigma[j]), 
             label=f'Component {j+1}')
plt.plot(x_range, sum(alpha[j] * norm.pdf(x_range, mu[j], sigma[j]) for j in range(k)), 
         'r--', linewidth=2, label='Mixture')
plt.legend()
plt.title('MCMC-EM Bayesian Mixture Model')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()