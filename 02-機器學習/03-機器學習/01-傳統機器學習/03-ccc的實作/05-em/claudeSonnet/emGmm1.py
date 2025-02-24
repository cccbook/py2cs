import numpy as np
from scipy.stats import multivariate_normal

def initialize_parameters(X, n_components):
    n_samples, n_features = X.shape
    weights = np.ones(n_components) / n_components
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covariances = [np.eye(n_features) for _ in range(n_components)]
    return weights, means, covariances

def e_step(X, weights, means, covariances):
    n_samples, n_components = len(X), len(weights)
    responsibilities = np.zeros((n_samples, n_components))
    
    for k in range(n_components):
        responsibilities[:, k] = weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
    
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

def m_step(X, responsibilities):
    n_samples, n_features = X.shape
    n_components = responsibilities.shape[1]
    
    weights = responsibilities.sum(axis=0) / n_samples
    means = np.dot(responsibilities.T, X) / responsibilities.sum(axis=0)[:, np.newaxis]
    covariances = []
    
    for k in range(n_components):
        diff = X - means[k]
        cov = np.dot(responsibilities[:, k] * diff.T, diff) / responsibilities[:, k].sum()
        covariances.append(cov)
    
    return weights, means, covariances

def em_gmm(X, n_components, n_iterations=100, tol=1e-6):
    weights, means, covariances = initialize_parameters(X, n_components)
    
    for _ in range(n_iterations):
        old_log_likelihood = np.sum([np.log(w * multivariate_normal.pdf(X, mean=m, cov=c)).sum() 
                                     for w, m, c in zip(weights, means, covariances)])
        
        # E-step
        responsibilities = e_step(X, weights, means, covariances)
        
        # M-step
        weights, means, covariances = m_step(X, responsibilities)
        
        # Check for convergence
        new_log_likelihood = np.sum([np.log(w * multivariate_normal.pdf(X, mean=m, cov=c)).sum() 
                                     for w, m, c in zip(weights, means, covariances)])
        if np.abs(new_log_likelihood - old_log_likelihood) < tol:
            break
    
    return weights, means, covariances

# Example usage
np.random.seed(42)
X = np.concatenate([
    np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100),
    np.random.multivariate_normal([5, 5], [[1.5, 0], [0, 1.5]], 100)
])

n_components = 2
weights, means, covariances = em_gmm(X, n_components)

print("Estimated weights:", weights)
print("Estimated means:", means)
print("Estimated covariances:", covariances)