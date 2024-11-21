# %%
import numpy as np
import torch
from scipy.sparse import coo_matrix

# size of the sparse matrix
N = int(1e9)  # matrix size
density = 1e-6  # sparsity

# actual number of non-zero elements
nnz = int(N * density)

# generate a sparse matrix randomly (COO format)
rows = np.random.randint(0, N, size=nnz)
cols = np.random.randint(0, N, size=nnz)
data = np.random.rand(nnz)

A_scipy = coo_matrix((data, (rows, cols)), shape=(N, N))

print("Sparse matrix shape:", A_scipy.shape)
print(f"Sparse matrix memory usage (approx): {A_scipy.data.nbytes + A_scipy.row.nbytes + A_scipy.col.nbytes} bytes")


# %%
def estimate_logdet_stochastic(A, n_samples=100):
    """Estimate logdet using stochastic trace estimation"""
    N = A.shape[0]
    total = 0
    for _ in range(n_samples):
        # Generate random vector with ±1
        v = np.random.choice([-1, 1], size=N)
        # Solve Ax = v
        try:
            x = np.zeros_like(v)  # Initialize solution vector
            # Use iterative solver for memory efficiency
            from scipy.sparse.linalg import bicgstab
            x, _ = bicgstab(A, v, maxiter=100)
            total += np.sum(v * x)
        except Exception as e:
            print(f"Solver error: {e}")
            return None
    return np.log(total / n_samples)

def estimate_logdet_block(A, block_size=1000):
    """Estimate logdet using block-wise computation"""
    try:
        N = A.shape[0]
        total_logdet = 0
        for i in range(0, N, block_size):
            end_idx = min(i + block_size, N)
            block = A[i:end_idx, i:end_idx].tocsc()
            # Use eigenvalues for smaller block
            from scipy.sparse.linalg import eigsh
            eigenvals, _ = eigsh(block, k=min(block_size, 10))
            total_logdet += np.sum(np.log(np.abs(eigenvals)))
        return total_logdet
    except Exception as e:
        print(f"Block computation error: {e}")
        return None

# Try alternative methods
print("\nTrying alternative methods:")
logdet_stochastic = estimate_logdet_stochastic(A_scipy)
if logdet_stochastic is not None:
    print(f"Stochastic Estimate of Log Determinant: {logdet_stochastic}")

logdet_block = estimate_logdet_block(A_scipy)
if logdet_block is not None:
    print(f"Block-wise Estimate of Log Determinant: {logdet_block}")

# %%
