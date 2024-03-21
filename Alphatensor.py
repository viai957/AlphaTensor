import torch
import time

def tensor_decomposed_matrix_multiplication_torch(A, B, u, v, w):
    """
    Computes the matrix product C = AB using a meta-algorithm parameterized by vectors u, v, w with PyTorch.

    :param A: Tensor of size n x n
    :param B: Tensor of size n x n
    :param u: List of n^2-length vectors u^(r)
    :param v: List of n^2-length vectors v^(r)
    :param w: List of n^2-length vectors w^(r)
    :return: Tensor C of size n x n, result of AB
    """
    n = A.size(0)
    R = len(u)  # Number of rank-1 matrices to sum
    C = torch.zeros((n, n), device=A.device)  # Initialize the output matrix C
    
    for r in range(R):
        # Compute the r-th rank-1 matrix
        m_r = (A.view(-1) * u[r]).sum() * (B.view(-1) * v[r]).sum()
        
        # Accumulate the weighted rank-1 matrices to form C
        C += w[r].view(n, n) * m_r
    
    return C

# Example usage:
n = 3  # Size of the matrices
R = 2  # Number of decompositions

# Initialize matrices A and B with random values for the example, convert them to PyTorch tensors
A_torch = torch.rand((n, n))
B_torch = torch.rand((n, n))

# Initialize parameters u, v, and w with random values for the example, convert them to PyTorch tensors
u_torch = [torch.rand((n**2,)) for _ in range(R)]
v_torch = [torch.rand((n**2,)) for _ in range(R)]
w_torch = [torch.rand((n**2,)) for _ in range(R)]

# Compute the matrix product C using PyTorch
C_torch = tensor_decomposed_matrix_multiplication_torch(A_torch, B_torch, u_torch, v_torch, w_torch)
print(C_torch)


# Function to perform timing analysis
def timing_analysis(n_values, R, device):
    timings = {}
    for n in n_values:
        A = torch.rand((n, n), device=device)
        B = torch.rand((n, n), device=device)
        u = [torch.rand((n**2,), device=device) for _ in range(R)]
        v = [torch.rand((n**2,), device=device) for _ in range(R)]
        w = [torch.rand((n**2,), device=device) for _ in range(R)]
        
        # Warm-up run for CUDA
        if device.type == 'cuda':
            tensor_decomposed_matrix_multiplication_torch(A, B, u, v, w)
            torch.cuda.synchronize(device)
        
        start_time = time.perf_counter()
        tensor_decomposed_matrix_multiplication_torch(A, B, u, v, w)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)  # Wait for CUDA to finish
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        timings[n] = elapsed_time
        print(f"Size: {n}, Time: {elapsed_time:.6f} seconds")

    return timings

# Example usage of the timing analysis function
n_values = [100, 200, 300, 400, 500]  # Different matrix sizes to test
R = 2  # Number of decompositions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timings = timing_analysis(n_values, R, device)

# Print the timings
for size, time_taken in timings.items():
    print(f"Matrix size: {size}x{size}, Time taken: {time_taken:.6f} seconds")

# Function to compare the performance of custom and default matrix multiplication
def compare_matrix_multiplication_performance(n, R):
    # Generate random matrices and vectors for decomposition
    A = torch.rand(n, n, device=device)
    B = torch.rand(n, n, device=device)
    u = [torch.rand(n**2, device=device) for _ in range(R)]
    v = [torch.rand(n**2, device=device) for _ in range(R)]
    w = [torch.rand(n**2, device=device) for _ in range(R)]

    # Time the tensor decomposition matrix multiplication
    start_time = time.time()
    tensor_decomposed_matrix_multiplication_torch(A, B, u, v, w)
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for CUDA operations to finish
    tensor_decomp_time = time.time() - start_time

    # Time PyTorch's built-in matrix multiplication
    start_time = time.time()
    torch.matmul(A, B)
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for CUDA operations to finish
    default_time = time.time() - start_time

    return tensor_decomp_time, default_time

# Example usage with matrix size 512x512 and R=10
n = 512  # Size of the matrices
R = 10   # Number of decompositions

# Compare the performance
tensor_decomp_time, default_time = compare_matrix_multiplication_performance(n, R)
print(f"Tensor Decomposed Matrix Multiplication Time: {tensor_decomp_time:.6f} seconds")
print(f"PyTorch Default Matrix Multiplication Time: {default_time:.6f} seconds")
