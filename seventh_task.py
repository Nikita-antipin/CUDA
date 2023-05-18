import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


kernel_code = """
    #define BLOCK_SIZE 16

    __global__ void scalar_multiply(float* matrix_a, float* matrix_b, int m, int n)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < n)
        {
            __shared__ float shared_a[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float shared_b[BLOCK_SIZE][BLOCK_SIZE];

            shared_a[threadIdx.y][threadIdx.x] = matrix_a[row * n + col];
            shared_b[threadIdx.y][threadIdx.x] = matrix_b[row * n + col];
            __syncthreads();

            shared_a[threadIdx.y][threadIdx.x] *= shared_b[threadIdx.y][threadIdx.x];
            __syncthreads();

            // Store the result back to global memory
            matrix_a[row * n + col] = shared_a[threadIdx.y][threadIdx.x];
        }
    }
"""


m, n = 4, 4

matrix_a = np.random.rand(m, n).astype(np.float32)
matrix_b = np.random.rand(m, n).astype(np.float32)

matrix_a_gpu = cuda.mem_alloc(matrix_a.nbytes)
matrix_b_gpu = cuda.mem_alloc(matrix_b.nbytes)

cuda.memcpy_htod(matrix_a_gpu, matrix_a)
cuda.memcpy_htod(matrix_b_gpu, matrix_b)

module = SourceModule(kernel_code)

scalar_multiply_kernel = module.get_function("scalar_multiply")

block_size = (16, 16, 1)
grid_size = (
    1,
    1,
    1
)

scalar_multiply_kernel(matrix_a_gpu, matrix_b_gpu, np.int32(m), np.int32(n),
                       block=block_size, grid=grid_size)

result_matrix = np.empty_like(matrix_a)
cuda.memcpy_dtoh(result_matrix, matrix_a_gpu)

print("Matrix A:")
print(matrix_a)

print("\nMatrix B:")
print(matrix_b)

print("\nScalar Multiplied Matrix (A * B):")
print(result_matrix)