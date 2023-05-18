import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule



kernel_code = """
    #define TILE_SIZE 16

    __global__ void transpose(float* input, float* output, int m, int n)
    {
        __shared__ float tile[TILE_SIZE][TILE_SIZE+1];

        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;

        if (row < m && col < n)
        {
            tile[threadIdx.y][threadIdx.x] = input[row * n + col];
        }

        __syncthreads();

        col = blockIdx.y * TILE_SIZE + threadIdx.x;
        row = blockIdx.x * TILE_SIZE + threadIdx.y;

        if (row < n && col < m)
        {
            output[row * m + col] = tile[threadIdx.x][threadIdx.y];
        }
    }
"""

m, n = 5, 5

input_matrix = np.arange(m * n, dtype=np.float32).reshape((m, n))

input_gpu = cuda.mem_alloc(input_matrix.nbytes)
output_gpu = cuda.mem_alloc(input_matrix.nbytes)

cuda.memcpy_htod(input_gpu, input_matrix)


module = SourceModule(kernel_code)

transpose_kernel = module.get_function("transpose")

block_size = (16, 16, 1)
grid_size = (
    1,
    1,
    1
)


transpose_kernel(input_gpu, output_gpu, np.int32(m), np.int32(n),
                 block=block_size, grid=grid_size)

output_matrix = np.empty_like(input_matrix)
cuda.memcpy_dtoh(output_matrix, output_gpu)

print("Input Matrix:")
print(input_matrix)

print("\nTransposed Matrix:")
print(output_matrix)