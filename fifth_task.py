import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #define TILE_SIZE 16

    __global__ void matrix_multiply(float* a, float* b, float* result, int m, int n, int k)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        float val = 0.0f;
        for (int i = 0; i < n; i += TILE_SIZE)
        {
            __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
            __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

            shared_a[threadIdx.y][threadIdx.x] = a[row * n + i + threadIdx.x];
            shared_b[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * k + col];

            __syncthreads();

            for (int j = 0; j < TILE_SIZE; j++)
            {
                val += shared_a[threadIdx.y][j] * shared_b[j][threadIdx.x];
            }

            __syncthreads();
        }

        if (row < m && col < k)
        {
            result[row * k + col] = val;
        }
    }
""")


a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)

m, n = a.shape
_, k = b.shape

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
result_gpu = cuda.mem_alloc(m * k * np.float32().itemsize)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

block_size = 16
grid_size = 1

matrix_multiply = mod.get_function("matrix_multiply")
matrix_multiply(a_gpu, b_gpu, result_gpu, np.int32(m), np.int32(n), np.int32(k), block=(block_size, block_size, 1), grid=(grid_size, 1))


result = np.empty((m, k), dtype=np.float32)
cuda.memcpy_dtoh(result, result_gpu)

print(result)