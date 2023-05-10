import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


mod = SourceModule("""
    __global__ 
    void matrix_vector_multiply(float* matrix, float* vector, float* result, int rows, int cols)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int row = tid; row < rows; row += stride)
        {
            float sum = 0.0f;
            for (int col = 0; col < cols; col++) {
                sum += vector[col] * matrix[col + row * cols];
            }
            result[row] = sum;
        }
        
    }
""")


matrix = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
vector = np.array([1, 1, 1], dtype=np.float32)


rows, cols = matrix.shape


matrix_gpu = cuda.mem_alloc(matrix.nbytes)
vector_gpu = cuda.mem_alloc(vector.nbytes)
result_gpu = cuda.mem_alloc(rows * np.float32().itemsize)


cuda.memcpy_htod(matrix_gpu, matrix)
cuda.memcpy_htod(vector_gpu, vector)


block_size = 2
grid_size = 1


matrix_vector_multiply = mod.get_function("matrix_vector_multiply")
matrix_vector_multiply(matrix_gpu, vector_gpu, result_gpu, np.int32(rows), np.int32(cols),
                      block=(block_size, 1, 1), grid=(grid_size, 1))


result = np.array([0 for i in range(rows)], dtype=np.float32)
cuda.memcpy_dtoh(result, result_gpu)


print(result)