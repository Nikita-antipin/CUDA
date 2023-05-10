import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


mod = SourceModule("""
    __global__ 
    void matrix_multiply(float* matrix1, float* matrix2, float* result, int m, int n, int k)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int col_in_matrix2 = 0; col_in_matrix2 < k; col_in_matrix2++){
            for (int row_in_matrix1 = tid; row_in_matrix1 < m; row_in_matrix1 += stride)
            {
                float sum = 0.0f;
                for (int col_in_matrix1 = 0; col_in_matrix1 < n; col_in_matrix1++) {
                    sum += matrix2[col_in_matrix2 + col_in_matrix1 * m] * matrix1[col_in_matrix1 + row_in_matrix1 * n];
                }
                result[col_in_matrix2 + row_in_matrix1 * n] = sum;
            }
        }
    }
""")



a = np.array([[2, 2, 3], [4, 5, 6]], dtype=np.float32)
b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)


m, n = a.shape
_, k = b.shape


a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
result_gpu = cuda.mem_alloc(m * k * np.float32().itemsize)


cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)


block_size = 2
grid_size = 1


matrix_multiply = mod.get_function("matrix_multiply")
matrix_multiply(a_gpu, b_gpu, result_gpu, np.int32(m), np.int32(n), np.int32(k), block=(block_size, 1, 1), grid=(grid_size, 1))


result = np.empty((m, k), dtype=np.float32)
cuda.memcpy_dtoh(result, result_gpu)


print(result)