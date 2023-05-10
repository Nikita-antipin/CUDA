import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


mod = SourceModule("""
    __global__ 
    void matrix_add(float* a, float* b, float* result, int rows, int cols)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = tid; i < rows * cols; i += stride){
            result[i] = a[i] + b[i];
        }
    }
""")

a = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)
b = np.array([[2, 2, 2], [2, 2, 2]], dtype=np.float32)

rows, cols = a.shape


block_size = 2
grid_size = 1


a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
result_gpu = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

matrix_add = mod.get_function("matrix_add")
matrix_add(a_gpu, b_gpu, result_gpu, np.int32(rows), np.int32(cols), block=(block_size, 1, 1),
           grid=(grid_size, 1))

result = np.empty_like(a)
cuda.memcpy_dtoh(result, result_gpu)

print(result)