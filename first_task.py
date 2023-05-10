import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


mod = SourceModule("""
    __global__ 
    void multiplyer(float* a, float* b, float* result, int size)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = tid; i < size; i += stride)
            result[i] = a[i] * b[i];
        
    }
""")


vector_number_one = np.array([1, 1, 1, 2, 2], dtype=np.float32)
vector_number_two = np.array([6, 7, 8, 9, 10], dtype=np.float32)


block_size = 1
grid_size = 1


a_gpu = cuda.mem_alloc(vector_number_one.nbytes)
b_gpu = cuda.mem_alloc(vector_number_two.nbytes)
result_gpu = cuda.mem_alloc(vector_number_one.nbytes)


cuda.memcpy_htod(a_gpu, vector_number_one)
cuda.memcpy_htod(b_gpu, vector_number_two)


multiplyer = mod.get_function("multiplyer")
multiplyer(a_gpu, b_gpu, result_gpu, np.int32(vector_number_one.size), block=(block_size, 1, 1), grid=(grid_size, 1))


result = np.empty_like(vector_number_one)
cuda.memcpy_dtoh(result, result_gpu)


print(result)