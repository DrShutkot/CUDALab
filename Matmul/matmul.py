import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
from string import Template
import time

kernel_code = Template("""
__global__ void matmul(float* a, float* b, float* c)
{
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int th_x = threadIdx.x;
    int th_y = threadIdx.y;
    float sum = 0.f;
    int index_a = ${N}*${BLOCK_SIZE} * block_y + ${N} * th_y;
    int index_b = ${BLOCK_SIZE} * block_x + th_x;
    for (int k = 0; k < ${N}; k++)
    {
    sum += a[index_a + k] * b[index_b + k*${N}];
    }
    int index_c = ${N}*${BLOCK_SIZE} * block_y + ${BLOCK_SIZE}*block_x;

    c[index_c + ${N} * th_y + th_x] = sum;
    
}
""")


N = 1024 # размер матрица 
BLOCK_SIZE = 32 # размер блока 


def create_matrix(N=2, random=False):
    
    """
    create_matrix(N)
    return a, b (ndarray, float32, with shape(N*N, 1)
    """
    if random == True:
        a = np.random.randn((N*N)).astype(np.float32)
        b = np.random.randn((N*N)).astype(np.float32) 
        c = np.zeros((N*N)).astype(np.float)
    else:
        a = np.array([x for x in range(N*N)]).astype(np.float32)
        b = np.array([x for x in range(N*N)]).astype(np.float32)
        c = np.zeros((N*N)).astype(np.float)
    
    return a, b, c
    
    
def matmul_cpu( a, b, alg = "auto" ):
    """
    Умножение матриц 
     c = matmul(a, b, alg="auto")
     a, b, - array
     alg - режим вычисления
     return c - ndarray, shape (N*N,)
    """
    if alg == "auto":
        c = a.reshape(N,N).dot(b.reshape(N, N))
    else:
        c = np.zeros(N*N)
        for x in range(N):
            for y in range(N):
                for i in range(N):
                    c[x*N+y] += a[x*N + i] * b[i * N + y]
                    
    return c


a, b, c_gpu = create_matrix(N, random = True)

compil = compiler.SourceModule(kernel_code.substitute(N=N, BLOCK_SIZE=BLOCK_SIZE)) # конструктор
matmul = compil.get_function("matmul") # добавляем функцию 

dimGrid = N // BLOCK_SIZE
dimBlock = BLOCK_SIZE

start = driver.Event()
stop = driver.Event()

# GPU 
start.record()
matmul( driver.In(a), driver.In(b), driver.Out(c_gpu),  block=(dimBlock, dimBlock, 1), grid=(dimGrid, dimGrid))
stop.record()
stop.synchronize()
time_g = stop.time_since(start)
print(f"Время работы GPU:{time_g}")


start_cpu = time.time()
c = matmul_cpu(a, b, alg="b")
end_cpu  = time.time()
time_cpu = end_cpu - start_cpu
print(f"Время работы CPU в ручном режиме: {time_cpu*1000}")

"""
#Не командный замер работы функций для матриц меньших размерностей и вычисления методом np.dot() (так как время сильно меньше секунды,
#обычный модуль time не подходит
import timeit
start_cpu_n = timeit.default_timer()
c = matmul_cpu(a, b)
end_cpu_n  = timeit.default_timer()
time_cpu_n = end_cpu_n - start_cpu_n
print(f"Время работы CPU np.dot(): {time_cpu_n*1000}")
"""

#print(np.all(c, c_gpu)) #точное сравнение 
print(np.allclose(c, c_gpu, 0.001)) # c допуском
