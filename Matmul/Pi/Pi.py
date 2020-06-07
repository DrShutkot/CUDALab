import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

N = 4096*4096
x = np.random.random(N)
y = np.random.random(N)
point = np.zeros(1).astype("int32")

mod = SourceModule("""
    __global__ void pi_gpu(double *x, double *y, int *point)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadCount = gridDim.x * blockDim.x;
    int N = 4096*4096;
    int points = 0;

    for(int i = idx; i < N; i+=threadCount){
    double values = x[i]*x[i]+y[i]*y[i];
       if (values < 1)
       {
         points++;
        }
    }

      atomicAdd(point, points);
    }
"""
                   )

func = mod.get_function("pi_gpu")
block_size = (256, 1, 1)
grid_size = (int(N / (128*block_size[0])), 1)
start_gpu = time.time()
func(cuda.InOut(x), cuda.InOut(y), cuda.Out(point),  block=block_size, grid=grid_size)
cuda.Context.synchronize()
#time.sleep(0.1)
end_gpu = time.time()
time_gpu = end_gpu - start_gpu
print("GPU")
print(point*4/N)
print(f"Time gpu {time_gpu*1000}")

def pi_cpu(x, y):
    count = x**2+y**2 # через numpy.where
    res = np.where((count < 1), 1, 0)
    return np.sum(res)*4/N

def pi_cpu2(x, y):
    count = [1 for i, j in zip(x,y) if i**2+j**2<1 ] # генератор списков
    return np.sum(count)*4/N

start = time.time()
pi = pi_cpu(x, y)
#time.sleep(0.1)
end = time.time()

time_cpu = (end-start)
print("CPU")
print(pi)
print(f"Time CPU {time_cpu*1000}")
print(f"Ускорение {time_cpu/time_gpu}")

start = time.time()
pi = pi_cpu2(x, y)
#time.sleep(0.1)
end = time.time()

time_cpu = (end-start)
print("CPU_2")
print(pi)
print(f"Time CPU {time_cpu*1000}")
print(f"Ускорение {time_cpu/time_gpu}")