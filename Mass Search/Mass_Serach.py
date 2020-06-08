import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import random
import string

mod = SourceModule("""
    __global__ void pi_gpu(int *H, int *n,   int *R, int *p)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadCount = gridDim.x * blockDim.x;
    int N = 512;

    for(int i = idx; i < N; i+=threadCount){
    p++;
    }
    }
""")

H_len = 4096*1024
N_len = 400
N_min = 2
N_max = 10


def buildblock(size):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(size))


H = buildblock(N_len)
N = []
for x in range(N_len):
    N.append(buildblock(random.randint(N_min, N_max)))

R = np.zeros((N_len, H_len)).astype("int32")

n = []
print(len(set(H)))
for x in set(H):
    for i, y in enumerate(N):
        for j, k in enumerate(y):
            if x == k:
                n.append((ord(x), i, j))

m = np.array(n)
for i, n_len in enumerate(N):
    R[i] = len(n_len)
H = np.array([ord(x) for x in H]).astype("int32")
func = mod.get_function("pi_gpu")
block_size = (256, 1, 1)
grid_size = (int(256*256 / (128*block_size[0])), 1)
start_gpu = time.time()
qq = np.zeros(1).astype("int32")

#func(cuda.InOut(H), cuda.InOut(m),  cuda.InOut(R), cuda.Out(qq),  block=block_size, grid=grid_size)
#H = np.array([ord(x) for x in H]).astype("int32")

def masssearch():
    for i, x in enumerate(H):
        for k in n:
            if x == k[0]:
                R[k[1], i - k[2]] -= 1
    return R


start = time.time()
new_R = masssearch()
time.sleep(0.1)
end = time.time()
print((end-start)*1000)