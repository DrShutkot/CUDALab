import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import random
import string


H_len = 512
N_len = 256
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
block_size = (32, 1, 1)
grid_size = (int(4096/ (128*block_size[0])), 1)
start_gpu = time.time()
qq = np.zeros(1).astype("int32")
nn = np.zeros(1).astype("int32")*len(n)
hh = np.zeros(1).astype("int32")*H_len

Hh = H.flatten()
Mm = m.flatten()
Rr = R.flatten()
Nn = np.zeros(1).astype("int32").flatten()*N_len
point = np.zeros(1).astype("int32")
mod = SourceModule("""
    __global__ void pi_gpu(int *H, int *H_len, int *N, int *n_len, int *n, int *R)
    {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
     for(int j=idx; j<*n_len; j++){
       for (int i=idy; i<*H_len; i++){
       int v = *n_len;
       if (H[i]==n[j*v]){
       int ix=n[j*v+1];
       int iy=n[j*v+2];
       int N_r = *N;
       
       R[ix*N_r+iy]--; # вычитаем, если совпало
       }
       }
       }
    
    }
""")



start = time.time()
func(cuda.InOut(Hh), cuda.InOut(hh), cuda.InOut(Nn), cuda.InOut(nn), cuda.InOut(Mm), cuda.InOut(Rr),
     block=block_size, grid=grid_size)
cuda.Context.synchronize()
end = time.time()
print("GPU")
time.sleep(0.1)
print((end-start)*1000)
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
