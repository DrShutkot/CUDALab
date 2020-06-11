import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import random
import string

mod = SourceModule("""
    __global__ void pi_gpu(int *R, int *H, int *n, int *len_N, int *len_n)
    {
    int N = len_N[0];
    //int idx = threadIdx.x* + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num = n[idx];
    int n1 = n[idx+1];
    int k = n[idx+2];
    for(int i=0; i<3;i++){
    if(H[i]==num){
    R[n1*2+i-k]=R[n1*2+i-k]-1;
    }
    }
    //atomicAdd(vb, *R);
    }
""")

H_len = 3
N_len = 2
N_min = 2
N_max = 2

N_len = np.int32(N_len)


def buildblock(size):
    return ''.join(random.choice(string.ascii_letters) for _ in range(size))


# H = buildblock(H_len)
H = "abc"
N = ['ab', 'bc']
# N = []
# for x in range(N_len):
# N.append(buildblock(random.randint(N_min, N_max)))
print("H")
print(H)
print("N")
print(N)
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
block_size = (2, 3, 1)
grid_size = (5, 1)
start_gpu = time.time()
qq = np.zeros(1).astype("int32")

Hh = H.flatten()
print(Hh)
Mm = m.flatten()
print(Mm)
Rr = R.flatten()
print(Rr)
Nn = np.zeros(1).astype("int32").flatten() * N_len
Nn1 = np.zeros(1).astype("int32").flatten() * H_len
xx, yy = m.shape
n_len = np.ones(1).astype('int32') * xx * yy
N_len_gpu = np.ones(1).astype('int32') * N_len
vb = np.zeros(R.shape)
# start = time.time()
g = np.int32(2)
func(cuda.InOut(R),
     cuda.InOut(Hh),
     cuda.InOut(m),
     cuda.InOut(N_len_gpu),
     cuda.InOut(n_len),
     grid =grid_size,
     block=block_size)
cuda.Context.synchronize()
print("dvfbfgbghbghbghbb")
print(R)
print(Hh)
print(N_len_gpu)
print("qqqqqqqq")
print(m)


# end = time.time()
# print("GPU")
# time.sleep(0.1)
# print((end-start)*1000)
# H = np.array([ord(x) for x in H]).astype("int32")

def masssearch():
    for i, x in enumerate(H):
        for k in n:
            if x == k[0]:
                R[k[1], i - k[2]] -= 1
    return R


start = time.time()
print(R)
new_R = masssearch()
print(new_R)
time.sleep(0.1)
end = time.time()
print((end - start) * 1000)