import numpy as np
from numba import cuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import random
import string

@cuda.jit
def mass_earch(R, H, N):
    x, y = cuda.grid(2)
    for k in range(N.shape[0]):
        if x<H.shape[0] and N[k][0]==H[x]:
            R[N[k][1], x-N[k][2]]-=1

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
#func = mod.get_function("pi_gpu")
#block_size = (2, 3, 1)
#grid_size = (5, 1)
#start_gpu = time.time()
#qq = np.zeros(1).astype("int32")

H_cuda = H.copy()
R_cuda = R.copy()
n_cuda = m.copy()

# start = time.time()
g = np.int32(2)
#func(cuda.InOut(R),
     #cuda.InOut(Hh),
     #cuda.InOut(m),
     #cuda.InOut(N_len_gpu),
     #cuda.InOut(n_len),
     #grid =grid_size,
     #block=block_size)
#cuda.Context.synchronize()
#print("dvfbfgbghbghbghbb")
start = time.time()
mass_earch(R_cuda, H_cuda, n_cuda)
end = time.time()
print(f"Cuda time = {end-s}")
print(f"R CUda {R_cuda}")

print(f"R {R}")


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
