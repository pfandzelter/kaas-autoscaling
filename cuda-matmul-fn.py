#!/usr/bin/env python3

# The fast matrix-multiply code is adapted from: https://numba.readthedocs.io/en/stable/cuda/examples.html#id30

import os
import time
import math
import struct
import pickle
from multiprocessing import shared_memory
import numpy as np
from numba import cuda, float32

if "WORKER_GPU" in os.environ:
    cuda.select_device(int(os.environ["WORKER_GPU"]))

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
# TPB should not be larger than 32 in this example
TPB = 16


@cuda.jit
def device_matmul(A, B, C):  # type: ignore
    """
    Perform matrix multiplication of C = A * B using CUDA shared memory.

    Reference: https://stackoverflow.com/a/64198479/13697228 by @RobertCrovella
    """
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x  # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.0)
    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < A.shape[0] and (tx + i * TPB) < A.shape[1]:
            sA[ty, tx] = A[y, tx + i * TPB]
        if x < B.shape[1] and (ty + i * TPB) < B.shape[0]:
            sB[ty, tx] = B[ty + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp


################
# Wrapper for device_matmul()
# Create a random square matrix and multiply it by itself on GPU.
# for reproducibility, we set a fixed seed for the rng
def square_gpu(mat_h: np.ndarray, N: int) -> float:  # type: ignore
    # N = int(n)

    # mat_h = np.random.default_rng(0).random((N, N))
    sq_h = np.zeros([N, N])

    cuda.pinned(mat_h)
    cuda.pinned(sq_h)

    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(sq_h.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(sq_h.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start = time.perf_counter()

    mat_d = cuda.to_device(mat_h)
    sq_d = cuda.to_device(sq_h)
    device_matmul[blockspergrid, threadsperblock](mat_d, mat_d, sq_d)
    sq_h = sq_d.copy_to_host()

    inner_time = time.perf_counter() - start

    return inner_time


def call(p: bytes) -> bytes:
    shm_name, N = pickle.loads(p)
    # print(f"have received shm_name {shm_name} and N {N}")
    # unpack shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    mat_h = np.ndarray((N, N), dtype=np.float64, buffer=shm.buf)  # type: ignore
    inner_time = square_gpu(mat_h, N)
    shm.close()

    return struct.pack("f", inner_time)


# start a new context on init
# and trigger numba jit
__rng = np.random.default_rng(0)
__mat_h = __rng.random((10, 10), dtype=np.float64)
square_gpu(__mat_h, 10)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        try:
            N = int(input("Enter dimension: "))
        except Exception:
            print("Required argument in stdin: N (int)")
            exit(-1)
    else:
        N = int(sys.argv[1])

    rng = np.random.default_rng(0)
    mat_h = rng.random((N, N), dtype=np.float64)

    inner_time = square_gpu(mat_h, N)

    print(f"@@@ Elapsed inner time (ms): {round(inner_time*1000, 3)}")
