#!/usr/bin/env python3

# The fast matrix-multiply code is adapted from: https://numba.readthedocs.io/en/stable/cuda/examples.html#id30

import os
import time
import struct
import pickle
from multiprocessing import shared_memory
import numpy as np
import numba

# only used in the GPU version
if "WORKER_GPU" in os.environ:
    # cuda.select_device(int(os.environ["WORKER_GPU"]))
    pass

# only one thread, because we have workers
numba.set_num_threads(1)

import warnings

warnings.simplefilter("ignore", category=UserWarning)


@numba.jit(nopython=True)
def matmul(A, B, C):  # type: ignore
    """Perform square matrix multiplication of C = A * B"""

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp


################
# Create a random square matrix and multiply it by itself on CPU.
# for reproducibility, we set a fixed seed for the rng
def square_gpu(mat_h: np.ndarray, N: int) -> float:  # type: ignore
    # mat_h = np.random.default_rng(0).random((N, N))
    sq_h = np.zeros([N, N])

    start = time.perf_counter()

    matmul(mat_h, mat_h, sq_h)

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

    # print("sending inner time: ", inner_time)

    return struct.pack("f", inner_time)


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
