#!/usr/bin/env python3
# Square a random matrix of a given size on the CPU (multi-threaded using numpy)
# The size of the matrix is given as the first argument.

from multiprocessing import shared_memory
import os
import pickle
import socket
import struct
import sys
import time
import typing

import numpy as np

NPY_FILE = ""


def prepare(N: int, copy: int = 0) -> None:
    global NPY_FILE

    N = int(N)
    copy_s = str(copy)

    NPY_FILE = f"test-{N}-{copy_s}.npy"

    try:
        # create a random array
        rng = np.random.default_rng(0)
        arr = rng.random((N, N), dtype=np.float64)

        # save it to a file
        np.save(NPY_FILE, arr)
    except Exception as e:
        print("Error preparing matrix: ", e)
        raise e


def cleanup() -> None:
    try:
        os.remove(NPY_FILE)
    except Exception as e:
        print("Error cleaning up matrix file: ", e)
        raise e


def run_client(N: int) -> typing.Tuple[float, float, float, bool]:
    global NPY_FILE

    if NPY_FILE == "":
        raise RuntimeError("Matrix file not prepared")

    outer_start = time.perf_counter()

    N = int(N)

    # following https://gist.github.com/lsena/a34c08dc385644165c99c12f793154a6#file-numpy_shared_memory-py
    # create a shared memory region
    d_size = int(np.dtype(np.float64).itemsize * np.prod((N, N)))
    shm = shared_memory.SharedMemory(create=True, size=d_size)

    # create a random numpy array on that region
    dst = np.ndarray(shape=(N, N), dtype=np.float64, buffer=shm.buf)  # type: ignore

    # read it from the file into the shared memory
    dst[:] = np.load(NPY_FILE, mmap_mode="r")

    setup_time = time.perf_counter() - outer_start

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect(("localhost", 8081))
        # client.sendall(struct.pack("i", N))
        client.sendall(pickle.dumps((shm.name, N)))
        inner_time_p = client.recv(8)

    cold_start, inner_time = struct.unpack("?f", inner_time_p)

    shm.close()
    shm.unlink()

    outer_time = time.perf_counter() - outer_start

    return (
        round(outer_time * 1000, 3),
        round(float(inner_time) * 1000, 3),
        round(setup_time * 1000, 3),
        cold_start,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        try:
            N = int(input("Enter dimension: "))
        except Exception as ex:
            print("Required argument in stdin: dimension of matrix (int)")
            exit(-1)
    else:
        N = int(sys.argv[1])

    prepare(N)

    outer_time, inner_time, setup_time, cold_start = run_client(N)

    cleanup()

    print(f"Elapsed outer time (ms): {outer_time}")
    print(f"Setup time (ms): {setup_time}")
    print("")
    print(f"@@@ Elapsed inner time (ms): {inner_time}")
    print(f"$$$ Cold start: {'yes' if cold_start else 'no'}")
