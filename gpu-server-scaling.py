#!/usr/bin/env python3

import argparse
import os
import time
import importlib
import multiprocessing as mp
from multiprocessing.connection import Connection as MultiprocessingConnection
import socketserver
import signal
import struct
import sys
import threading
import typing
import warnings


def _recv_function(
    recver: MultiprocessingConnection, function_module: str, cuda_device: int
) -> None:
    os.environ["WORKER_GPU"] = str(cuda_device)

    try:
        fn = importlib.import_module(function_module)
    except Exception as e:
        raise ImportError(
            f"no file {function_module}.py was found or there was an error importing it -- make sure that you only run this container with a custom function"
        ) from e

    try:
        getattr(fn, "call")
    except Exception as e:
        raise ImportError(
            f"{function_module}.py is present but method call() could not be found"
        ) from e

    TIMEOUT = 60

    while True:
        if recver.poll(TIMEOUT):
            msg = recver.recv()
            try:
                rsp = fn.call(msg)
            except Exception as e:
                print(f"Error: {e}")
                rsp = b""
            recver.send(rsp)
            continue
        break

    print("Time out! Stopping this worker process...", file=sys.stderr)

    exit(0)


#######################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KaaS GPU Server")
    parser.add_argument(
        "function",
        type=str,
        help="function to import. use a relative or full path to the Python module. the module should expose a function of the signature 'call(p: bytes) -> bytes'.",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port to start TCP socket server on",
        default=8080,
    )
    parser.add_argument(
        "--num-gpus",
        "-g",
        type=int,
        help="number of GPUs to use (currently assumes that GPU IDs are numerical, starting from 0)",
    )
    parser.add_argument(
        "--max-req-per-gpu",
        "-m",
        type=int,
        help="number of tasks allowed to run on a single GPU before it is considered 'full' and a new GPU is allocated",
    )
    parser.add_argument(
        "--message-size",
        type=int,
        default=1024,
        help="message size buffer to accept for incoming messages",
    )

    args = parser.parse_args()

    function = args.function
    port = args.port
    available_gpus = args.num_gpus
    message_size = args.message_size
    max_req_per_gpu = args.max_req_per_gpu

    print(f"Starting autoscaling server for function {function}")

    # boot a few backends
    servers = []
    pipes = []

    def _boot_processes(gpu: int) -> None:
        for i in range(max_req_per_gpu):
            recver, sender = mp.Pipe()

            p = mp.Process(target=_recv_function, args=(recver, function, gpu))
            p.start()

            servers.append(p)
            pipes.append(sender)

    def _stop_processes(signum: int, frame: typing.Optional[typing.Any]) -> None:
        print(f"Recved signal {signum}, stopping processes", end="", file=sys.stderr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(servers)):
                servers[i].join(1)
                servers[i].kill()
                pipes[i].close()
                print(".", end="", file=sys.stderr)
        print("\n", end="", file=sys.stderr)
        print(f"Exiting...", file=sys.stderr)
        exit(0)

    signal.signal(signal.SIGINT, _stop_processes)
    signal.signal(signal.SIGTERM, _stop_processes)

    print("Server ready!")

    # stores the number of in-flight requests per worker
    gpu_load: typing.List[int] = []
    worker_load: typing.Dict[int, typing.List[int]] = {}
    lock = threading.Lock()

    class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            global gpu_load
            global worker_load
            global lock
            with lock:
                cold_start = False
                # find the gpu with the least in-flight requests
                if len(gpu_load) > 0:
                    gpu_to_use = gpu_load.index(min(gpu_load))
                    avail_worker = worker_load[gpu_to_use].index(
                        min(worker_load[gpu_to_use])
                    )
                    worker_to_use = gpu_to_use * max_req_per_gpu + avail_worker

                # if all workers are full, boot new ones on the next GPU
                # only boot a new worker if there are still GPUs available
                # in theory, we would otherwise just need to wait
                # but we will not try this out here
                if len(gpu_load) == 0 or worker_load[gpu_to_use][avail_worker] >= 1:
                    if len(servers) >= available_gpus * max_req_per_gpu:
                        print(f"@@@ ERROR all workers are full at {time.time()}")
                        _ = self.request.recv(message_size)
                        self.request.sendall(struct.pack("?f", False, 0.0))
                        return

                    next_gpu = len(gpu_load)

                    # start new workers on the next GPU
                    _boot_processes(next_gpu)
                    print(
                        f"@@@ Booted {max_req_per_gpu} new workers on GPU {next_gpu} at {time.time()}"
                    )

                    # add a new entry to the gpu load list
                    gpu_load.append(0)
                    worker_load[next_gpu] = [0] * max_req_per_gpu

                    avail_worker = 0
                    worker_to_use = next_gpu * max_req_per_gpu + avail_worker
                    gpu_to_use = next_gpu
                    cold_start = True

                # increment the in-flight count for the worker
                gpu_load[gpu_to_use] = gpu_load[gpu_to_use] + 1
                worker_load[gpu_to_use][avail_worker] = 1
                print(f"Using worker {worker_to_use} on GPU {gpu_to_use}")

            msg = self.request.recv(message_size)
            pipes[worker_to_use].send(msg)
            rsp = pipes[worker_to_use].recv()
            inner_time = struct.unpack("f", rsp)[0]
            self.request.sendall(struct.pack("?f", cold_start, inner_time))

            with lock:
                # decrement the in-flight count for the worker to release resource
                gpu_load[gpu_to_use] = gpu_load[gpu_to_use] - 1
                worker_load[gpu_to_use][avail_worker] = 0
                print(f"Released worker {worker_to_use} on GPU {gpu_to_use}")

    class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        pass

    ThreadedTCPServer.allow_reuse_address = True

    tries_to_open = 0
    while tries_to_open < 5:
        try:
            server = ThreadedTCPServer(("localhost", port), ThreadedTCPRequestHandler)

            # crudely signal that the server is ready
            with open("/tmp/server-ready.nil", "w"):
                pass

            server.serve_forever()
        except Exception as e:
            tries_to_open = tries_to_open + 1

            print(f"Port {port} not yet available, try again in {tries_to_open}s...")
            print(e)
            time.sleep(tries_to_open)

    print(f"Error: port {port} did not become ready")
    _stop_processes(0, 0)
