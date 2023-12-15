#!/usr/bin/env python3

import argparse
import datetime
import importlib
import multiprocessing as mp
import os
import threading
import time


class Log:
    def __init__(
        self,
        timestamp: float,
        inner_time_ms: str,
        outer_time_ms: str,
        setup_time_ms: str,
        cold_start: str,
        copy: int,
    ):
        self.timestamp = timestamp
        self.inner_time_ms = inner_time_ms
        self.outer_time_ms = outer_time_ms
        self.setup_time_ms = setup_time_ms
        self.cold_start = cold_start
        self.copy = copy


def _worker(
    client: str,
    arg: str,
    copy: int,
    log_queue: mp.Queue,  # type: ignore
    term_queue: mp.Queue,  # type: ignore
) -> None:
    try:
        fn = importlib.import_module(client)
    except Exception as e:
        raise ImportError(
            f"no file {client}.py was found or there was an error importing it -- make sure that you only run this container with a custom function"
        ) from e

    try:
        getattr(fn, "run_client")
    except AttributeError as e:
        raise ImportError(
            f"{client}.py is present but method run_client() could not be found"
        ) from e

    try:
        getattr(fn, "prepare")
        fn.prepare(arg, copy)

    except AttributeError as e:
        print("No prepare() method found, skipping")

    while True:
        try:
            # see if we should terminate
            if term_queue.get_nowait():
                break
        except:
            pass

        t_0 = time.perf_counter()
        ts = time.time()

        _, inner_time_ms, setup_time_ms, cold_start = fn.run_client(arg)

        t_1 = time.perf_counter()

        outer_time_ms = str(round((t_1 - t_0) * 1000, 5))

        log_queue.put(
            Log(
                timestamp=ts,
                inner_time_ms=inner_time_ms,
                outer_time_ms=outer_time_ms,
                setup_time_ms=setup_time_ms,
                cold_start=cold_start,
                copy=copy,
            )
        )

    try:
        getattr(fn, "cleanup")
        fn.cleanup()

    except AttributeError as e:
        print("No clean() method found, skipping")


if __name__ == "__main__":
    # arguments
    # client to run
    # input
    # name of the experiment (adapted from sharp)
    # name of the task (adapted from sharp)
    # description of the experiment (adapted from sharp)
    # minimum parallel requests
    # maximum parallel requests
    # step size
    # interval until next step

    parser = argparse.ArgumentParser(description="Autoscaling KaaS Load Generator")

    parser.add_argument(
        "--client",
        type=str,
        help="client to run. use a relative or full path to the Python file.",
    )

    parser.add_argument(
        "--input",
        type=str,
        help="input to the client.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="name of the experiment.",
    )

    parser.add_argument(
        "--task-name",
        type=str,
        help="name of the task.",
    )

    parser.add_argument(
        "--experiment-description",
        type=str,
        help="description of the experiment.",
    )

    parser.add_argument(
        "--min-parallel",
        type=int,
        help="minimum parallel requests.",
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        help="maximum parallel requests.",
    )

    parser.add_argument(
        "--step-size",
        type=int,
        help="step size.",
    )

    parser.add_argument(
        "--interval",
        type=int,
        help="interval until next step in seconds.",
    )

    args = parser.parse_args()

    # we start a worker that logs everything
    # we start min_parallel workers
    # each worker continuously sends requests to the server, until the program ends
    # we start a new worker every interval seconds

    # some prep work:
    # create a directory for results (if it does not exist)
    results_dir = f"results-{args.experiment_name}"
    os.makedirs(results_dir, exist_ok=True)

    results_file = f"{results_dir}/{args.task_name}.csv"

    # create a markdown file that saves the parameters (in Eitan's spirit)
    description_file = f"{results_dir}/{args.task_name}.md"

    start_time = datetime.datetime.now(datetime.timezone.utc)

    with open(description_file, "w") as f:
        f.write(f"This file describes the fields in the file {results_file}.\n")
        f.write(f"The measurements were run starting on {start_time}.\n")
        f.write(f"The experiment was run on the machine {os.uname().nodename}.\n")

        f.write("\n")

        f.write("## Parameters\n")
        f.write(f"Client: {args.client}\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Experiment name: {args.experiment_name}\n")
        f.write(f"Task name: {args.task_name}\n")
        f.write(f"Experiment description: {args.experiment_description}\n")
        f.write(f"Minimum parallel requests: {args.min_parallel}\n")
        f.write(f"Maximum parallel requests: {args.max_parallel}\n")
        f.write(f"Step size: {args.step_size}\n")
        f.write(f"Interval: {args.interval}\n")

        f.write("\n")

    # now let's start with the actual experiment
    results_queue = mp.Queue()  # type: ignore

    concurrency = args.min_parallel

    # start the worker that logs everything
    def _logger(log_queue: mp.Queue) -> None:  # type: ignore
        global concurrency

        with open(results_file, "w") as f:
            f.write(
                "timestamp,inner_time_ms,outer_time_ms,setup_time,cold_start,copy,concurrency\n"
            )

            while True:
                result = log_queue.get()
                if result == "END":
                    break

                f.write(
                    f"{result.timestamp},{result.inner_time_ms},{result.outer_time_ms},{result.setup_time_ms},{result.cold_start},{result.copy},{concurrency}\n"
                )
                print(
                    "Outer time: ",
                    float(result.outer_time_ms) / 1000.0,
                    "(cold)" if result.cold_start else "",
                )

    logger = threading.Thread(target=_logger, args=(results_queue,))
    logger.start()

    workers = []
    term_queue = mp.Queue()  # type: ignore

    for i in range(concurrency):
        worker = mp.Process(
            target=_worker,
            args=(
                args.client,
                args.input,
                i,
                results_queue,
                term_queue,
            ),
        )
        worker.start()
        workers.append(worker)
        print(f"Started client {i} ({time.time()})")

    # now we start the actual experiment
    while concurrency < args.max_parallel:
        time.sleep(args.interval)

        for i in range(args.step_size):
            # start a new worker
            worker = mp.Process(
                target=_worker,
                args=(
                    args.client,
                    args.input,
                    concurrency + i,
                    results_queue,
                    term_queue,
                ),
            )
            worker.start()
            workers.append(worker)
            print(f"Started client {concurrency + i} ({time.time()})")

        concurrency += args.step_size

    time.sleep(args.interval)

    # stop all workers
    for worker in workers:
        term_queue.put(True)

    for worker in workers:
        worker.join()

    # stop the logger
    results_queue.put("END")
    logger.join()

    # write the description file
    with open(description_file, "a") as f:
        f.write("\n")
        f.write(
            f"The experiment terminated on {datetime.datetime.now(datetime.timezone.utc)}.\n"
        )
        f.write(
            f"The experiment ran for {datetime.datetime.now(datetime.timezone.utc) - start_time}.\n"
        )
