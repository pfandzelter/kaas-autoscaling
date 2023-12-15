#!/usr/bin/env python3

import os
import sys

if __name__ == "__main__":
    # arg 1: input file
    # arg 2: output file

    if len(sys.argv) < 3:
        print("Usage: python3 sortserverlogs.py <input file> <output file>")
        exit(-1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    useful_logs = []

    with open(input_file, "r") as f:
        for line in f.readlines():
            if line.startswith("@@@ Booted"):
                # need to split a line of the format "@@@ Booted 4 new workers on GPU 0 at 1694604659.1637785" into num_workers, gpu, timestamp
                # we can do this by splitting on spaces and then parsing the last element as a float
                e = line.split(" ")
                num_workers = int(e[2])
                gpu = int(e[7])
                timestamp = float(e[9])

                useful_logs.append((num_workers, gpu, timestamp))

            elif line.startswith("@@@ Error"):
                # need to split a line of the format "@@@ ERROR all workers are full at 1694604659.1637785" to get the timestamp
                e = line.split(" ")
                timestamp = float(e[7])

                useful_logs.append((0, 0, timestamp))

    # sort by timestamp
    useful_logs.sort(key=lambda x: x[2])

    # write to output file
    with open(output_file, "w") as f:
        f.write("num_workers,gpu,timestamp\n")
        for log in useful_logs:
            f.write(f"{log[0]},{log[1]},{log[2]}\n")
