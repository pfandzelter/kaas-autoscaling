# Kernel-as-a-Service: GPU Autoscaling Evaluation

With "Kernel-as-a-Service", we have proposed adopting a serverless programming model for heterogeneous hardware accelerators including GPUs, FPGAs, TPUs, and QPUs (quantum processing unit).
Underlying our evaluation are different prototype implementations for these accelerators.
For legal reasons, we cannot make most of them available as open-source software.

In this repository, find the source code to replicate the autoscaling evaluation of our paper.
The implementation of this experiment was done entirely at TU Berlin and can thus be distributed with a [permissive open-source license](./LICENSE) (we note, however, that the experiment execution was done at Hewlett Packard Labs).

You can obtain an author copy of our paper here: [pfandzelter.com](https://pfandzelter.com/publication/2023-kaas/kaas-middleware2023.pdf)

If you use this software in a publication, please cite the original paper:

```bibtex
@inproceedings{pfandzelter2023kernel,
    author = "Pfandzelter, Tobias and Dhakal, Aditya and Frachtenberg, Eitan and Chalamasetti, Sai Rahul and Emmot, Darel and Hogade, Ninad and Hong Enriquez, Rolando Pablo and Rattihalli, Gourav and Bermbach, David and Milojicic, Dejan",
    title = "Kernel-as-a-Service: A Serverless Programming Model for Heterogeneous Hardware Accelerators",
    booktitle = "Proceedings of the 24th International Middleware Conference",
    pages = "192--206",
    month = dec,
    year = 2023,
    publisher = "ACM",
    address = "New York, NY, USA",
    series = "Middleware '23",
    location = "Bologna, Italy",
    url = "https://doi.org/10.1145/3590140.3629115",
}
```

## Setup

Our experiment was run on a GPU cluster with eight Nvidia Tesla V100 SXM2 GPUs.
The following tooling is required:

- `nvidia-smi` to configure the GPUs (most notably MPS) and measure power consumption
- Python3 with `python-pip`
- `numba-cuda` (see [the documentation for installation instructions](https://numba.pydata.org/numba-doc/latest/user/installing.html))
- `numpy` (see version information in [`requirements.txt`](./requirements.txt))

## Running

Use the `autoscale.sh` script to start the experiment on a cluster.
If required, you can configure the parameters in that script.

The experiment has four main components:

1. The `gpu-server-scaling.py` script loads the configured kernel (passed on the command line) and exposes a TCP endpoint. It is responsible for monitoring in-flight requests and allocating resources.
1. `cuda-matmul-fn.py` is our kernel code, calculating a matrix multiplication on a GPU using CUDA.
1. `cuda-matmul-client.py` is the client code for that kernel, a simple script generating a random matrix and sending it for processing to the kernel through KaaS.
1. `load.py` is an autoscaling load generator invoking the client code in parallel, scaling from a few to many concurrent requests depending on the given parameters.

There are also CPU variants available of our kernel.
Use `autoscale-cpu.sh` to trial out KaaS without GPUs.

## Analysis

Your results (including GPU monitoring data) will end up in the `results-autoscaling` directory.
Use the included `sortserverlogs.py` and `plot-autoscaling.ipynb` to generate results and plots.
