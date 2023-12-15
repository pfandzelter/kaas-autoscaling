# Autoscaling Experiment

1. Install the prerequisites (Debian 11)

    ```sh
    sudo apt-get update
    sudo apt-get install rsync python3-pip python3-dev python3-venv
    ```

1. Prepare the execution environment

    ```sh
    # create a venv
    python3.9 -m venv .venv/
    source .venv/bin/activate

    # install dependencies
    python3 -m pip install -r requirements.txt
    ```

1. Run the experiments

    ```sh
    # activate the venv
    source .venv/bin/activate

    # run the script
    ./autoscale.sh
    ```

1. Do the analysis. See `plot-autoscaling.ipynb`.

## Measuring GPU Percentage

Here is what ChatGPT recommends:

```sh
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 10
```

But that's probably a bit too limited for us.

With Performance Co-Pilot (PCP):

```sh
# install pcp
# may already be installed
# install pmdanvidia
# on debian:
sudo apt-get update
sudo apt-get install pmdanvidia pcp-gui

# may need to install:
cd /var/lib/pcp/pmdas/nvidia
sudo ./Install

# collect metrics
# every second
# pmdumptext may also be possible
pmrep localhost:nvidia.gpuactive

pmdumptext -l -t 0.001s localhost:nvidia.gpuactive -f "%s"
```
