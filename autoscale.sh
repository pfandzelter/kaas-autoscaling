#!/usr/bin/env bash
#

OUTER_REPEATS=2  # How many times to repeat each run, for statistical analysis
# INNER_REPEATS=5  # How many times to repeat each number of parallel requests
INTERVAL=10      # How long to wait between each scaling step (in seconds)
AVAILABLE_GPUS=4
MAX_MPL_PER_GPU=2 # Maximum number of parallel requests per GPU
MIN_MPL=1        # Minimum number of parallel requests
MAX_MPL=20       # Maximum number of parallel requests

KERNEL="cuda-matmul"
FN="cuda-matmul-fn"
CLIENT="cuda-matmul-client"
N="10000"         # Matrix size

SERVERLOGS=serverlogs.txt

GPU_MONITORING_SCRIPT="nvidia-smi --query-gpu=timestamp,gpu_bus_id,utilization.gpu --format=csv --lms=100"

for repeat in $(seq 1 $OUTER_REPEATS)
do

    echo "Starting repeat $repeat/$OUTER_REPEATS"

    # start MPS everywhere
    echo "Starting MPS everywhere..."
    echo "quit" | sudo nvidia-cuda-mps-control
    sleep 1

    for GPU in $(seq 0 1 $((AVAILABLE_GPUS-1)))
    do
        sudo nvidia-smi -i $GPU -c EXCLUSIVE_PROCESS
    done

    sleep 1
    sudo nvidia-cuda-mps-control -d
    sleep 1

    # MPS running everywhere!


    # start the GPU server in the background
    # clean up old run first
    if [ -f /tmp/server-ready.nil ]
    then
        rm /tmp/server-ready.nil
    fi

    # silencing stderr because it's noisy
    PYTHONUNBUFFERED=1 ./gpu-server-scaling.py $FN --port 8081 --num-gpus $AVAILABLE_GPUS --max-req-per-gpu $MAX_MPL_PER_GPU > $SERVERLOGS 2> /dev/null &
    SERVER_PID=$!

    # wait for the server to become ready
    echo -n "waiting for server to become ready"
    until [ -f /tmp/server-ready.nil ]
    do
    echo -n "."
    sleep 1
    done
    echo ""
    echo "Server ready! Starting experiment."

    # start the GPU monitoring
    echo "Starting GPU monitoring..."
    $GPU_MONITORING_SCRIPT > gpu-monitoring.csv &
    GPU_MONITORING_PID=$!

    ./load.py --client $CLIENT --input $N --experiment-name autoscaling --task-name "autoscaling-$KERNEL-$N-$repeat" --experiment-description "Autoscaling of a multicore GPU app ($KERNEL) with adapted KaaS" --min-parallel $MIN_MPL --max-parallel $MAX_MPL --step-size 1 --interval $INTERVAL

    # kill the GPU monitoring
    echo "Killing GPU monitoring..."
    kill $GPU_MONITORING_PID

    # done!
    # kill the server
    echo "Experiment $repeat/$OUTER_REPEATS done! Killing server."
    sleep 20
    kill $SERVER_PID
    echo "Server killed."
    echo "Sorting server logs..."
    ./sortserverlogs.py $SERVERLOGS "results-autoscaling/autoscaling-$KERNEL-$N-$repeat-serverlogs.csv"
    rm $SERVERLOGS
    mv gpu-monitoring.csv "results-autoscaling/autoscaling-$KERNEL-$N-$repeat-gpu-monitoring.csv"
    echo "Waiting 20 seconds for the server to shut down."
    sleep 20

    # reset the GPUs again
    echo "Resetting GPUs..."
    for GPU in $(seq 0 1 $((AVAILABLE_GPUS-1)))
    do
        sudo nvidia-smi -i $GPU -c DEFAULT
    done

    sleep 1
    sudo nvidia-cuda-mps-control -d
    echo "Done waiting. Moving on to next experiment."
done
