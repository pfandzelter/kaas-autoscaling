#!/usr/bin/env bash
#

OUTER_REPEATS=1  # How many times to repeat each run, for statistical analysis
# INNER_REPEATS=5  # How many times to repeat each number of parallel requests
INTERVAL=10      # How long to wait between each scaling step (in seconds)
AVAILABLE_GPUS=8
MAX_MPL_PER_GPU=4 # Maximum number of parallel requests per GPU
MIN_MPL=1        # Minimum number of parallel requests
# MAX_MPL=$(( AVAILABLE_GPUS * MAX_MPL_PER_GPU ))       # Maximum number of parallel requests
MAX_MPL=10

KERNEL="cuda-matmul-cpu"
FN="cuda-matmul-cpu"
CLIENT="cuda-matmul-client"
N="500"         # Matrix size

SERVERLOGS=serverlogs.txt

for repeat in $(seq 1 $OUTER_REPEATS)
do

    echo "Starting repeat $repeat/$OUTER_REPEATS"
    # MPS running everywhere!

    # start the GPU server in the background
    # clean up old run first
    if [ -f /tmp/server-ready.nil ]
    then
        rm /tmp/server-ready.nil
    fi

    # silencing stderr because it's noisy
    PYTHONUNBUFFERED=1 ./gpu-server-scaling.py $FN --port 8080 --num-gpus $AVAILABLE_GPUS --max-req-per-gpu $MAX_MPL_PER_GPU > $SERVERLOGS 2> /dev/null &
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

    ./load.py --client $CLIENT --input $N --experiment-name autoscaling --task-name "autoscaling-$KERNEL-$N-$repeat" --experiment-description "Autoscaling of a multicore CPU app with adapted KaaS" --min-parallel $MIN_MPL --max-parallel $MAX_MPL --step-size 1 --interval $INTERVAL

    # done!
    # kill the server
    echo "Experiment $repeat/$OUTER_REPEATS done! Killing server."
    sleep 20
    kill $SERVER_PID
    echo "Server killed."
    echo "Sorting server logs..."
    ./sortserverlogs.py $SERVERLOGS "results-autoscaling/autoscaling-matmul-$N-$repeat-serverlogs.csv"
    rm $SERVERLOGS
    echo "Waiting 20 seconds for the server to shut down."
    sleep 20
    echo "Done waiting. Moving on to next experiment."
done
