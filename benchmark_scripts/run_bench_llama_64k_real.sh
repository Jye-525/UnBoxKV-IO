# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# nsys profile --force-overwrite true -o $output_dir/log-$log_str-nsys -t cuda,nvtx deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options}

# DeepSpeed Team
export MII_CACHE_PATH=/home/jieye/viper2/benchmark_mii/mii_cache

SRC_DIR=/home/jieye/viper2/benchmark_mii
MODEL_DIR=/grand/projects/VeloC/jye/viper2/huggingface-hub
#MODEL_DIR=/grand/projects/RECUP/jye/viper2/huggingface-hub

#PROF_TYPE=2 # 1: enable nsys profiling; 2: disable profiling
MONITOR=False # True: enable monitoring; False: disable monitoring
BACKEND=vllm # choices: vllm, mii
# MODE_NAME=Llama-2-7b-hf # Llama-2-7b-hf, Yarn-Llama-2-7b-64k
MODE_NAME=Yarn-Llama-2-7b-64k # Llama-2-7b-hf, Yarn-Llama-2-7b-64k
#MODE_NAME=meta-llama-3.1-8b
# MODE_NAME=opt-13b
NUM_PROMPTS=$1
NRANKS_PER_NODE=1
BLOCK_SIZE=16
EN_CHUNKED_PREFILL=False # True: enable chunked prefill; False: disable chunked prefill
CHUNK_SIZE=512
MAX_BATCHED_TOKENS=8192
EN_MIXED_BATCH=False # True: enable mixed batch; False: disable mixed batch
TRY=$2

GPU_U=40
PREM_MODE=recompute  # choices: recompute, swap
use_case=tst_kv_pattern
CPU_MEM=pinned
Test_latency=False # True: test latency; False: test throughput

DATA_SET="/lus/grand/projects/VeloC/jye/viper2/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
#DATA_SET="/lus/grand/projects/VeloC/jye/viper2/datasets/processed_leval.jsonl"
#DATA_SET="/lus/grand/projects/RECUP/jye/viper2/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_NAME="ShareGPT"
#DATASET_NAME="Leval"
GEN_LEN=3 #leave empty for None, otherwise specify the generation length

if [ "$PREM_MODE" = "recompute" ]; then
    PREM_MODE_STR="recmp"
elif [ "$PREM_MODE" = "swap" ]; then
    PREM_MODE_STR="swap"
else
    echo "Invalid preemption mode: ${PREM_MODE}. Exiting..."
    exit -1
fi


if [ ${BACKEND} = "vllm" ];
then
    # RESULT_DIR="vllm_hipc_llama_64k/${use_case}"
    RESULT_DIR="vllm_ipdps/${use_case}"
elif [ ${BACKEND} = "mii" ];
then
    RESULT_DIR="mii_results"
    #RESULT_DIR="mii_results_profile"
else
    echo "Invalid backend: ${BACKEND}. Exiting..."
    exit -1
fi


OUTPUT_DIR=/home/jieye/viper2/benchmark_mii/${RESULT_DIR}/
LOG_STR=log-${BACKEND}-${MODE_NAME}-g${GEN_LEN}-block${BLOCK_SIZE}-r${NUM_PROMPTS}-${PREM_MODE_STR}-G${GPU_U}-${CPU_MEM}-${TRY}-${DATASET_NAME}-fixedchunk

# Create output directory
if [[ ! -e $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
fi

function start_gpu_monitor() {
    echo "Monitoring starting......"
    # start gpu monitor & host memory monitor
    for gpu_id in $(seq 0 $((NRANKS_PER_NODE - 1))); 
    do
        setsid python ${SRC_DIR}/monitor_gpu.py $gpu_id "${OUTPUT_DIR}/${LOG_STR}-monitor-${gpu_id}.csv" &
        monitor_pid[$gpu_id]=$!
        echo "Monitoring started for GPU $gpu_id at PID ${monitor_pid[$gpu_id]}."
    done
    setsid python ${SRC_DIR}/monitor_host_mem.py "${OUTPUT_DIR}/${LOG_STR}-monitor-vmem.csv" &
}

function stop_gpu_monitor() {
    echo "Monitoring stopping......"
    # Terminate monitoring for all GPUs and host memory
    for gpu_id in $(seq 0 $((NRANKS_PER_NODE - 1))); 
    do
        echo "Killing the monitoring script for GPU $gpu_id at PID ${monitor_pid[$gpu_id]}."
        kill -2 ${monitor_pid[$gpu_id]}
        echo "SIGTERM (kill -2) instructed to monitoring script for GPU $gpu_id."
        wait ${monitor_pid[$gpu_id]}
        echo "Killed the monitoring script for GPU $gpu_id."
    done

    kill -2 $(pgrep -f monitor_host_mem.py)
    echo "SIGTERM (kill -2) instructed to monitoring script for host memory."
    wait $(pgrep -f monitor_host_mem.py)
    echo "Killed the monitoring script for host memory."
}


# Run the Benchmark
if [ ${MONITOR} = "True" ];then
    start_gpu_monitor
fi

sleep 5

echo "Deploying the model and start benchmark latency test...."
if [ $Test_latency = "True" ]; then
    prog_name="benchmark_latency.py"
    LOG_STR="${LOG_STR}-latency"
    echo "Testing latency..." 
else
    prog_name="benchmark_throughput.py"
    LOG_STR="${LOG_STR}-thrpt" 
    echo "Testing throughput..."
fi

# running with profiling disabled
if [ ${BACKEND} = "vllm" ] && [ $EN_CHUNKED_PREFILL = "True" ]; then
    LOG_STR="${LOG_STR}-prefill-chunked${CHUNK_SIZE}"
    run_cmd="python ${SRC_DIR}/non-persistent/${prog_name} \
            --backend ${BACKEND} \
            --model ${MODEL_DIR}/${MODE_NAME} \
            --tensor-parallel-size ${NRANKS_PER_NODE} \
            --dataset ${DATA_SET} \
            --output-len ${GEN_LEN} \
            --num-prompts ${NUM_PROMPTS} \
            --enforce-eager \
            --enable-chunked-prefill \
            --max-num-batched-tokens ${MAX_BATCHED_TOKENS} \
            --block-size ${BLOCK_SIZE} \
            --preemption-mode ${PREM_MODE} \
            > ${OUTPUT_DIR}/${LOG_STR}.log 2>&1"
else
    if [ ${BACKEND} = "mii" ]; then
        LOG_STR="${LOG_STR}-chunked${CHUNK_SIZE}"
    fi
    # CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=3 
    run_cmd="python ${SRC_DIR}/non-persistent/${prog_name} \
            --backend ${BACKEND} \
            --model ${MODEL_DIR}/${MODE_NAME} \
            --tensor-parallel-size ${NRANKS_PER_NODE} \
            --dataset ${DATA_SET} \
            --num-prompts ${NUM_PROMPTS} \
            --enforce-eager \
            --block-size ${BLOCK_SIZE} \
            --preemption-mode ${PREM_MODE} "

    #CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=3 --physcpubind=24
    # run_cmd="CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=3 /soft/compilers/cudatoolkit/cuda-12.2.2/bin/nsys profile --force-overwrite true -o ${OUTPUT_DIR}/${LOG_STR}-nsys -t cuda,nvtx python ${SRC_DIR}/non-persistent/${prog_name} \
    #        --backend ${BACKEND} \
    #        --model ${MODEL_DIR}/${MODE_NAME} \
    #        --tensor-parallel-size ${NRANKS_PER_NODE} \
    #        --dataset ${DATA_SET} \
    #        --num-prompts ${NUM_PROMPTS} \
    #        --enforce-eager \
    #        --block-size ${BLOCK_SIZE} \
    #        --preemption-mode ${PREM_MODE} "

    if [[ ! -z "${GEN_LEN}" ]]; then
        run_cmd=${run_cmd}" --output-len ${GEN_LEN} "
    fi
            
    if [ $EN_MIXED_BATCH = "True" ]; then
        run_cmd=${run_cmd}" --enable-mixed-batch "
    fi
    run_cmd=${run_cmd}"> ${OUTPUT_DIR}/${LOG_STR}.log 2>&1"
fi

echo $run_cmd
eval $run_cmd

echo "End benchmark test...."

if [ ${MONITOR} = "True" ];then
    stop_gpu_monitor
fi
echo "All done!"
