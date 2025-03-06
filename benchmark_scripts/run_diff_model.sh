#!/bin/bash
prefill_len=256 #
gen_len=10
max_batched_len=2048
nums=2 #32, 64, 128

function restart_ray() {
    ray stop
    ray stop

    rm -rf $HOME/ray/tmp*

    ray start --head --temp-dir=/home/jieye/ray --disable-usage-stats
}

function run_vllm_default() {
    echo "Start running fixed-length prefill with "${prefill_len} "tokens using vLLM default batching."
    ./run_bench_llama_64k_real_latest.sh $nums 1 ${max_batched_len} ${prefill_len} "False" "False" ${gen_len}
    sleep 5
}

function run_mix_batching() {
    echo "Start running fixed-length prefill with "${prefill_len} "tokens using mix batching."
    ./run_bench_llama_64k_real_latest.sh $nums 1 ${max_batched_len} ${prefill_len} "True" "False" ${gen_len}
    sleep 5
}

function run_chunked_prefill() {
    echo "Start running fixed-length prefill with "${prefill_len} "tokens using chunked-prefill batching."
    #max_batched_len=(2048 1024 512)
    max_batched_len=(4096 2048 1024 512) 
    for i in "${max_batched_len[@]}"
    do
        #restart_ray 
        ./run_bench_llama_64k_real_latest.sh $nums 1 $i ${prefill_len} "False" "True" ${gen_len}
        sleep 5
    done
}

function run_decode_batch() {
    echo "Start decode batch test....."
    #nums=(4 8 16 32 64 128 256 512)
    #nums=(64 128 256 512)
    nums=(2 4 8 16 32 64 128 256)
    for i in "${nums[@]}"
    do
        ./run_bench_llama_64k_real_latest.sh $i 1 ${max_batched_len} ${prefill_len} "False" "False" ${gen_len}
    done
}


run_vllm_default
#run_mix_batching
#run_chunked_prefill
#run_decode_batch
