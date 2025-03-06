# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
export MII_CACHE_PATH=/home/jieye/viper2/benchmark_mii/mii_cache

MODELS=(meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Llama-2-70b-hf tiiuae/falcon-40B tiiuae/falcon-180B microsoft/phi-2 mistralai/Mixtral-8x7B-v0.1)

for MODEL in ${MODELS[@]}; do
    python ./run_benchmark.py --model ${MODEL} --stream
    # python ./run_benchmark.py --model ${MODEL} --stream --vllm
done

# Extra runs for Mixtral with non-default settings
# python ./run_benchmark.py --model mistralai/Mixtral-8x7B-v0.1 --stream --tp_size 4 --mean_prompt_length 500 --mean_max_new_tokens 150 500 1024
# python ./run_benchmark.py --model mistralai/Mixtral-8x7B-v0.1 --stream --tp_size 4 --mean_prompt_length 500 --mean_max_new_tokens 150 500 1024 --vllm


# EN_MIXED_BATCH=False # True: enable mixed batch; False: disable mixed batch
# EN_CHUNKED_PREFILL=True
prefill_len=4000 #
max_batched_len=4096
echo "Start running fixed-length prefill with "${prefill_len} "tokens using vLLM default batching."
./run_bench_llama_64k_real_latest.sh 10 1 $i ${prefill_len} False False
sleep 5
echo "Start running fixed-length prefill with "${prefill_len} "tokens using mix batching."
./run_bench_llama_64k_real_latest.sh 10 1 $i ${prefill_len} True False
sleep 5
echo "Start running fixed-length prefill with "${prefill_len} "tokens using chunked-prefill batching."

max_batched_len=(4096 2048 1024 512)
for i in "${max_batched_len[@]}"
do
    #./run_bench_llama_64k_real_latest.sh $i 1
    ./run_bench_llama_64k_real_latest.sh 10 1 $i ${prefill_len} False True
    sleep 5
done
