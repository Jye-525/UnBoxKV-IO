"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from datetime import datetime
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from request_generator import (fixed_request_length_generator, uniform_request_length_generator,
                               zipf_request_length_generator, 
                               sample_ShareGPT_requests, sample_LEval_requests)

def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: Optional[int],
    max_num_seqs: Optional[int] = None,
    gpu_memory_utilization: float = 0.85,
    swap_space: int = 4,
    download_dir: Optional[str] = None,
    block_size: int = 16,
    preemption_mode: Optional[str] = None,
    enable_mixed_batch: bool = False,
    enable_warmup: bool = True
) -> float:
    assert not use_beam_search # for non-beam-search
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs, 
        disable_log_stats=False,
        swap_space=swap_space,
        block_size=block_size,
        preemption_mode = preemption_mode,
        enable_mixed_batch = enable_mixed_batch
    )

    prompts = []
    sampling_params = []
    # Add the requests to the engine.
    for prompt, prompt_len, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
            ))

    # Warmup iters
    if enable_warmup:
        warmup_start_time = datetime.now().strftime("%m-%d %H:%M:%S")
        print(f"INFO {warmup_start_time} Start warmup...")
        warmup_iters = 1
        max_tokens_num = 10
        input_len = 1024
        warmup_prompts = "hi" * (input_len - 1)
        warmup_sampling_params = SamplingParams(
            n=n,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=max_tokens_num,
        )

        for _ in range(warmup_iters):
            llm.generate(warmup_prompts, warmup_sampling_params, use_tqdm=False)  
        time.sleep(5)
        
    # reset LLM's metrics
    llm.reset_metrics()
    bench_start_time = datetime.now().strftime("%m-%d %H:%M:%S")
    print(f"INFO {bench_start_time} Start benchmarking...")
    torch.cuda.nvtx.range_push("llm_generate")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    end = time.time()
    torch.cuda.nvtx.range_pop()
    for output in outputs:
        print(f"RequestOutput(request_id={output.request_id}, request_type={output.request_type}, finished={output.finished}, metrics={output.metrics})", flush=True)
    #llm.close_json_writer()
    return end - start

def run_mii(
    requests: List[Tuple[str, int, int]],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
    max_model_len: Optional[int] = None,
    enable_warmup: bool = True 
) -> float:
    from mii import pipeline
    llm = pipeline(model, tensor_parallel=tensor_parallel_size)
    prompts = [prompt for prompt, _, _ in requests]

    # warmup iters
    if enable_warmup:
        warmup_start_time = datetime.now().strftime("%m-%d %H:%M:%S")
        print(f"INFO {warmup_start_time} Start warmup...")
        warmup_iters = 2
        warmup_prompts = "hi" * (1024 - 1) 
        for _ in range(warmup_iters):
            if max_model_len is not None:
                llm(warmup_prompts, max_new_tokens=output_len, top_p=0.9, max_length=max_model_len)
            else:
                llm(warmup_prompts, max_new_tokens=output_len, top_p=0.9)
        time.sleep(5)
    
    # benchmark
    bench_start_time = datetime.now().strftime("%m-%d %H:%M:%S")
    print(f"INFO {bench_start_time} Start benchmarking...")
    start = time.time()
    if max_model_len is not None:
        llm(prompts, max_new_tokens=output_len, top_p=0.9, max_length=max_model_len)
    else:
        llm(prompts, max_new_tokens=output_len, top_p=0.9)
    end = time.time() 
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    
    if args.generator == "synthetic":
        if args.synthetic_type == 'fixed':
            requests = fixed_request_length_generator(args.num_prompts, args.input_len, args.output_len)
        
        elif args.synthetic_type == 'uniform':
            requests = uniform_request_length_generator(args.num_prompts, args.min_len, args.max_len, args.prefill_to_decode_ratio)
        elif args.synthetic_type == 'zipf':
            requests = zipf_request_length_generator(args.num_prompts, args.min_len, args.max_len, args.seed, args.prefill_to_decode_ratio)
        else:
            raise ValueError(f"Unknown synthetic type: {args.synthetic_type}")
        
        # prompt = "hello" * (args.input_len - 1)
        # requests = [(prompt, args.input_len, args.output_len)
        #             for _ in range(args.num_prompts)]
    else:
        assert args.dataset is not None
        # Here the max_num_batched_tokens = max_context_length <= max_model_length
        #num_prompts=32
        num_prompts=args.num_prompts
        if args.workload == "ShareGPT":
            requests = sample_ShareGPT_requests(args.dataset, num_prompts, tokenizer,
                                   args.output_len, args.max_num_batched_tokens, is_test_mode=True)
        elif args.workload == "leval":
            requests = sample_LEval_requests(args.dataset, num_prompts, tokenizer,
                                   args.output_len, args.max_num_batched_tokens)
        else:
            raise ValueError(f"Unknown workload: {args.workload}, dataset: {args.dataset}")
        print(f"Loaded requests from {args.dataset} successfully.")

    print(f"Loaded {len(requests)} requests, requiring {args.num_prompts} requests.")
    print(f"Original order of the requests")
    for request in requests:
        print(f"User Requests: prompt_len={request[1]}, output_len={request[2]}, sequence_len={request[1]+request[2]}...")

    #random.seed(5678) # 5678 or 9721
    #random.seed(9721)
    #random.shuffle(requests)
    #print(f"Shuffled order of the requests")
    #for request in requests:
    #    print(f"User Requests: prompt_len={request[1]}, output_len={request[2]}, sequence_len={request[1]+request[2]}...")

    requests = requests[:args.num_prompts]
    print(f"Selected required requests {args.num_prompts}")
    for request in requests:
        print(f"User Requests: prompt_len={request[1]}, output_len={request[2]}, sequence_len={request[1]+request[2]}...")

    if args.backend == "vllm":
        if args.enable_chunked_prefill:
            print(f"Chunked prefill is enabled. max_num_batched_tokens={args.max_num_batched_tokens}, max_num_batched_seqs={args.max_num_seqs}") 
        elif args.enable_mixed_batch:
            print(f"Mixed batch is enabled. max_num_batched_tokens={args.max_num_batched_tokens}, max_num_batched_seqs={args.max_num_seqs}")
        else:
            print(f"Running vLLM with default batching strategy. max_num_batched_tokens={args.max_num_batched_tokens}, max_num_batched_seqs={args.max_num_seqs}")

        enable_warmup = True
        elapsed_time = run_vllm(
            requests, args.model, args.tokenizer, 
            args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
            args.trust_remote_code, args.dtype, args.max_model_len,
            args.enforce_eager, args.kv_cache_dtype, args.device,
            args.enable_prefix_caching, args.enable_chunked_prefill,
            args.max_num_batched_tokens, args.max_num_seqs, args.gpu_memory_utilization, args.swap_space,
            args.download_dir, args.block_size, args.preemption_mode,
            args.enable_mixed_batch, enable_warmup)
        
        ## This one is only used to control the single test case to trigger the recompute_vs_swap
        #elapsed_time = run_vllm_single_test(
        #    requests, args.model, args.tokenizer, 
        #    args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
        #    args.trust_remote_code, args.dtype, args.max_model_len,
        #    args.enforce_eager, args.kv_cache_dtype, args.device,
        #    args.enable_prefix_caching, args.enable_chunked_prefill,
        #    args.max_num_batched_tokens, args.gpu_memory_utilization, args.swap_space,
        #    args.download_dir, args.block_size, args.input_len, args.output_len, args.preemption_mode 
        #)
    
    elif args.backend == "mii":
        enable_warmup = True
        elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                               args.output_len, enable_warmup)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    
    print(f"Total_num_tokens: {total_num_tokens}")
    print(f"End-to-End latency: {elapsed_time:.2f} seconds") 
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")
    print(f"Per_token_time: {(elapsed_time * 1000)/total_num_tokens:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "mii"],
                        default="vllm")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface') 
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--generator",
                        type=str,
                        choices=["synthetic", "real"],
                        default="synthetic")
    parser.add_argument("--workload",
                        type=str,
                        choices=["ShareGPT", "leval"],
                        default="ShareGPT")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.This need to be set when --generator is set to real.")
    parser.add_argument("--synthetic-type",
                        type=str,
                        choices=["fixed", "uniform", "zipf"],
                        default="fixed")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request, only used when --synthetic-type is fixed.")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the output length from the dataset.Used when --synthetic-type is fixed or --generator is real.")
    parser.add_argument("--min-len",
                        type=int,
                        default=4,
                        help="Minimum tokens for a request, only used when --synthetic-type is uniform.")
    parser.add_argument("--max-len",
                        type=int,
                        default=2048,
                        help="Maximum tokens for a request, only used when --synthetic-type is uniform.")
    parser.add_argument("--prefill-to-decode-ratio",
                        type=float,
                        default=1.0,
                        help="Prefill to decode ratio, only used when --synthetic-type is uniform.")
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.95,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--swap_space",type=int,
                        default=64,
                        help="cpu swap space for kv cache")
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument('--max-model-len',
                        type=int,
                        default=8192,
                        #default=None,
                        help='Maximum length of a sequence (including prompt and output). '
                        'If None, will be derived from the model.')
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type. '
        'FP8_E5M2 (without scaling) is only supported on cuda version greater '
        'than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for '
        'common inference criteria.')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help='device type for vLLM execution, supporting CUDA and CPU.')
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--max-num-seqs',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument('--block-size',
                        type=int,
                        default=16,
                        help='block size for kv cache')
    
    parser.add_argument('--preemption-mode',
                        type=str,
                        default="recompute",
                        choices=["recompute", "swap"],
                        help='preemption mode for processing kv cache overflow')
    
    parser.add_argument("--enable-mixed-batch", action="store_true",
                        help="enable mixed batch for vLLM backend")

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    
    assert args.num_prompts > 0
    assert args.generator == "synthetic" or args.generator == "real"
    if args.generator == "synthetic":
        if args.synthetic_type == "fixed":
            assert args.input_len is not None
            assert args.output_len is not None
        else:
            assert args.min_len is not None
            assert args.max_len is not None
    else:
        assert args.dataset is not None


    if args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)
