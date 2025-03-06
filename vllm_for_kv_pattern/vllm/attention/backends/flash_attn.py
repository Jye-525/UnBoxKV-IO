"""Attention layer with Flash and PagedAttention.

NOTE(woosuk): At the moment, this file includes a lot of duplicated code from
XFormers backend. The duplicated code will be removed once we use flash-attn or
flashinfer for all the attention operations.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from flash_attn import flash_attn_varlen_func

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
import time
from vllm.async_json_writer import AsyncJsonWriter

class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "FlashAttentionMetadata":
        return FlashAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class FlashAttentionMetadata(AttentionMetadataPerStage,
                             PagedAttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch.
    max_query_len: Optional[int]
    # Maximum sequence length in the batch.
    max_seq_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool


class FlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        layer_id: Optional[int] = None,
        AsyncJsonWriter: Optional[AsyncJsonWriter] = None,
        kv_write_pattern_trace_record: Optional[torch.Tensor] = None,
        k_cache_read_pattern: Optional[torch.Tensor] = None,
        v_cache_read_pattern: Optional[torch.Tensor] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")
        # print(f"FlashAttentionImpl: num_heads={num_heads}, head_size={head_size}, scale={scale}, num_kv_heads={num_kv_heads}, alibi_slopes={alibi_slopes}, sliding_window={sliding_window} num_queries_per_kv={self.num_queries_per_kv} suppored_head_sizes={suppored_head_sizes}", flush=True)
        self.layer_id = layer_id
        self.step = 0

        self.AsyncJsonWriter = AsyncJsonWriter
        self.kv_write_pattern_trace_record = kv_write_pattern_trace_record

        # 4 * 100 * 3 = 1200
        # self.k_cache_read_pattern = torch.full((10, 1200), -1, dtype=torch.int64, device='cuda:0')
        # self.v_cache_read_pattern = torch.full((10, 1200), -1, dtype=torch.int64, device='cuda:0')
        self.k_cache_read_pattern = k_cache_read_pattern
        self.v_cache_read_pattern = v_cache_read_pattern
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata[FlashAttentionMetadata],
        kv_scale: float,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """ 
        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        # print(f"FlashAttentionImpl: query={query.shape}, key={key.shape}, value={value.shape}, num_prefill_tokens={attn_metadata.num_prefill_tokens}, num_decode_tokens={attn_metadata.num_decode_tokens}", flush=True)
        self.step += 1
        start_1 = time.time()
        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            start = time.time()
            torch.cuda.nvtx.range_push("write_to_kv_cache")
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                attn_metadata.kv_cache_dtype,
                                                kv_scale, self.kv_write_pattern_trace_record)
            torch.cuda.nvtx.range_pop()
            # print(f"FlashAttentionImpl: PagedAttention.write_to_paged_cache layer {self.layer_id} took {time.time() - start} seconds", flush=True)
            self.write_kv_time_us = (time.time() - start) * 1e6

            if self.step > 3:
                # step 1 is for profiling; step 2, 3 is the warmup step, so skip it
                seq_lens = None
                # num_reqs = attn_metadata.num_prefills + attn_metadata.num_decodes
                # Here assume that a batch only contain either prefill or decode requests
                if attn_metadata.decode_metadata is not None:
                    # the batch includes decode requests
                    seq_lens = [1] * attn_metadata.num_decode_tokens 

                if attn_metadata.prefill_metadata is not None:
                    # the batch includes prefill requests
                    seq_lens = attn_metadata.prefill_metadata.seq_lens
                
                # collect the data in each layer
                real_step = self.step - 3
                kv_cache_range = key_cache.numel() * key_cache.element_size()
                 
                kv_write_pattern_info = {
                    "layer_id": self.layer_id,
                    "step": real_step,
                    "start_time": start,
                    "cpu_duration(us)": self.write_kv_time_us,
                    "kcache_start_addr": key_cache.data_ptr(),
                    "vcache_start_addr": value_cache.data_ptr(),
                    "kv_cache_range": kv_cache_range,
                    'seq_lens': seq_lens,
                    "kv_write_pattern_record": self.kv_write_pattern_trace_record[:num_tokens, :].cpu().tolist(),
                }

                # (ToDO: Jie)for llama model, need to change to fit other models
                # Call the AsyncJsonWriter to write the data to the file
                # if self.layer_id == 0:
                #     print(f"layer {self.layer_id} step {real_step} kv_cache_elem {key_cache.numel()} elem_size {key_cache.element_size()}", flush=True)
                self.AsyncJsonWriter.record_kv_write_pattern(kv_write_pattern_info)
        start_2 = time.time()

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        # print(f"FlashAttentionImpl: output.is_cuda={output.is_cuda}, decode_query.is_cuda={decode_query.is_cuda}, query={query.is_cuda}, key={key.is_cuda}, value={value.is_cuda}", flush=True)

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        start_attn = time.time()
        torch.cuda.nvtx.range_push("attn_kernel")
        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                out = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_seq_len,
                    max_seqlen_k=prefill_meta.max_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                )
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                # TODO(Hai) this triton kernel has regression issue (broke) to
                # deal with different data types between KV and FP8 KV cache,
                # to be addressed separately.
                # print(f"FlashAttentionImpl: PagedAttention.forward_prefix block_tables.numel()={prefill_meta.block_tables.numel()} block_tables={prefill_meta.block_tables}", flush=True)
                output[:num_prefill_tokens] = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.subquery_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.context_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                    self.sliding_window[0],
                )
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            # print(f"FlashAttentionImpl: PagedAttention.forward_decode block_tables.numel()={decode_meta.block_tables.numel()} block_tables={decode_meta.block_tables}", flush=True)
            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                decode_meta.block_tables,
                decode_meta.seq_lens_tensor,
                decode_meta.max_seq_len,
                attn_metadata.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
                self.k_cache_read_pattern,
                self.v_cache_read_pattern
            )

            if self.step > 3:
                real_step = self.step - 3 
                dur = (time.time() - start_1) * 1e6
                kv_read_pattern_info = {
                        "layer_id": self.layer_id,
                        "step": real_step,
                        "start_time": start_1,
                        "cpu_duration(us)": dur,
                        "kcache_start_addr": key_cache.data_ptr(),
                        "vcache_start_addr": value_cache.data_ptr(),
                        "kv_cache_range": kv_cache_range,
                        'seq_lens': seq_lens,
                        'req_stride': 240,
                        'block_acc_stride': 20,
                        "k_read_pattern_record": self.k_cache_read_pattern[:num_decode_tokens, :].cpu().tolist(),
                        "v_read_pattern_record": self.v_cache_read_pattern[:num_decode_tokens, :].cpu().tolist(),
                }

                # write the read pattern to the file
                self.AsyncJsonWriter.record_kv_read_pattern(kv_read_pattern_info)

        torch.cuda.nvtx.range_pop()
        self.attention_time = time.time() - start_attn
        # print(f"FlashAttentionImpl: layer {self.layer_id} step {self.step} attn_time {self.attention_time * 1e6}, wr_cache={(start_2 - start_1) * 1e6}, other={(start_attn - start_2) * 1e6}, total = {(time.time() - start_1) * 1e6}", flush=True)
        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)
