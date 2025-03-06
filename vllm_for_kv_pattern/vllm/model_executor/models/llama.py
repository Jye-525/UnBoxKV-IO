# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.utils import is_hip
import time
from vllm.utils import instrument_w_nvtx
from vllm.logger import init_logger
from vllm.async_json_writer import AsyncJsonWriter
logger = init_logger(__name__)


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QKVParallelLinear] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        # print(f"LlamaMLP MergedColumnParallelLinear input.shape={x.shape} output.shape={gate_up.shape}", flush=True) 
        x = self.act_fn(gate_up)
        # print(f"LlamaMLP act_fn input.shape={gate_up.shape} output.shape={x.shape}", flush=True) 
        x, _ = self.down_proj(x)
        # print(f"LlamaMLP RowParallelLinear output.shape={x.shape}", flush=True) 
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        sliding_window: Optional[int] = None,
        layer_id: Optional[int] = None,
        AsyncJsonWriter: Optional[AsyncJsonWriter] = None,
        kv_write_pattern_trace_record: Optional[torch.Tensor] = None,
        k_cache_read_pattern: Optional[torch.Tensor] = None,
        v_cache_read_pattern: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # This will be overwritten by model initialization if we are using it.
        # N.B. currently we only support per tensor scalar scaling factors
        # & only applicable to ROCm (AMD GPU).
        # The scaling factor convention we are assuming is
        # quantized_value * scaling_factor ~= true_value
        # which is consistent with the practice of setting
        # scaling_factor = tensor_amax / FPtype_max
        self.kv_scale = 1.0

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )
        # print(f"LlamaAttention QKVParallelLinear hidden_size={hidden_size} head_dim={self.head_dim} total_num_heads={self.total_num_heads} total_num_kv_heads={self.total_num_kv_heads}", flush=True)
        # print(f"LlamaAttention self.q_size={self.q_size} self.kv_size={self.kv_size} bias={bias} quant_conf={quant_config}", flush=True)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              sliding_window=sliding_window,
                              layer_id=layer_id,
                              AsyncJsonWriter=AsyncJsonWriter,
                              kv_write_pattern_trace_record=kv_write_pattern_trace_record,
                              k_cache_read_pattern=k_cache_read_pattern,
                              v_cache_read_pattern=v_cache_read_pattern)
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        torch.cuda.nvtx.range_push("qkv_proj")
        qkv, _ = self.qkv_proj(hidden_states)
        torch.cuda.nvtx.range_pop()
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        torch.cuda.nvtx.range_push("attention")
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                                self.kv_scale)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("out_proj")
        output, _ = self.o_proj(attn_output)
        torch.cuda.nvtx.range_pop() 
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: Optional[int] = None,
        AsyncJsonWriter: Optional[AsyncJsonWriter] = None,
        kv_write_pattern_trace_record: Optional[torch.Tensor] = None,
        k_cache_read_pattern: Optional[torch.Tensor] = None,
        v_cache_read_pattern: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        sliding_window = getattr(config, "sliding_window", None)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            sliding_window=sliding_window,
            layer_id=layer_id,
            AsyncJsonWriter=AsyncJsonWriter,
            kv_write_pattern_trace_record=kv_write_pattern_trace_record,
            k_cache_read_pattern=k_cache_read_pattern,
            v_cache_read_pattern=v_cache_read_pattern
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            # print(f"LlamaDecoderLayer input_layernorm input.shape={residual.shape} output.shape={hidden_states.shape}", flush=True)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
            # print(f"LlamaDecoderLayer input_layernorm-x output1.shape={hidden_states.shape} output2.shape={residual.shape}", flush=True)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        # print(f"LlamaDecoderLayer self_attn positions.shape={positions.shape} output.shape={hidden_states.shape}", flush=True) 

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        # print(f"LlamaDecoderLayer post_attention_layernorm output1.shape={hidden_states.shape} output2.shape={residual.shape}", flush=True)
        hidden_states = self.mlp(hidden_states)
        # print(f"LlamaDecoderLayer mlp output.shape={hidden_states.shape}", flush=True)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        # async_json_writer: Optional[AsyncJsonWriter] = None, 
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        # self.layers = nn.ModuleList([
        #     LlamaDecoderLayer(config, quant_config)
        #     for _ in range(config.num_hidden_layers)
        # ])
        # create the AsyncJsonWriter object
        # self.async_json_writer = async_json_writer
        self.async_json_writer = AsyncJsonWriter("./kv_pattern/kv_write_pattern_trace.json", "./kv_pattern/kv_read_pattern_trace.json")
        # create the kv_write_pattern_trace_record tensor
        # use max_model_length as the maximum number of tokens in the model
        # each token has 6 elements: [0]=key_start, [1]=key_end, [2]=value_start, [3]=value_end, [4]=start_time, [5]=end_time
        self.kv_write_pattern_trace_record = torch.zeros((config.max_position_embeddings, 5), dtype=torch.int64, device='cuda:0')
        # 4 * max_blocks * 3 = 4 * 20 * 3 = 240
        self.k_cache_read_pattern = torch.zeros((10, 240), dtype=torch.int64, device='cuda:0')
        self.k_cache_read_pattern.fill_(-1)
        self.v_cache_read_pattern = torch.zeros((10, 240), dtype=torch.int64, device='cuda:0')
        self.v_cache_read_pattern.fill_(-1)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, quant_config, id, self.async_json_writer, 
                              self.kv_write_pattern_trace_record, 
                              self.k_cache_read_pattern,
                              self.v_cache_read_pattern)
            for id in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None
        logger.info(f"before LlamaModel forward() input_ids.shape={input_ids.shape}, positions.shape={positions.shape} hidden_states.shape={hidden_states.shape} residual={residual}")
        start = time.time()
        for i in range(len(self.layers)):
            # print(f"LlamaModel forward() layer-{i} kv_caches[i].shape={None if kv_caches[i] is None else kv_caches[i].shape}", flush=True)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
            # print(f"LlamaModel forward() layer-{i} positions.shape={positions.shape} residual.shape={residual.shape}", flush=True)
        end = time.time()
        # print(f"LlamaModel forward() time cost (us): {(end-start) * 1e6}", flush=True)    
        hidden_states, _ = self.norm(hidden_states, residual)
        # self.async_json_writer.close()
        # print(f"LlamaModel last norm hidden_states.shape={hidden_states.shape} residual.shape={residual.shape}", flush=True)
        return hidden_states
    
    def __del__(self):
        self.async_json_writer.close()
        # print(f"LLamaModel close successfully........", flush=True)


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, quant_config, lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    @instrument_w_nvtx
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    @instrument_w_nvtx
    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        # print(f"LlamaForCausalLM compute_logits(): hidden_states.shape={hidden_states.shape} logits.shape={logits.shape}", flush=True)
        return logits

    @instrument_w_nvtx
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")
