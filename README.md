# UnBoxKV-IO

Speeding up each inference request is instrumental in achieving high throughput and latency at scale. KV Cache is usually avoid redundant recomputation in each decode iteration. The free GPU space available to the KV cache is a scarce resource that needs to be managed in an efficient way in order to minimize the overhead of redundant recomputations. This work characterize the impact of KV caching. Specifically, we instrument vLLM to measure and analyze fine-grain KV cache access patterns during different inference stages (prefill, decode). We also study the recomputation and swap overhead for handling KV cache overflow problem in several scenarios that involve concurrent inference requests using several benchmarks. The results show some interesting observations and insights for optimizing the KV cache management and batching strategies.



