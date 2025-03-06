from typing import List, Tuple, Optional
import random, math
from transformers import PreTrainedTokenizerBase
import json
import numpy as np

EPS = 1e-8
class ZipfGenerator:
    def __init__(
        self, min: int, max: int, theta: float, seed: int
    ) -> None:
        self._min = min
        self._max = max
        self._items = max - min + 1
        self._theta = theta
        self._zeta_2 = self._zeta(2, self._theta)
        self._alpha = 1.0 / (1.0 - self._theta)
        self._zetan = self._zeta(self._items, self._theta)
        self._eta = (1 - np.power(2.0 / self._items, 1 - self._theta)) / (
            1 - self._zeta_2 / (self._zetan + EPS)
        )
        self._seed = seed
        self._generator = np.random.RandomState(seed)

    def _zeta(self, count: float, theta: float) -> float:
        return np.sum(1 / (np.power(np.arange(1, count), theta)))

    def _next(self) -> int:
        u = self._generator.random_sample()
        uz = u * self._zetan

        if uz < 1.0:
            return self._min

        if uz < 1.0 + np.power(0.5, self._theta):
            return self._min + 1

        return self._min + int(
            (self._items) * np.power(self._eta * u - self._eta + 1, self._alpha)
        )

    def next(self) -> int:
        retval = self._next()
        # if self._scramble:
        #     retval = self._min + hash(str(retval) + str(self._seed)) % self._items
        return retval

def fixed_request_length_generator(num_requests, input_len, output_len = None) -> List[Tuple[str, int, int]]:
    assert input_len > 0, "The input length of a request should be > 0"
    prompt = "hello" * (input_len - 1)
    if output_len is None:
        output_len = 1
    requests = [(prompt, input_len, output_len) for _ in range(num_requests)]
    return requests

# Find a paper with a smarter generator

def uniform_request_length_generator(num_requests, min_len, max_len, prefill_to_decode_ratio) -> List[Tuple[str, int, int]]:
    assert min_len > 4, "The minimum input length of a request should be > 4"

    requests = []
    for _ in range(num_requests):
        total_tokens = random.uniform(min_len, max_len)
        decode_tokens = math.ceil(total_tokens / (1 + prefill_to_decode_ratio))
        if decode_tokens < 1:
            # Ensure that the decode_tokens is at least 1, we can make the ratio larger for prefill only
            decode_tokens = 1
        prefill_tokens = total_tokens - decode_tokens
        
        prefill_tokens = int(prefill_tokens)
        decode_tokens = int(decode_tokens)
        assert prefill_tokens >= 1 and decode_tokens >= 1, "Adjust the min_len, max_len, and prefill_to_decode_ratio to generate valid input lengths"
        prompt = "hello" * (prefill_tokens - 1)
        requests.append((prompt, prefill_tokens, decode_tokens))
        
    return requests

def zipf_request_length_generator(num_requests, min_len, max_len, seed, prefill_to_decode_ratio) -> List[Tuple[str, int, int]]:
    assert min_len > 4, "The minimum input length of a request should be > 4"
    zipf_generator = ZipfGenerator(min_len, max_len, 0.6, seed)
    requests = []
    for _ in range(num_requests):
        total_tokens = zipf_generator.next()
        decode_tokens = math.ceil(total_tokens / (1 + prefill_to_decode_ratio))
        if decode_tokens < 1:
            # Ensure that the decode_tokens is at least 1, we can make the ratio larger for prefill only
            decode_tokens = 1
        prefill_tokens = total_tokens - decode_tokens
        
        prefill_tokens = int(prefill_tokens)
        decode_tokens = int(decode_tokens)
        assert prefill_tokens >= 1 and decode_tokens >= 1, "Adjust the min_len, max_len, and prefill_to_decode_ratio to generate valid input lengths"
        prompt = "hello" * (prefill_tokens - 1)
        requests.append((prompt, prefill_tokens, decode_tokens))
        
    return requests

def sample_ShareGPT_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
    max_context_len: Optional[int] = 2048,
    is_test_mode: Optional[bool] = False
) -> List[Tuple[str, int, int]]:
    #if fixed_output_len is not None and fixed_output_len < 4:
    #   raise ValueError("output_len too small")
    
    # sample with the ShareGPT_V3 dataset
    min_promp_len = 4
    min_out_len = 1024
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    total_reqs = num_requests #when min_promp_len = 512
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        filtered_dataset_len = len(filtered_dataset) 
        if filtered_dataset_len == total_reqs:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        if filtered_dataset_len >= num_requests:
        # if filtered_dt_len >= num_requests and filtered_dt_len < total_reqs - 1:
            output_len = 1 # to make these requests are prefill-only requests
        else:
            output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len

        if is_test_mode:
            if prompt_len < min_promp_len:
                # Prune too short sequences.
                continue
        else:
            if prompt_len < min_promp_len or output_len < min_out_len:
                # Prune too short sequences.
                continue
        
        if prompt_len > max_context_len or prompt_len + output_len > max_context_len:
            # Prune too long sequences.
            continue 
        filtered_dataset.append((prompt, prompt_len, output_len))
 
    return filtered_dataset


def sample_LEval_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        fixed_output_len: Optional[int],
        max_context_len: Optional[int] = 2048,
    ) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    
    # sample with the LEval dataset 
    min_promp_len = 1024
    min_out_len = 1024
    # Load the dataset.
    dataset = []
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            json_dataset = json.loads(line)
            dataset.append((json_dataset['prompt'], json_dataset['output']))
    
    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                        ) if fixed_output_len is None else fixed_output_len
        if prompt_len < min_promp_len or output_len < min_out_len:
            # Prune too short sequences.
            continue

        if prompt_len + output_len > max_context_len:
            # Prune too short sequences or too long sequences.
            continue

        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset
