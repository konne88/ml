# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Callable, List

import torch
from torch import Tensor
import torch.nn.functional as F

from llm import Score, Decoder, Transformer, AttentionHead, Token
from llama.params import LlamaLayerParams, LlamaParams


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@dataclass
class Embedding:
    index: int
    layer_index: int
    data: Tensor


Cache = dict[int, dict[int, dict[int, Tensor]]]

key_cache: Cache = {}
value_cache: Cache = {}


def cached(cache: Cache, input_index: int, layer_index: int, head_index: int, get_tensor: Callable[[], Tensor]):
    try:
        return cache[input_index][layer_index][head_index]
    except KeyError:
        tensor = get_tensor()
        input_cache = cache.get(input_index, {})
        layer_cache = input_cache.get(layer_index, {})
        layer_cache[head_index] = tensor
        input_cache[layer_index] = layer_cache
        cache[input_index] = input_cache
        return tensor


def llamaHead(params: LlamaLayerParams, head_index: int, head_dim: int, freqs_cis: Tensor) -> AttentionHead[Embedding, Tensor, Tensor, Tensor]:
    def apply_rotary_emb(index: int, x: Tensor) -> Tensor:
        x = torch.view_as_complex(x.view(-1, 2))   # type:ignore
        return torch.view_as_real(x * freqs_cis[index]).flatten()

    def query(current: Embedding) -> Tensor:
        return apply_rotary_emb(current.index, params.query[head_index](params.attention_norm(current.data)))

    def key(other: Embedding) -> Tensor:
        return cached(key_cache, other.index, other.layer_index, head_index, lambda:
                      apply_rotary_emb(other.index, params.key[head_index](
                          params.attention_norm(other.data))))

    def value(other: Embedding) -> Tensor:
        return cached(value_cache, other.index, other.layer_index, head_index, lambda:
                      params.value[head_index](params.attention_norm(other.data)))

    def combine(query: Tensor, key: Tensor) -> float:
        return math.exp((torch.matmul(query, key).item() / math.sqrt(head_dim)))

    return AttentionHead(
        score=Score(
            query=query,
            key=key,
            combine=combine
        ),
        value=value,
    )


def llamaLayer(layer_index: int, params: LlamaLayerParams, n_heads: int, head_dim: int, freqs_cis: Tensor) -> Decoder[Embedding, Tensor, Tensor, Tensor]:
    def process(current: Embedding, focused: List[Tensor]) -> Embedding:
        h = torch.stack(focused).flatten()
        h = current.data + params.output(h)
        n = params.process_norm(h)
        out = h + params.down(F.silu(params.gate(n)) * params.up(n))
        return Embedding(index=current.index, layer_index=layer_index+1, data=out)

    return Decoder(
        heads=[llamaHead(params, head_index, head_dim, freqs_cis)
               for head_index in range(n_heads)],
        process=process
    )


def llama(params: LlamaParams, max_seq_len: int, temperature: float = 0.7, top_p: float = 0.95) -> Transformer[Embedding, Tensor, Tensor, Tensor]:
    freqs_cis = precompute_freqs_cis(params.head_dim, max_seq_len * 2)

    def embed(index: int, token: Token) -> Embedding:
        return Embedding(index=index, layer_index=0, data=params.embed(token))

    def unembed(embedding: Embedding) -> Token:
        logits = params.unembed(params.unembed_norm(embedding.data))
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(sample_top_p(probs, top_p).item())

    return Transformer(
        embed=embed,
        unembed=unembed,
        decoders=[llamaLayer(layer_index, layer, params.n_heads, params.head_dim, freqs_cis)
                  for layer_index, layer in enumerate(params.layers)]
    )
