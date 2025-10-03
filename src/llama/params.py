from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
from torch import Tensor

from torch.nn.functional import linear
from safetensors import safe_open
from llm import Token

WeightDict = Dict[str, Tensor]


def myLinear(weights: Tensor) -> Callable[[Tensor], Tensor]:
    return lambda x: linear(x, weights)


def loadLinear(weight_dict: WeightDict, path: str) -> Callable[[Tensor], Tensor]:
    return myLinear(weight_dict[path].float())


def loadLinearHeads(weight_dict: WeightDict, n_heads: int, path: str) -> List[Callable[[Tensor], Tensor]]:
    weights = weight_dict[path].float()
    (_, dim) = weights.shape
    weights = weights.t().view(dim, n_heads, -1)
    return [myLinear(weights[:, head_index, :].t()) for head_index in range(n_heads)]


def loadLinearGroupHeads(weight_dict: WeightDict, n_groups: int, n_heads: int, path: str) -> List[Callable[[Tensor], Tensor]]:
    weights = weight_dict[path].float()
    (_, dim) = weights.shape
    weights = weights.t().view(dim, n_groups, -1)
    return [myLinear(weights[:, group_index, :].t())
            for group_index in range(n_groups)
            for _ in range(n_heads // n_groups)]


def norm(x: Tensor) -> Tensor:
    eps: float = 1e-06
    return x * torch.rsqrt(x.pow(2).mean() + eps)  # type:ignore


def loadNorm(weight_dict: WeightDict, path: str) -> Callable[[Tensor], Tensor]:
    weights = weight_dict[path].float()
    return lambda x: norm(x) * weights


def loadEmbedding(weight_dict: WeightDict, path: str) -> Callable[[Token], Tensor]:
    weights = weight_dict[path].float()
    return lambda token: weights[token]


@dataclass
class LlamaLayerParams:
    attention_norm: Callable[[Tensor], Tensor]
    query: List[Callable[[Tensor], Tensor]]
    key: List[Callable[[Tensor], Tensor]]
    value: List[Callable[[Tensor], Tensor]]
    output: Callable[[Tensor], Tensor]

    process_norm: Callable[[Tensor], Tensor]
    gate: Callable[[Tensor], Tensor]
    down: Callable[[Tensor], Tensor]
    up: Callable[[Tensor], Tensor]


@dataclass
class LlamaParams:
    embed: Callable[[Token], Tensor]
    unembed_norm: Callable[[Tensor], Tensor]
    unembed: Callable[[Tensor], Tensor]

    layers: List[LlamaLayerParams]

    n_heads: int
    head_dim: int


def llama7BParams(path="/home/vscode/.llama/checkpoints/Llama-2-7b-chat/consolidated.00.pth") -> LlamaParams:
    n_layers = 32
    n_heads = 32
    dim = 4096

    print("Start loading weights")
    weight_dict = torch.load(path, map_location="cpu")
    print("Done loading weights")

    return LlamaParams(
        embed=loadEmbedding(weight_dict, "tok_embeddings.weight"),
        unembed_norm=loadNorm(weight_dict, "norm.weight"),
        unembed=loadLinear(weight_dict, "output.weight"),
        layers=[
            LlamaLayerParams(
                attention_norm=loadNorm(
                    weight_dict, f"layers.{layer_id}.attention_norm.weight"),
                key=loadLinearHeads(
                    weight_dict, n_heads, f"layers.{layer_id}.attention.wk.weight"),
                value=loadLinearHeads(
                    weight_dict, n_heads, f"layers.{layer_id}.attention.wv.weight"),
                query=loadLinearHeads(
                    weight_dict, n_heads, f"layers.{layer_id}.attention.wq.weight"),
                output=loadLinear(
                    weight_dict, f"layers.{layer_id}.attention.wo.weight"),
                process_norm=loadNorm(
                    weight_dict, f"layers.{layer_id}.ffn_norm.weight"),
                gate=loadLinear(
                    weight_dict, f"layers.{layer_id}.feed_forward.w1.weight"),
                down=loadLinear(
                    weight_dict, f"layers.{layer_id}.feed_forward.w2.weight"),
                up=loadLinear(
                    weight_dict, f"layers.{layer_id}.feed_forward.w3.weight"),
            )
            for layer_id in range(n_layers)],
        head_dim=dim // n_heads,
        n_heads=n_heads
    )
