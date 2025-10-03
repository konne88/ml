from typing import Protocol, TypeVar
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterator, Generic, List, TypeVar
from utils import prefixes

Token = int
Embedding = TypeVar('Embedding')
Value = TypeVar('Value', bound="VectorSpace")


class VectorSpace(Protocol):
    def __add__(self: Value, other: Value) -> Value:
        ...

    def __mul__(self: Value, other: float) -> Value:
        ...

    def __truediv__(self: Value, other: float) -> Value:
        ...


@dataclass
class AttentionHead(Generic[Embedding, Value]):
    score: Callable[[Embedding, Embedding], float]
    value: Callable[[Embedding], Value]


@dataclass
class Decoder(Generic[Embedding, Value]):
    heads: List[AttentionHead[Embedding, Value]]
    process: Callable[[Embedding, List[Value]], Embedding]


@dataclass
class Transformer(Generic[Embedding, Value]):
    embed: Callable[[int, Token], Embedding]
    decoders: List[Decoder[Embedding, Value]]
    unembed: Callable[[Embedding], Token]


Query = TypeVar('Query')
Key = TypeVar('Key')


class Score(Generic[Embedding, Query, Key]):
    def __init__(self,
                 query: Callable[[Embedding], Query],
                 key: Callable[[Embedding], Key],
                 combine: Callable[[Query, Key], float]):
        self.query = query
        self.key = key
        self.combine = combine

    def __call__(self, current: Embedding, other: Embedding):
        return self.combine(self.query(current), self.key(other))


def attend_to(inputs: List[Embedding],
              score: Callable[[Embedding, Embedding], float],
              value: Callable[[Embedding], Value]) -> Value:
    result: Value = 0  # type: ignore
    total_score = 0
    last_input = inputs[-1]

    for input in inputs:
        result += value(input) * score(last_input, input)
        total_score += score(last_input, input)
    return result / total_score


def decode(layer: Decoder[Embedding, Value], embeddings: List[Embedding]) -> Embedding:
    current = embeddings[-1]
    focused = [attend_to(embeddings, head.score, head.value)
               for head in layer.heads]
    return layer.process(current, focused)


def transform(transformer: Transformer[Embedding, Value], tokens: List[Token]) -> Token:
    embeddings = [transformer.embed(index, token)
                  for (index, token) in enumerate(tokens)]
    for layer in transformer.decoders:
        embeddings = [decode(layer, prefix)
                      for prefix in prefixes(embeddings)]
    return transformer.unembed(embeddings[-1])


def autocomplete(transformer: Transformer[Embedding, Value], max_seq_len: int, tokens: List[Token]) -> Iterator[Token]:
    yield from tokens
    num_prompt_tokens = len(tokens)
    for _ in range(max_seq_len - num_prompt_tokens):
        next_token = transform(transformer, tokens)
        tokens.append(next_token)
        yield next_token
