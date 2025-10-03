from typing import Protocol, TypeVar
from dataclasses import dataclass
from typing import Callable, Iterator, Generic, List, TypeVar

Token = int
Embedding = TypeVar('Embedding')
Query = TypeVar('Query')
Key = TypeVar('Key')
Value = TypeVar('Value', bound="VectorSpace")


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


@dataclass
class AttentionHead(Generic[Embedding, Query, Key, Value]):
    score: Score[Embedding, Query, Key]
    value: Callable[[Embedding], Value]


@dataclass
class Decoder(Generic[Embedding, Query, Key, Value]):
    heads: List[AttentionHead[Embedding, Query, Key, Value]]
    process: Callable[[Embedding, List[Value]], Embedding]


@dataclass
class Transformer(Generic[Embedding, Query, Key, Value]):
    embed: Callable[[int, Token], Embedding]
    decoders: List[Decoder[Embedding, Query, Key, Value]]
    unembed: Callable[[Embedding], Token]


class VectorSpace(Protocol):
    def __add__(self: Value, other: Value) -> Value:
        ...

    def __mul__(self: Value, other: float) -> Value:
        ...

    def __truediv__(self: Value, other: float) -> Value:
        ...


def attend_to(inputs: List[Embedding],
              score: Callable[[Embedding, Embedding], float],
              value: Callable[[Embedding], Value]) -> Value:
    result: Value = 0.0  # type: ignore
    total_score = 0.0
    last_input = inputs[-1]

    for input in inputs:
        result += value(input) * score(last_input, input)
        total_score += score(last_input, input)
    return result / total_score


def decode(layer: Decoder[Embedding, Query, Key, Value],
           embeddings: List[Embedding]) -> Embedding:
    focused = [attend_to(embeddings, head.score, head.value)
               for (_, head) in enumerate(layer.heads)]
    current = embeddings[-1]
    return layer.process(current, focused)


def transform(transformer: Transformer[Embedding, Query, Key, Value], embeddings: List[List[Embedding]], index: int, token: Token) -> Token:
    current = transformer.embed(index, token)
    for layer_index, layer in enumerate(transformer.decoders):
        embeddings[layer_index].append(current)
        current = decode(layer, embeddings[layer_index])
    return transformer.unembed(current)


def autocomplete(transformer: Transformer[Embedding, Query, Key, Value], max_seq_len: int, tokens: List[Token]) -> Iterator[Token]:
    prompt_len = len(tokens)
    embeddings: List[List[Embedding]] = [[] for _ in transformer.decoders]

    for i in range(max_seq_len - 1):
        token = tokens[i]
        yield token
        next_token = transform(transformer, embeddings, i, token)
        if (i + 1 >= prompt_len):
            tokens.append(next_token)
