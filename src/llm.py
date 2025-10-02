from dataclasses import dataclass
from typing import Callable, Generic, Iterator, List, Tuple, TypeVar
from utils import normalize, weighted_average


Token = int
Embedding = TypeVar('Embedding')
Query = TypeVar('Query')
Key = TypeVar('Key')
Value = TypeVar('Value')


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


def attend_to(head: AttentionHead[Embedding, Query, Key, Value], keys: List[Key], values: List[Value], current: Embedding) -> Value:
    query = head.score.query(current)
    keys.append(head.score.key(current))
    values.append(head.value(current))
    scores = [head.score.combine(query, key) for key in keys]
    scores = normalize(scores)
    return weighted_average(scores, values)


def decode(layer: Decoder[Embedding, Query, Key, Value], keys: List[List[Key]], values: List[List[Value]], current: Embedding) -> Embedding:
    focused = [attend_to(head, keys[head_index], values[head_index], current)
               for (head_index, head) in enumerate(layer.heads)]
    return layer.process(current, focused)


def transform(transformer: Transformer[Embedding, Query, Key, Value], keys: List[List[List[Key]]], values: List[List[List[Value]]], index: int, token: Token) -> Token:
    current = transformer.embed(index, token)
    for layer_index, layer in enumerate(transformer.decoders):
        current = decode(
            layer, keys[layer_index], values[layer_index], current)
    return transformer.unembed(current)


def autocomplete(transformer: Transformer[Embedding, Query, Key, Value], max_seq_len: int, tokens: List[Token]) -> Iterator[Token]:
    keyCache: List[List[List[Key]]] = emptyCache(transformer)
    valueCache: List[List[List[Value]]] = emptyCache(transformer)
    n_prompt_tokens = len(tokens)

    for i in range(max_seq_len - 1):
        token = tokens[i]
        yield token
        next_token = transform(transformer, keyCache, valueCache, i, token)
        if (i + 1 >= n_prompt_tokens):
            tokens.append(next_token)


A = TypeVar("A")


def emptyCache(transformer: Transformer[Embedding, Query, Key, Value]) -> List[List[List[A]]]:
    return [[[] for _ in layer.heads] for layer in transformer.decoders]
