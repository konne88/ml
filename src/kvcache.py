from torch import Tensor
from llm import *

T = TypeVar('T')
Embedding = TypeVar('Embedding', bound="Index")


class Index(Protocol):
    @property
    def index(self) -> int:
        ...


Cache = dict[int, dict[int, dict[int, T]]]


def cached(cache: Cache, input_index: int, decoder_index: int, head_index: int, get_t: Callable[[], T]) -> T:
    try:
        return cache[input_index][decoder_index][head_index]
    except KeyError:
        t = get_t()
        input_cache = cache.get(input_index, {})
        decoder_cache = input_cache.get(decoder_index, {})
        decoder_cache[head_index] = t
        input_cache[decoder_index] = decoder_cache
        cache[input_index] = input_cache
        return t


def kvcache(transformer: Transformer[Embedding, Query, Key, Value]) -> Transformer[Embedding, Query, Key, Value]:
    key_cache: Cache = {}
    value_cache: Cache = {}
    return Transformer(
        embed=transformer.embed,
        unembed=transformer.unembed,
        decoders=[
            Decoder(
                heads=[
                    AttentionHead(
                        score=Score(
                            combine=head.score.combine,
                            query=head.score.query,
                            key=lambda input, decoder_index=decoder_index, head_index=head_index, head=head: cached(
                                key_cache, input.index, decoder_index, head_index, lambda: head.score.key(input))
                        ),
                        value=lambda input, decoder_index=decoder_index, head_index=head_index, head=head: cached(
                            value_cache, input.index, decoder_index, head_index, lambda: head.value(input))
                    )
                    for head_index, head in enumerate(decoder.heads)
                ],
                process=decoder.process
            )
            for decoder_index, decoder in enumerate(transformer.decoders)
        ]
    )
