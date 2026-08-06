"""Small R 4.4-compatible RNG subset used for the paper's center holdouts."""

from __future__ import annotations

import math

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF


class RMersenneTwister:
    """Replicate R's Mersenne-Twister initialization and rejection sampling."""

    def __init__(self, seed: int) -> None:
        """Initialize from the integer accepted by R's ``set.seed``."""
        value = seed & 0xFFFFFFFF
        for _ in range(50):
            value = (69069 * value + 1) & 0xFFFFFFFF
        self._state: list[int] = []
        for _ in range(_N):
            value = (69069 * value + 1) & 0xFFFFFFFF
            self._state.append(value)
        self._index = _N
        self._next_uint32()

    def _twist(self) -> None:
        for position in range(_N - _M):
            value = (self._state[position] & _UPPER_MASK) | (
                self._state[position + 1] & _LOWER_MASK
            )
            self._state[position] = (
                self._state[position + _M] ^ (value >> 1) ^ (_MATRIX_A if value & 1 else 0)
            )
        for position in range(_N - _M, _N - 1):
            value = (self._state[position] & _UPPER_MASK) | (
                self._state[position + 1] & _LOWER_MASK
            )
            self._state[position] = (
                self._state[position + _M - _N] ^ (value >> 1) ^ (_MATRIX_A if value & 1 else 0)
            )
        value = (self._state[-1] & _UPPER_MASK) | (self._state[0] & _LOWER_MASK)
        self._state[-1] = self._state[_M - 1] ^ (value >> 1) ^ (_MATRIX_A if value & 1 else 0)
        self._index = 0

    def _next_uint32(self) -> int:
        if self._index >= _N:
            self._twist()
        value = self._state[self._index]
        self._index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & 0xFFFFFFFF

    def uniform(self) -> float:
        """Return one R-compatible uniform draw in [0, 1)."""
        return self._next_uint32() / 4294967296.0

    def _index_rejection(self, size: int) -> int:
        bits = math.ceil(math.log2(size))
        mask = (1 << bits) - 1
        while True:
            value = math.floor(self.uniform() * 65536) & mask
            if value < size:
                return value

    def sample_without_replacement(self, population: int, sample_size: int) -> list[int]:
        """Match ``sample.int(population, sample_size, replace = FALSE)``."""
        if not 0 <= sample_size <= population:
            raise ValueError("sample size must be between zero and population")
        candidates = list(range(population))
        result: list[int] = []
        remaining = population
        for _ in range(sample_size):
            position = self._index_rejection(remaining)
            result.append(candidates[position])
            remaining -= 1
            candidates[position] = candidates[remaining]
        return result


def center_holdout_indices(
    population: int,
    mode: str,
    *,
    seed: int = 20_240_110,
) -> list[int]:
    """Return the paper-order Mixtral/happiness center holdout indices."""
    target_call = {"ntp": 9, "fa": 21}.get(mode)
    if target_call is None:
        raise ValueError(f"unsupported mode: {mode}")
    generator = RMersenneTwister(seed)
    result: list[int] = []
    for _ in range(target_call):
        result = generator.sample_without_replacement(population, 20)
    return result
