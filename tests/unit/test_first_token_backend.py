from __future__ import annotations

import re
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="requires the culture extra")

from culture.backend import TransformersBackend  # noqa: E402
from culture.registry import CULTURE_MODELS  # noqa: E402

PIECE = re.compile(r" ?[A-Za-z]+|[^A-Za-z ]| ")

VOCABULARY = 8


class _Tokenizer:
    def __init__(self, *, fold_case: bool = False) -> None:
        self._fold_case = fold_case
        self._ids: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        pieces = PIECE.findall(text.lower() if self._fold_case else text)
        return [self._ids.setdefault(piece, len(self._ids)) for piece in pieces]


class _Model:
    device = "cpu"

    def __init__(self) -> None:
        self.kept: list[int] = []

    def __call__(self, *, input_ids: Any, attention_mask: Any, logits_to_keep: int) -> Any:
        self.kept.append(logits_to_keep)
        rows = input_ids.shape[0]
        shares = torch.arange(1, VOCABULARY + 1, dtype=torch.float32).repeat(rows, 1)
        shares = shares + input_ids[:, :1].float()
        shares = shares / shares.sum(dim=-1, keepdim=True)
        return SimpleNamespace(logits=torch.log(shares).unsqueeze(1))


def _encode(prompts: Sequence[str]) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([[len(prompt)] for prompt in prompts]),
        "attention_mask": torch.ones((len(prompts), 1), dtype=torch.long),
    }


def _backend(
    tokenizer: _Tokenizer | None = None, model: _Model | None = None
) -> TransformersBackend:
    backend = object.__new__(TransformersBackend)
    backend.__dict__.update(
        _torch=torch,
        _oom_backoffs=0,
        _batch_size=4,
        _model_spec=CULTURE_MODELS["qwen3_vl_2b"],
        _tokenizer=tokenizer or _Tokenizer(),
        _model=model or _Model(),
        _encode=_encode,
    )
    return backend


def test_each_continuation_resolves_to_the_one_token_that_follows_the_stub() -> None:
    tokenizer = _Tokenizer()

    resolved = _backend(tokenizer).continuation_token_ids("Answer:", (" A", " B", " C"))

    assert resolved == [2, 3, 4]
    assert tokenizer.encode("Answer: B") == [0, 1, 3]


def test_a_continuation_of_more_than_one_token_is_refused() -> None:
    with pytest.raises(RuntimeError, match="as one token"):
        _backend().continuation_token_ids("Answer:", (" A", " B."))


def test_two_continuations_on_the_same_token_are_refused() -> None:
    with pytest.raises(RuntimeError, match="the same token"):
        _backend(_Tokenizer(fold_case=True)).continuation_token_ids("Answer:", (" A", " a"))


def test_the_first_token_probabilities_are_read_at_the_given_ids_without_renormalising() -> None:
    model = _Model()
    backend = _backend(model=model)

    rows = backend.first_token_batch(["ab", "abcd"], [0, 7])

    first = sum(range(1, VOCABULARY + 1)) + VOCABULARY * 2
    second = sum(range(1, VOCABULARY + 1)) + VOCABULARY * 4
    assert rows[0] == pytest.approx([3 / first, 10 / first])
    assert rows[1] == pytest.approx([5 / second, 12 / second])
    assert model.kept == [1]
    assert backend.first_token_batch([], [0, 7]) == []


def test_the_token_ids_stay_with_every_half_of_a_batch_that_did_not_fit() -> None:
    backend = _backend()
    seen: list[tuple[int, tuple[int, ...]]] = []

    def run(prompts: Sequence[str], *, token_ids: Sequence[int]) -> list[list[float]]:
        if len(prompts) > 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        seen.append((len(prompts), tuple(token_ids)))
        return [[float(len(prompt))] for prompt in prompts]

    backend.__dict__["_first_token_batch"] = run

    rows = backend.first_token_batch(["a", "bb", "ccc"], [5])

    assert rows == [[1.0], [2.0], [3.0]]
    assert seen == [(1, (5,))] * 3
