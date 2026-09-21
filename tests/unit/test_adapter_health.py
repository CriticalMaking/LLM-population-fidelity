from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from culture import adapter_health


def safetensors_bytes(tensors: dict[str, np.ndarray]) -> bytes:
    """A minimal float32 safetensors file, so the reader is exercised for real."""
    header: dict[str, Any] = {}
    body = bytearray()
    for name, array in tensors.items():
        values = np.ascontiguousarray(array, dtype=np.float32)
        start = len(body)
        body.extend(values.tobytes())
        header[name] = {
            "dtype": "F32",
            "shape": list(values.shape),
            "data_offsets": [start, len(body)],
        }
    encoded = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + bytes(body)


def _adapter(
    directory: Path,
    *,
    factor_a: np.ndarray,
    factor_b: np.ndarray,
    trainer_state: dict[str, Any] | None = None,
    vocabulary: int = 1000,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_model.safetensors").write_bytes(
        safetensors_bytes(
            {
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": factor_a,
                "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": factor_b,
            }
        )
    )
    (directory / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {str(index): index for index in range(vocabulary)}}})
    )
    if trainer_state is not None:
        (directory / "trainer_state.json").write_text(json.dumps(trainer_state))
    return directory


def _state(*, first_loss: float, accuracy: float) -> dict[str, Any]:
    return {
        "epoch": 32.77,
        "global_step": 3900,
        "log_history": [
            {"loss": first_loss, "step": 25},
            {"loss": 0.5, "mean_token_accuracy": accuracy, "step": 3900},
            {"eval_loss": 0.4, "eval_mean_token_accuracy": accuracy, "step": 3900},
        ],
    }


def test_update_norm_applies_the_alpha_over_r_scaling(tmp_path: Path) -> None:
    directory = _adapter(
        tmp_path / "scaled",
        factor_a=np.eye(2),
        factor_b=np.eye(2),
    )
    weights = directory / "adapter_model.safetensors"

    unscaled = adapter_health.update_norms(weights, lora_alpha=8, rank=8)
    doubled = adapter_health.update_norms(weights, lora_alpha=16, rank=8)

    assert unscaled["modules"] == 1
    assert unscaled["update_norm_mean"] == np.float32(math.sqrt(2))
    assert doubled["update_norm_mean"] == np.float32(2 * math.sqrt(2))


def test_update_norm_matches_the_product_it_avoids_forming(tmp_path: Path) -> None:
    generator = np.random.default_rng(20240110)
    factor_a = generator.normal(size=(4, 16))
    factor_b = generator.normal(size=(12, 4))
    directory = _adapter(tmp_path / "random", factor_a=factor_a, factor_b=factor_b)

    measured = adapter_health.update_norms(
        directory / "adapter_model.safetensors", lora_alpha=16, rank=8
    )

    expected = 2.0 * float(
        np.linalg.norm(factor_b.astype(np.float32) @ factor_a.astype(np.float32))
    )
    assert measured["update_norm_mean"] == np.float32(expected)


def test_a_converged_adapter_is_healthy(tmp_path: Path) -> None:
    directory = _adapter(
        tmp_path / "healthy",
        factor_a=np.full((2, 4), 0.01),
        factor_b=np.full((4, 2), 0.01),
        trainer_state=_state(first_loss=2.0, accuracy=0.97),
    )

    record = adapter_health.measure(directory, {"r": 8, "lora_alpha": 16})

    assert record["verdict"] == adapter_health.VERDICT_HEALTHY
    assert record["eval_token_accuracy"] == 0.97
    assert record["global_step"] == 3900


def test_a_diverged_adapter_from_a_working_base_is_not_blamed_on_the_base(
    tmp_path: Path,
) -> None:
    directory = _adapter(
        tmp_path / "diverged",
        factor_a=np.full((2, 4), 900.0),
        factor_b=np.full((4, 2), 900.0),
        trainer_state=_state(first_loss=3.0, accuracy=0.12),
    )

    record = adapter_health.measure(directory, {"r": 8, "lora_alpha": 16})

    assert record["verdict"] == adapter_health.VERDICT_DIVERGED
    assert record["first_loss_over_guess"] < 1.0


def test_a_first_loss_above_the_random_guess_ceiling_names_the_base(tmp_path: Path) -> None:
    directory = _adapter(
        tmp_path / "broken-base",
        factor_a=np.full((2, 4), 900.0),
        factor_b=np.full((4, 2), 900.0),
        trainer_state=_state(first_loss=29.5, accuracy=0.11),
    )

    record = adapter_health.measure(directory, {"r": 8, "lora_alpha": 16})

    assert record["verdict"] == adapter_health.VERDICT_BROKEN_BASE
    assert record["random_guess_loss"] == math.log(1000)
    assert record["first_loss_over_guess"] > 1.0


def test_a_missing_trainer_state_is_unknown_never_a_pass(tmp_path: Path) -> None:
    directory = _adapter(
        tmp_path / "unmeasured",
        factor_a=np.full((2, 4), 0.01),
        factor_b=np.full((4, 2), 0.01),
    )

    record = adapter_health.measure(directory, {"r": 8, "lora_alpha": 16})

    assert record["verdict"] == adapter_health.VERDICT_UNKNOWN
    assert record["eval_token_accuracy"] is None
    assert record["modules"] == 1


def test_the_last_checkpoints_trainer_state_is_read_when_none_is_staged(tmp_path: Path) -> None:
    directory = _adapter(
        tmp_path / "unstaged",
        factor_a=np.full((2, 4), 0.01),
        factor_b=np.full((4, 2), 0.01),
    )
    for step, accuracy in ((100, 0.10), (3900, 0.97), (900, 0.40)):
        checkpoint = directory / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(_state(first_loss=2.0, accuracy=accuracy))
        )

    found = adapter_health.trainer_state_path(directory)

    assert found is not None
    assert found.parent.name == "checkpoint-3900"
    assert (
        adapter_health.measure(directory, {"r": 8, "lora_alpha": 16})["eval_token_accuracy"] == 0.97
    )


def test_an_unreadable_adapter_reports_no_modules_rather_than_raising(tmp_path: Path) -> None:
    directory = tmp_path / "truncated"
    directory.mkdir()
    (directory / "adapter_model.safetensors").write_bytes(b"not a safetensors file")

    measured = adapter_health.update_norms(
        directory / "adapter_model.safetensors", lora_alpha=16, rank=8
    )

    assert measured["modules"] == 0
    assert measured["update_norm_mean"] is None
    assert adapter_health.measure(directory, {"r": 8, "lora_alpha": 16})["verdict"] == (
        adapter_health.VERDICT_UNKNOWN
    )


def _distributional_state(*, divergences: list[float]) -> dict[str, Any]:
    return {
        "epoch": 3.5,
        "global_step": 3000,
        "log_history": [
            {"loss": 0.085, "step": 25},
            *(
                {"eval_loss": value / 5, "eval_kl": value, "step": 200 * (index + 1)}
                for index, value in enumerate(divergences)
            ),
        ],
    }


def test_a_distribution_matching_adapter_is_read_on_its_held_out_divergence(
    tmp_path: Path,
) -> None:
    directory = _adapter(
        tmp_path / "distributional",
        factor_a=np.full((2, 4), 0.01),
        factor_b=np.full((4, 2), 0.01),
        trainer_state=_distributional_state(divergences=[0.513, 0.505, 0.349, 0.388]),
    )

    record = adapter_health.measure(directory, {"r": 8, "lora_alpha": 32})

    assert record["verdict"] == adapter_health.VERDICT_HEALTHY
    assert record["eval_token_accuracy"] is None
    assert record["first_eval_kl"] == 0.513
    assert record["best_eval_kl"] == 0.349


def test_a_distribution_matching_adapter_that_never_improved_has_diverged(
    tmp_path: Path,
) -> None:
    directory = _adapter(
        tmp_path / "flat",
        factor_a=np.full((2, 4), 900.0),
        factor_b=np.full((4, 2), 900.0),
        trainer_state=_distributional_state(divergences=[0.51, 0.62, 0.77]),
    )

    record = adapter_health.measure(directory, {"r": 8, "lora_alpha": 32})

    assert record["verdict"] == adapter_health.VERDICT_DIVERGED


def test_an_adapter_with_no_trainer_state_stays_unknown(tmp_path: Path) -> None:
    directory = _adapter(
        tmp_path / "stateless",
        factor_a=np.full((2, 4), 0.01),
        factor_b=np.full((4, 2), 0.01),
    )

    record = adapter_health.measure(directory, {"r": 8, "lora_alpha": 32})

    assert record["verdict"] == adapter_health.VERDICT_UNKNOWN
    assert record["best_eval_kl"] is None
