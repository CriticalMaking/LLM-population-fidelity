from __future__ import annotations

import json
from pathlib import Path

import pytest

import culture
from culture import adapters as culture_adapters
from culture.palette import arm_fill, arm_hatch, arm_marker, arm_tone
from culture.population import sources
from culture.registry import CULTURE_MODELS

MODEL = "qwen3_vl_2b"


def _checkpoint(root: Path, arm: str, *, payload: bytes) -> Path:
    directory = root / arm / MODEL / culture.checkpoint_condition(arm)
    directory.mkdir(parents=True)
    (directory / "adapter_model.safetensors").write_bytes(payload)
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": CULTURE_MODELS[MODEL].base_model_id,
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "exclude_modules": ".*(vision_tower|audio_tower).*",
            }
        )
    )
    (directory / "TRAINING_DONE").write_text("")
    (directory / "tokenizer_config.json").write_text("{}")
    return directory


def test_global_is_a_finetuned_arm_that_is_not_a_culture() -> None:
    assert culture.GLOBAL_ARM == "global"
    assert culture.GLOBAL_ARM not in culture.CULTURES
    assert culture.GLOBAL_ARM in culture.ARMS
    assert culture.GLOBAL_ARM in culture.FINETUNED_ARMS
    assert culture.is_global("global")
    assert not culture.is_global("german")
    assert not culture.is_base("global")


def test_each_arm_names_the_checkpoint_directory_its_objective_wrote() -> None:
    assert culture.checkpoint_condition("global") == "distributional"
    for name in culture.CULTURES:
        assert culture.checkpoint_condition(name) == "cultural"


def test_resolvers_take_global_by_name_and_under_all() -> None:
    assert culture.resolve_cultures(["global"]) == ["global"]
    assert culture.resolve_finetuned_cultures(["global"]) == ["global"]
    assert "global" in culture.resolve_cultures(["all"])
    assert "global" in culture.resolve_finetuned_cultures(["all"])
    with pytest.raises(ValueError, match="unknown cultures"):
        culture.resolve_cultures(["distributional"])


def test_the_global_adapter_stages_from_its_own_condition_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, "global", payload=b"weights-global")
    staged_root = tmp_path / "models" / "culture"
    monkeypatch.setattr(culture_adapters, "ADAPTERS_ROOT", staged_root)
    monkeypatch.setattr(culture_adapters, "ADAPTERS_MANIFEST", staged_root / "ADAPTERS.json")

    manifest = culture.copy_adapters(root, culture.resolve_models([MODEL]), ["global"])

    assert (staged_root / MODEL / "global" / "adapter_model.safetensors").read_bytes() == (
        b"weights-global"
    )
    assert manifest["conditions"] == ["distributional"]
    record = manifest["adapters"][0]
    assert record["culture"] == "global"
    assert record["condition"] == "distributional"
    assert record["source"].endswith("global/qwen3_vl_2b/distributional")


def test_a_missing_global_checkpoint_names_the_distributional_directory(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, "german", payload=b"weights-german")
    with pytest.raises(FileNotFoundError, match="global/qwen3_vl_2b/distributional"):
        culture_adapters.adapter_source(root, MODEL, "global")


def test_the_global_arm_reads_as_a_variant_rather_than_a_culture() -> None:
    assert culture.arm_display("global") == "distribution-matched"
    assert CULTURE_MODELS[MODEL].run_label("global") == (
        "Qwen3-VL-2B-Thinking + distribution-matched LoRA"
    )
    assert CULTURE_MODELS[MODEL].run_label("german") == "Qwen3-VL-2B-Thinking + german LoRA"


def test_the_global_arm_draws_apart_from_the_cultures_that_are_run() -> None:
    run_arms = ("german", "spanish", "spanish-mx")
    assert arm_tone("global").ink not in {arm_tone(name).ink for name in run_arms}
    assert arm_marker("global") not in {arm_marker(name) for name in run_arms}
    assert arm_fill("global") not in {arm_fill(name) for name in run_arms}
    assert arm_hatch("global") not in {arm_hatch(name) for name in run_arms}


def test_the_fidelity_pipeline_looks_for_the_global_arm_on_every_model() -> None:
    found = {(run.key, run.arm) for run in sources()}
    assert (MODEL, "global") in found
    assert ("gemma4_31b", "global") in found


def test_the_global_run_writes_beside_the_culture_runs() -> None:
    assert culture.run_slug(MODEL, "global") == "culture/qwen3_vl_2b/global"
    assert culture.csv_stems(MODEL, "global", "d_happy") == (
        "NTP-qwen3_vl_2b-global-d_happy.csv",
        "FA-qwen3_vl_2b-global-d_happy.csv",
    )


def test_staging_one_arm_keeps_the_rest_of_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, "german", payload=b"weights-german")
    _checkpoint(root, "global", payload=b"weights-global")
    staged_root = tmp_path / "models" / "culture"
    monkeypatch.setattr(culture_adapters, "ADAPTERS_ROOT", staged_root)
    monkeypatch.setattr(culture_adapters, "ADAPTERS_MANIFEST", staged_root / "ADAPTERS.json")
    models = culture.resolve_models([MODEL])

    culture.copy_adapters(root, models, ["german"])
    manifest = culture.copy_adapters(root, models, ["global"])

    assert [(record["model_key"], record["culture"]) for record in manifest["adapters"]] == [
        (MODEL, "german"),
        (MODEL, "global"),
    ]
    assert manifest["counts"] == {"copied": 1, "reused": 0, "adapters": 2}
    assert manifest["conditions"] == ["cultural", "distributional"]


def test_the_ledger_drops_an_arm_whose_weights_are_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, "german", payload=b"weights-german")
    _checkpoint(root, "global", payload=b"weights-global")
    staged_root = tmp_path / "models" / "culture"
    monkeypatch.setattr(culture_adapters, "ADAPTERS_ROOT", staged_root)
    monkeypatch.setattr(culture_adapters, "ADAPTERS_MANIFEST", staged_root / "ADAPTERS.json")
    models = culture.resolve_models([MODEL])

    culture.copy_adapters(root, models, ["german"])
    (staged_root / MODEL / "german" / "adapter_model.safetensors").unlink()
    manifest = culture.copy_adapters(root, models, ["global"])

    assert [record["culture"] for record in manifest["adapters"]] == ["global"]


def test_subpop_is_a_finetuned_arm_that_is_not_a_culture() -> None:
    assert culture.SUBPOP_ARM == "subpop"
    assert culture.SUBPOP_ARM not in culture.CULTURES
    assert culture.SUBPOP_ARM in culture.ARMS
    assert culture.SUBPOP_ARM in culture.FINETUNED_ARMS
    assert not culture.is_base("subpop")
    assert not culture.is_global("subpop")
    assert culture.checkpoint_condition("subpop") == "subpop"
    assert culture.resolve_cultures(["subpop"]) == ["subpop"]
    assert culture.resolve_finetuned_cultures(["subpop"]) == ["subpop"]


def test_the_subpop_arm_reads_as_a_variant_rather_than_a_culture() -> None:
    assert culture.arm_display("subpop") == "subgroup-matched"
    assert CULTURE_MODELS[MODEL].run_label("subpop") == (
        "Qwen3-VL-2B-Thinking + subgroup-matched LoRA"
    )


def test_the_subpop_adapter_stages_from_its_own_condition_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, "global", payload=b"weights-global")
    _checkpoint(root, "subpop", payload=b"weights-subpop")
    staged_root = tmp_path / "models" / "culture"
    monkeypatch.setattr(culture_adapters, "ADAPTERS_ROOT", staged_root)
    monkeypatch.setattr(culture_adapters, "ADAPTERS_MANIFEST", staged_root / "ADAPTERS.json")
    models = culture.resolve_models([MODEL])

    culture.copy_adapters(root, models, ["global"])
    manifest = culture.copy_adapters(root, models, ["subpop"])

    assert (staged_root / MODEL / "subpop" / "adapter_model.safetensors").read_bytes() == (
        b"weights-subpop"
    )
    assert manifest["conditions"] == ["distributional", "subpop"]
    record = manifest["adapters"][-1]
    assert record["culture"] == "subpop"
    assert record["condition"] == "subpop"
    assert record["source"].endswith("subpop/qwen3_vl_2b/subpop")


def test_the_subpop_arm_draws_apart_from_global_and_the_cultures_that_are_run() -> None:
    others = ("german", "spanish", "spanish-mx", "global")
    assert arm_tone("subpop").ink not in {arm_tone(name).ink for name in others}
    assert arm_marker("subpop") not in {arm_marker(name) for name in others}
    assert arm_fill("subpop") not in {arm_fill(name) for name in others}
    assert arm_hatch("subpop") not in {arm_hatch(name) for name in others}


def test_the_subpop_run_writes_beside_the_culture_runs() -> None:
    assert culture.run_slug(MODEL, "subpop", 3) == "culture/qwen3_vl_2b/subpop/rep3"
    assert culture.csv_stems(MODEL, "subpop", "d_trust") == (
        "NTP-qwen3_vl_2b-subpop-d_trust.csv",
        "FA-qwen3_vl_2b-subpop-d_trust.csv",
    )
