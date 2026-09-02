from __future__ import annotations

import json
from pathlib import Path

import pytest

import culture
from culture import adapters as culture_adapters
from machine_bias_reproduction.config import FIGURES_ROOT, OUTPUTS_ROOT, paths_for


def test_registry_covers_ten_cultures_and_nine_models() -> None:
    assert len(culture.CULTURES) == 10
    assert set(culture.CULTURE_MODELS) == {
        "gemma4_31b",
        "gemma4_e4b",
        "qwen3_vl_8b",
        "qwen3_vl_2b",
        "llama3_2_3b",
        "muse_glimmer_30b",
        "luna",
        "terra",
        "sol",
    }
    for key, model in culture.CULTURE_MODELS.items():
        assert model.key == key
        assert model.batch_size > 0


def test_three_served_models_share_the_api_backend() -> None:
    served = {
        key for key, model in culture.CULTURE_MODELS.items() if model.backend != "transformers"
    }
    assert served == {"luna", "terra", "sol"}
    assert culture.served_keys() == ("luna", "terra", "sol")
    for key in served:
        assert culture.CULTURE_MODELS[key].backend == "openai"
        assert culture.CULTURE_MODELS[key].dtype == "api"
        assert culture.is_served(key)
    assert not culture.is_served("gemma4_31b")


def test_models_load_in_a_format_that_fits_a_single_32gb_card() -> None:
    assert culture.CULTURE_MODELS["gemma4_31b"].quantization == "nf4"
    assert culture.CULTURE_MODELS["gemma4_e4b"].quantization is None
    assert culture.CULTURE_MODELS["gemma4_e4b"].dtype == "bfloat16"
    assert culture.CULTURE_MODELS["qwen3_vl_8b"].quantization is None
    assert culture.CULTURE_MODELS["qwen3_vl_8b"].dtype == "bfloat16"
    assert culture.CULTURE_MODELS["qwen3_vl_2b"].quantization is None
    assert culture.CULTURE_MODELS["qwen3_vl_2b"].dtype == "bfloat16"
    assert culture.CULTURE_MODELS["llama3_2_3b"].quantization is None
    assert culture.CULTURE_MODELS["llama3_2_3b"].dtype == "bfloat16"
    assert culture.CULTURE_MODELS["muse_glimmer_30b"].quantization == "nf4"
    assert culture.CULTURE_MODELS["muse_glimmer_30b"].dtype == "bfloat16"


def test_only_the_llama_arm_loads_through_the_text_only_auto_class() -> None:
    text_only = {
        key
        for key, model in culture.CULTURE_MODELS.items()
        if model.modality == culture.TEXT_MODALITY
    }
    assert text_only == {"llama3_2_3b"}
    assert culture.CULTURE_MODELS["qwen3_vl_2b"].modality == culture.VISION_TEXT_MODALITY
    assert culture.CULTURE_MODELS["gemma4_31b"].modality == culture.VISION_TEXT_MODALITY


def test_resolve_defaults_and_rejects_unknown_names() -> None:
    assert len(culture.resolve_models(None)) == len(culture.CULTURE_MODELS)
    assert culture.resolve_cultures(None) == list(culture.ARMS)
    assert [model.key for model in culture.resolve_models(["qwen3_vl_8b"])] == ["qwen3_vl_8b"]
    with pytest.raises(ValueError, match="unknown model keys"):
        culture.resolve_models(["not-a-model"])
    with pytest.raises(ValueError, match="unknown cultures"):
        culture.resolve_cultures(["klingon"])


def test_run_paths_nest_under_the_shared_output_roots() -> None:
    paths = culture.run_paths("gemma4_31b", "portuguese", "d_trust")
    assert paths.outputs == OUTPUTS_ROOT / "culture" / "gemma4_31b" / "portuguese" / "d_trust"
    assert paths.figures == FIGURES_ROOT / "culture" / "gemma4_31b" / "portuguese" / "d_trust"
    assert paths.source == culture.run_slug("gemma4_31b", "portuguese")


def test_the_base_arm_nests_beside_the_finetuned_runs() -> None:
    paths = culture.run_paths("gemma4_31b", culture.BASE_ARM, "d_happy")
    assert paths.outputs == OUTPUTS_ROOT / "culture" / "gemma4_31b" / "base" / "d_happy"
    assert culture.csv_stems("gemma4_31b", culture.BASE_ARM, "d_happy") == (
        "NTP-gemma4_31b-base-d_happy.csv",
        "FA-gemma4_31b-base-d_happy.csv",
    )


def test_every_registered_arm_resolves_to_a_run_path() -> None:
    for model_key in culture.CULTURE_MODELS:
        for arm in culture.ARMS:
            paths = culture.run_paths(model_key, arm, "d_happy")
            assert paths.outputs == OUTPUTS_ROOT / "culture" / model_key / arm / "d_happy"


def test_a_hyphenated_culture_keeps_its_hyphen_in_the_paths_and_stems() -> None:
    paths = culture.run_paths("gemma4_31b", "spanish-mx", "d_happy")
    assert paths.outputs == OUTPUTS_ROOT / "culture" / "gemma4_31b" / "spanish-mx" / "d_happy"
    assert culture.csv_stems("gemma4_31b", "spanish-mx", "d_happy") == (
        "NTP-gemma4_31b-spanish-mx-d_happy.csv",
        "FA-gemma4_31b-spanish-mx-d_happy.csv",
    )


def test_paths_for_still_rejects_sources_that_escape_the_roots() -> None:
    assert paths_for("archived", "d_happy").source == "archived"
    assert paths_for("fresh", "d_happy").source == "fresh"
    for bad in ("culture/../../etc", "culture/only-one", "culture/a/b/c", "elsewhere"):
        with pytest.raises(ValueError, match="unknown source"):
            paths_for(bad, "d_happy")
    for bad_question in ("../etc", "d happy", "d/happy"):
        with pytest.raises(ValueError, match="unsafe question"):
            paths_for("archived", bad_question)


def test_csv_stems_name_the_model_culture_and_question() -> None:
    ntp, fa = culture.csv_stems("qwen3_vl_8b", "german", "d_religiousp")
    assert ntp == "NTP-qwen3_vl_8b-german-d_religiousp.csv"
    assert fa == "FA-qwen3_vl_8b-german-d_religiousp.csv"


def test_matched_cultures_are_only_those_with_wvs_respondents() -> None:
    import pandas as pd

    from culture.matching import MATCHED_CULTURES, countries_for, restrict_to

    assert MATCHED_CULTURES == ("english", "german", "spanish", "spanish-mx")
    frame = pd.DataFrame(
        {
            "subpopulation": [
                "Germany 2018 Female 25-34 High Working Married",
                "Mexico 2005 Male <25 Low Student Single",
            ]
        }
    )
    assert len(restrict_to(frame, countries_for("german"))) == 1
    assert restrict_to(frame, countries_for("korean")).empty


def _fake_checkpoint(root: Path, model_key: str, culture_name: str, *, payload: bytes) -> Path:
    directory = root / culture_name / model_key / "cultural"
    directory.mkdir(parents=True)
    (directory / "adapter_model.safetensors").write_bytes(payload)
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": culture.CULTURE_MODELS[model_key].base_model_id,
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "exclude_modules": ".*(vision_tower|audio_tower).*",
            }
        )
    )
    (directory / "TRAINING_DONE").write_text("")
    (directory / "tokenizer_config.json").write_text("{}")
    for step in (100, 200):
        step_directory = directory / f"checkpoint-{step}"
        step_directory.mkdir()
        (step_directory / "optimizer.pt").write_bytes(b"x" * 64)
    return directory


def test_copy_adapters_stages_only_the_final_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    _fake_checkpoint(root, "qwen3_vl_8b", "german", payload=b"weights-german")
    staged_root = tmp_path / "models" / "culture"
    monkeypatch.setattr(culture_adapters, "ADAPTERS_ROOT", staged_root)
    monkeypatch.setattr(culture_adapters, "ADAPTERS_MANIFEST", staged_root / "ADAPTERS.json")

    manifest = culture.copy_adapters(
        root,
        culture.resolve_models(["qwen3_vl_8b"]),
        ["german"],
    )

    destination = staged_root / "qwen3_vl_8b" / "german"
    assert (destination / "adapter_model.safetensors").read_bytes() == b"weights-german"
    assert not list(destination.glob("checkpoint-*"))
    assert manifest["counts"] == {"copied": 1, "reused": 0, "adapters": 1}
    record = manifest["adapters"][0]
    assert record["base_model_name_or_path"] == "Qwen/Qwen3-VL-8B-Thinking"
    assert record["target_modules"] == ["k_proj", "o_proj", "q_proj", "v_proj"]
    assert "vision_tower" in record["exclude_modules"]


def test_copy_adapters_reuses_a_matching_stage_and_repairs_a_broken_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    _fake_checkpoint(root, "qwen3_vl_8b", "german", payload=b"weights-german")
    staged_root = tmp_path / "models" / "culture"
    monkeypatch.setattr(culture_adapters, "ADAPTERS_ROOT", staged_root)
    monkeypatch.setattr(culture_adapters, "ADAPTERS_MANIFEST", staged_root / "ADAPTERS.json")
    models = culture.resolve_models(["qwen3_vl_8b"])

    culture.copy_adapters(root, models, ["german"])
    again = culture.copy_adapters(root, models, ["german"])
    assert again["counts"]["reused"] == 1

    truncated = staged_root / "qwen3_vl_8b" / "german" / "adapter_model.safetensors"
    truncated.write_bytes(b"truncated")
    repaired = culture.copy_adapters(root, models, ["german"])
    assert repaired["counts"]["copied"] == 1
    assert truncated.read_bytes() == b"weights-german"


def test_copy_adapters_rejects_an_unfinished_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    directory = _fake_checkpoint(root, "qwen3_vl_8b", "german", payload=b"w")
    (directory / "TRAINING_DONE").unlink()
    monkeypatch.setattr(culture_adapters, "ADAPTERS_ROOT", tmp_path / "staged")
    with pytest.raises(FileNotFoundError, match="TRAINING_DONE"):
        culture.copy_adapters(root, culture.resolve_models(["qwen3_vl_8b"]), ["german"])


def test_copy_adapters_rejects_a_base_model_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    directory = _fake_checkpoint(root, "qwen3_vl_8b", "german", payload=b"w")
    (directory / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "some/other-model"})
    )
    staged_root = tmp_path / "staged"
    monkeypatch.setattr(culture_adapters, "ADAPTERS_ROOT", staged_root)
    monkeypatch.setattr(culture_adapters, "ADAPTERS_MANIFEST", staged_root / "ADAPTERS.json")
    with pytest.raises(ValueError, match="registry expects"):
        culture.copy_adapters(root, culture.resolve_models(["qwen3_vl_8b"]), ["german"])


def test_staged_adapter_points_at_the_copy_not_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged_root = tmp_path / "models" / "culture"
    monkeypatch.setattr(culture_adapters, "ADAPTERS_ROOT", staged_root)
    with pytest.raises(FileNotFoundError, match="not staged"):
        culture.staged_adapter("qwen3_vl_8b", "german")


def test_culture_prompts_are_the_papers_prompts_byte_for_byte() -> None:
    from machine_bias_reproduction.config import UPSTREAM_DATA
    from machine_bias_reproduction.data import load_wvs
    from machine_bias_reproduction.prompts import prompt_records

    records = prompt_records(load_wvs().head(20), "fa", "d_happy")[:5]
    for record in records:
        upstream = UPSTREAM_DATA / "prompts" / "FA" / "d_happy" / f"{record.prompt_id}.txt"
        assert record.text.encode() == upstream.read_bytes()


def test_no_chat_contract_survives() -> None:
    import machine_bias_reproduction.inference as inference
    import machine_bias_reproduction.prompts as prompts

    assert not hasattr(prompts, "build_culture_messages")
    assert not hasattr(prompts, "CULTURE_QUESTION_TEMPLATE")
    assert not hasattr(prompts, "culture_prompt_records")
    assert not hasattr(inference, "parse_culture_answer")
