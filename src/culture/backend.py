from __future__ import annotations

import math
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from machine_bias_reproduction.inference import strip_trailing_newline
from machine_bias_reproduction.questions import Question, resolve_question

from .adapters import read_adapter_config
from .registry import BASE_ARM, CHECKPOINT_CONDITION_DIRECTORY, CultureModel

FA_MAX_NEW_TOKENS = 12


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _snapshot_revision(repo_id: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id, local_files_only=True)).name
    except Exception:
        return None


def _answer_mass(values: Sequence[float]) -> float:
    """How much probability the answer letters got, with no answer read as zero.

    A row carries no answer two ways: the model spends its mass elsewhere, or
    the forward pass came back non-finite. A NaN that reaches the caller as a
    number would be averaged into a preflight and normalised into a
    distribution. Both cases mean the same thing, so both read as zero and the
    caller's zero-mass path flags them.
    """
    mass = math.fsum(values)
    return mass if math.isfinite(mass) and mass > 0 else 0.0


def _seeded_sampler_class() -> Any:
    import torch
    from transformers import LogitsProcessor

    class SeededBatchSampler(LogitsProcessor):  # type: ignore[misc]
        def __init__(self, generators: Sequence[Any], temperature: float) -> None:
            self._generators = list(generators)
            self._temperature = temperature

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            probabilities = torch.softmax(scores.float() / self._temperature, dim=-1)
            probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
            degenerate = probabilities.sum(dim=-1) <= 0
            if bool(degenerate.any()):
                probabilities = probabilities.clone()
                probabilities[degenerate] = 0.0
                probabilities[degenerate, scores[degenerate].argmax(dim=-1)] = 1.0
            drawn = torch.cat(
                [
                    torch.multinomial(probabilities[row], 1, generator=generator)
                    for row, generator in enumerate(self._generators)
                ]
            )
            forced = torch.full_like(scores, float("-inf"))
            forced.scatter_(1, drawn.unsqueeze(1), 0.0)
            return forced

    return SeededBatchSampler


class TransformersBackend:
    def __init__(
        self,
        model: CultureModel,
        adapters: dict[str, Path],
        question: str | Question,
        *,
        batch_size: int | None = None,
        fa_max_new_tokens: int = FA_MAX_NEW_TOKENS,
    ) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForImageTextToText, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "culture inference requires: uv sync --locked --extra culture"
            ) from error

        self._torch = torch
        self._model_spec = model
        self._question = resolve_question(question)
        self._batch_size = batch_size or model.batch_size
        self._fa_max_new_tokens = fa_max_new_tokens
        self._answer_tokens = self._question.ntp_tokens
        self._answer_columns = self._question.answer_columns
        self._oom_backoffs = 0
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sampler_class = _seeded_sampler_class()

        names = list(adapters)
        self._arm = CHECKPOINT_CONDITION_DIRECTORY if names else BASE_ARM
        first = names[0] if names else BASE_ARM
        self._tokenizer_source = str(adapters[first]) if names else model.base_model_id
        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_source)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = self._load_base(model, AutoModelForImageTextToText, torch)
        self._model_class = type(base).__name__
        self._attn_implementation = getattr(base.config, "_attn_implementation", None)
        if names:
            peft_model = PeftModel.from_pretrained(base, adapters[first], adapter_name=first)
            for name in names[1:]:
                peft_model.load_adapter(adapters[name], adapter_name=name)
            self._model = peft_model
            self._adapter_config = read_adapter_config(adapters[first])
        else:
            self._model = base
            self._adapter_config = {}
        self._model.eval()
        self._adapters = dict(adapters)
        self._active: str | None = None
        self.set_culture(first)

        self._answer_token_ids, self._multi_token_answers = self._resolve_answer_tokens()
        self._answer_index = torch.tensor(self._answer_token_ids, device=self._model.device)
        self._base_revision = _snapshot_revision(model.base_model_id)

    def _load_base(self, model: CultureModel, auto_class: Any, torch: Any) -> Any:
        if model.quantization == "nf4":
            from transformers import BitsAndBytesConfig

            return auto_class.from_pretrained(
                model.base_model_id,
                quantization_config=BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
                device_map=self._device,
            )
        return auto_class.from_pretrained(
            model.base_model_id,
            dtype=model.dtype,
            device_map=self._device,
        )

    def _with_oom_backoff(
        self,
        prompts: Sequence[str],
        run: Any,
        **kwargs: Any,
    ) -> list[Any]:
        torch = self._torch
        try:
            return list(run(prompts, **kwargs))
        except torch.OutOfMemoryError:
            if len(prompts) == 1:
                raise
            torch.cuda.empty_cache()
            self._oom_backoffs += 1
            middle = len(prompts) // 2
            left = dict(kwargs)
            right = dict(kwargs)
            if "seeds" in kwargs:
                left["seeds"] = kwargs["seeds"][:middle]
                right["seeds"] = kwargs["seeds"][middle:]
            return self._with_oom_backoff(prompts[:middle], run, **left) + self._with_oom_backoff(
                prompts[middle:], run, **right
            )

    def _resolve_answer_tokens(self) -> tuple[list[int], list[str]]:
        resolved: list[int] = []
        multi_token: list[str] = []
        for token in self._answer_tokens:
            encoded = self._tokenizer.encode(token, add_special_tokens=False)
            if not encoded:
                raise RuntimeError(f"tokenizer produced no tokens for answer {token!r}")
            if len(encoded) > 1:
                multi_token.append(token)
            resolved.append(int(encoded[0]))
        if len(set(resolved)) != len(resolved):
            collisions = {
                token: identifier
                for token, identifier in zip(self._answer_tokens, resolved, strict=True)
            }
            raise RuntimeError(
                f"tokenizer for {self._model_spec.base_model_id} maps two answers of "
                f"{self._question.var} to the same first token: {collisions}"
            )
        return resolved, multi_token

    def _prepare(self, prompt: str) -> str:
        return strip_trailing_newline(prompt)

    def set_culture(self, culture: str) -> None:
        if not self._adapters:
            if culture != BASE_ARM:
                raise KeyError(f"backend loaded the base model; cannot serve {culture!r}")
            self._active = culture
            return
        if culture not in self._adapters:
            raise KeyError(f"finetuned weights not attached: {culture}")
        if culture != self._active:
            self._model.set_adapter(culture)
            self._active = culture

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def token_length(self, prompt: str) -> int:
        return len(self._tokenizer.encode(self._prepare(prompt), add_special_tokens=True))

    def _encode(self, prompts: Sequence[str]) -> Any:
        return self._tokenizer(
            [self._prepare(prompt) for prompt in prompts],
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(self._model.device)

    def describe(self) -> dict[str, Any]:
        return {
            "name": "transformers",
            "arm": self._arm,
            "model_class": self._model_class,
            "version": _package_version("transformers"),
            "torch": _package_version("torch"),
            "peft": _package_version("peft"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "accelerate": _package_version("accelerate"),
            "device": self._device,
            "dtype": self._model_spec.dtype,
            "quantization": self._model_spec.quantization,
            "attn_implementation": self._attn_implementation,
            "batch_size": self._batch_size,
            "fa_max_new_tokens": self._fa_max_new_tokens,
            "base_model_id": self._model_spec.base_model_id,
            "base_revision": self._base_revision,
            "tokenizer_source": self._tokenizer_source,
            "question": self._question.var,
            "prompt_contract": "paper",
            "answer_tokens": dict(zip(self._answer_columns, self._answer_tokens, strict=True)),
            "answer_token_ids": dict(
                zip(self._answer_columns, self._answer_token_ids, strict=True)
            ),
            "multi_token_answers": self._multi_token_answers,
            "oom_backoffs": self._oom_backoffs,
            "lora_target_modules": (
                sorted(self._adapter_config.get("target_modules") or [])
                if self._adapter_config
                else None
            ),
            "lora_exclude_modules": self._adapter_config.get("exclude_modules"),
            "modalities_used": ["text"],
        }

    def ntp_batch(self, prompts: Sequence[str]) -> list[dict[str, float]]:
        if not prompts:
            return []
        return self._with_oom_backoff(prompts, self._ntp_batch)

    def _ntp_batch(self, prompts: Sequence[str]) -> list[dict[str, float]]:
        if not prompts:
            return []
        torch = self._torch
        encoded = self._encode(prompts)
        with torch.inference_mode():
            output = self._model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                logits_to_keep=1,
            )
        probabilities = torch.softmax(output.logits[:, -1, :].float(), dim=-1)
        selected = probabilities.index_select(1, self._answer_index).cpu()
        results: list[dict[str, float]] = []
        for row in selected:
            values = [float(value) for value in row]
            mass = _answer_mass(values)
            if mass <= 0:
                results.append({"mass": 0.0, "degenerate": True})
                continue
            results.append(
                {
                    "mass": mass,
                    **dict(
                        zip(
                            self._answer_columns,
                            [value / mass for value in values],
                            strict=True,
                        )
                    ),
                }
            )
        return results

    def ntp_mass_probe(self, prompts: Sequence[str]) -> list[float]:
        if not prompts:
            return []
        masses: list[float] = []
        for start in range(0, len(prompts), self._batch_size):
            chunk = prompts[start : start + self._batch_size]
            masses.extend(self._with_oom_backoff(chunk, self._ntp_mass_probe))
        return masses

    def _ntp_mass_probe(self, prompts: Sequence[str]) -> list[float]:
        torch = self._torch
        encoded = self._encode(prompts)
        with torch.inference_mode():
            output = self._model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                logits_to_keep=1,
            )
        probabilities = torch.softmax(output.logits[:, -1, :].float(), dim=-1)
        selected = probabilities.index_select(1, self._answer_index).cpu()
        return [_answer_mass([float(value) for value in row]) for row in selected]

    def ntp(self, prompt: str) -> dict[str, float]:
        return self.ntp_batch([prompt])[0]

    def full_answer_batch(
        self,
        prompts: Sequence[str],
        *,
        seeds: Sequence[int | None],
    ) -> list[str]:
        if not prompts:
            return []
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must have the same length")
        return self._with_oom_backoff(prompts, self._full_answer_batch, seeds=seeds)

    def _full_answer_batch(
        self,
        prompts: Sequence[str],
        *,
        seeds: Sequence[int | None],
    ) -> list[str]:
        torch = self._torch
        generators = []
        for seed in seeds:
            generator = torch.Generator(device=self._model.device)
            if seed is None:
                generator.seed()
            else:
                generator.manual_seed(int(seed))
            generators.append(generator)

        encoded = self._encode(prompts)
        prompt_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self._model.generate(  # type: ignore[no-untyped-call]
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=self._fa_max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                logits_processor=[self._sampler_class(generators, 0.7)],
                pad_token_id=self._tokenizer.pad_token_id,
            )
        completions = generated[:, prompt_length:]
        return [
            str(self._tokenizer.decode(row, skip_special_tokens=True)) for row in completions.cpu()
        ]

    def full_answer(self, prompt: str, *, seed: int | None) -> str:
        return self.full_answer_batch([prompt], seeds=[seed])[0]
