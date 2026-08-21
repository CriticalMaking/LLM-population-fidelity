from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from machine_bias_reproduction.config import GLOBAL_SEED, PROJECT_ROOT
from machine_bias_reproduction.inference import strip_trailing_newline
from machine_bias_reproduction.questions import Question, resolve_question

from .registry import BASE_ARM, CultureModel

FA_MAX_NEW_TOKENS = 12

TOP_LOGPROBS = 20

MAX_API_ATTEMPTS = 6

BACKOFF_BASE_SECONDS = 1.0

BACKOFF_CAP_SECONDS = 30.0

APPROX_CHARS_PER_TOKEN = 4

MODEL_ID_ENV_SUFFIX = "_MODEL_ID"

API_KEY_ENV = "OPENAI_API_KEY"

FA_TEMPERATURE = 0.7

NTP_TEMPERATURE = 0.0

REASONING_EFFORTS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

REASONING_EFFORT = "medium"

RETRYABLE_ERRORS = frozenset(
    {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "APIStatusError",
    }
)

UNSUPPORTED_CODES = frozenset({"unsupported_parameter", "unsupported_value"})

_UNSUPPORTED_MESSAGE = re.compile(
    r"unsupported\s+(?:value|parameter)|only\s+the\s+default|does\s+not\s+support",
    re.I,
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _answer_mass(values: Sequence[float]) -> float:
    mass = math.fsum(values)
    return mass if math.isfinite(mass) and mass > 0 else 0.0


def _retryable(error: BaseException) -> bool:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    names = {type(error).__name__} | {base.__name__ for base in type(error).__mro__}
    return bool(names & RETRYABLE_ERRORS)


def _unsupported_parameter(error: BaseException, name: str) -> bool:
    if getattr(error, "param", None) != name:
        return False
    if getattr(error, "code", None) in UNSUPPORTED_CODES:
        return True
    return bool(_UNSUPPORTED_MESSAGE.search(str(getattr(error, "message", "") or error)))


def model_id_env(model_key: str) -> str:
    return f"{model_key.upper()}{MODEL_ID_ENV_SUFFIX}"


def _reasoning_tokens(response: Any) -> int:
    details = getattr(getattr(response, "usage", None), "completion_tokens_details", None)
    tokens = getattr(details, "reasoning_tokens", None)
    return int(tokens) if isinstance(tokens, int) else 0


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError("api inference requires: uv sync --locked --extra api") from error
    load_dotenv(PROJECT_ROOT / ".env")


def _build_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("api inference requires: uv sync --locked --extra api") from error
    if not os.environ.get(API_KEY_ENV):
        raise RuntimeError(f"{API_KEY_ENV} is not set; put it in {PROJECT_ROOT / '.env'}")
    return OpenAI()


def _resolve_model_id(model: CultureModel, model_id: str | None) -> str:
    if model_id:
        return model_id
    variable = model_id_env(model.key)
    served = os.environ.get(variable)
    if not served:
        raise RuntimeError(
            f"{variable} is not set; put the id {model.label} is served under "
            f"in {PROJECT_ROOT / '.env'}"
        )
    return served


class OpenAIBackend:
    ntp_degrades_to_fa = True

    def __init__(
        self,
        model: CultureModel,
        question: str | Question,
        *,
        batch_size: int | None = None,
        fa_max_new_tokens: int = FA_MAX_NEW_TOKENS,
        reasoning_effort: str | None = REASONING_EFFORT,
        client: Any | None = None,
        model_id: str | None = None,
    ) -> None:
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"unknown reasoning effort {reasoning_effort!r}; "
                f"expected one of {', '.join(REASONING_EFFORTS)}"
            )
        self._model_spec = model
        self._question = resolve_question(question)
        self._batch_size = batch_size or model.batch_size
        self._fa_max_new_tokens = fa_max_new_tokens
        self._reasoning_effort = reasoning_effort
        self._answer_tokens = self._question.ntp_tokens
        self._answer_columns = self._question.answer_columns
        self._answer_keys = [token.strip() for token in self._answer_tokens]
        self._active: str | None = None
        self._api_attempts = 0
        self._api_retries = 0
        self._temperature_supported = True
        self._logprobs_supported = True
        self._reasoning_supported = reasoning_effort is not None
        self._reasoning_tokens: list[int] = []
        self.last_reasoning_tokens: list[int] = []
        if client is None:
            _load_environment()
            self._client = _build_client()
        else:
            self._client = client
        self._model_id = _resolve_model_id(model, model_id)
        self.set_culture(BASE_ARM)

    def _prepare(self, prompt: str) -> str:
        return strip_trailing_newline(prompt)

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": self._prepare(prompt)}]

    def _request(self, **kwargs: Any) -> Any:
        last: BaseException | None = None
        if self._reasoning_supported and self._reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._reasoning_effort
        for attempt in range(MAX_API_ATTEMPTS):
            if not self._temperature_supported:
                kwargs.pop("temperature", None)
            if not self._reasoning_supported:
                kwargs.pop("reasoning_effort", None)
            self._api_attempts += 1
            try:
                return self._client.chat.completions.create(model=self._model_id, **kwargs)
            except BaseException as error:
                if "reasoning_effort" in kwargs and _unsupported_parameter(
                    error, "reasoning_effort"
                ):
                    self._reasoning_supported = False
                    continue
                if "temperature" in kwargs and _unsupported_parameter(error, "temperature"):
                    self._temperature_supported = False
                    continue
                if not _retryable(error) or attempt == MAX_API_ATTEMPTS - 1:
                    raise
                last = error
                self._api_retries += 1
                time.sleep(min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2**attempt))
        raise RuntimeError(f"the api gave up after {MAX_API_ATTEMPTS} attempts") from last

    def _map(self, run: Any, items: Sequence[Any]) -> list[Any]:
        if not items:
            return []
        if len(items) == 1 or self._batch_size <= 1:
            return [run(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(self._batch_size, len(items))) as pool:
            return list(pool.map(run, items))

    def _answer_probabilities(self, prompt: str) -> list[float]:
        try:
            response = self._request(
                messages=self._messages(prompt),
                max_completion_tokens=1,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
                temperature=NTP_TEMPERATURE,
                seed=GLOBAL_SEED,
            )
        except BaseException as error:
            if _unsupported_parameter(error, "logprobs"):
                self._logprobs_supported = False
                return [0.0 for _ in self._answer_keys]
            raise
        content = response.choices[0].logprobs.content
        if not content:
            return [0.0 for _ in self._answer_keys]
        found: dict[str, float] = {}
        for entry in content[0].top_logprobs:
            key = str(entry.token).strip()
            if key in self._answer_keys:
                found[key] = found.get(key, 0.0) + math.exp(float(entry.logprob))
        return [found.get(key, 0.0) for key in self._answer_keys]

    def _ntp_one(self, prompt: str) -> dict[str, float]:
        values = self._answer_probabilities(prompt)
        mass = _answer_mass(values)
        if mass <= 0:
            return {"mass": 0.0, "degenerate": True}
        return {
            "mass": mass,
            **dict(zip(self._answer_columns, [value / mass for value in values], strict=True)),
        }

    def set_culture(self, culture: str) -> None:
        if culture != BASE_ARM:
            raise KeyError(f"api model {self._model_spec.key} serves only the {BASE_ARM} variant")
        self._active = culture

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def token_length(self, prompt: str) -> int:
        return max(1, len(self._prepare(prompt)) // APPROX_CHARS_PER_TOKEN)

    @property
    def mean_reasoning_tokens(self) -> float | None:
        if not self._reasoning_tokens:
            return None
        return math.fsum(self._reasoning_tokens) / len(self._reasoning_tokens)

    def describe(self) -> dict[str, Any]:
        return {
            "name": "openai",
            "arm": BASE_ARM,
            "version": _package_version("openai"),
            "model_id": self._model_id,
            "model_id_env": model_id_env(self._model_spec.key),
            "base_model_id": self._model_spec.base_model_id,
            "base_revision": None,
            "dtype": self._model_spec.dtype,
            "quantization": self._model_spec.quantization,
            "batch_size": self._batch_size,
            "concurrency": self._batch_size,
            "fa_max_new_tokens": self._fa_max_new_tokens,
            "top_logprobs": TOP_LOGPROBS,
            "logprobs_supported": self._logprobs_supported,
            "ntp_temperature": NTP_TEMPERATURE if self._temperature_supported else None,
            "fa_temperature": FA_TEMPERATURE if self._temperature_supported else None,
            "reasoning_effort": self._reasoning_effort if self._reasoning_supported else None,
            "reasoning_effort_requested": self._reasoning_effort,
            "reasoning_effort_supported": self._reasoning_supported,
            "mean_reasoning_tokens": self.mean_reasoning_tokens,
            "question": self._question.var,
            "prompt_contract": "paper-as-single-user-message",
            "seed_semantics": "best_effort_api_seed",
            "answer_tokens": dict(zip(self._answer_columns, self._answer_tokens, strict=True)),
            "api_attempts": self._api_attempts,
            "api_retries": self._api_retries,
            "modalities_used": ["text"],
        }

    def ntp_batch(self, prompts: Sequence[str]) -> list[dict[str, float]]:
        return self._map(self._ntp_one, list(prompts))

    def ntp_mass_probe(self, prompts: Sequence[str]) -> list[float]:
        results = self._map(self._answer_probabilities, list(prompts))
        return [_answer_mass(values) for values in results]

    def ntp(self, prompt: str) -> dict[str, float]:
        return self._ntp_one(prompt)

    def _full_answer_one(self, item: tuple[str, int | None]) -> tuple[str, int]:
        prompt, seed = item
        arguments: dict[str, Any] = {
            "messages": self._messages(prompt),
            "max_completion_tokens": self._fa_max_new_tokens,
            "temperature": FA_TEMPERATURE,
        }
        if seed is not None:
            arguments["seed"] = int(seed)
        response = self._request(**arguments)
        thought = _reasoning_tokens(response)
        self._reasoning_tokens.append(thought)
        return str(response.choices[0].message.content or ""), thought

    def full_answer_batch(
        self,
        prompts: Sequence[str],
        *,
        seeds: Sequence[int | None],
    ) -> list[str]:
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must have the same length")
        results = self._map(self._full_answer_one, list(zip(prompts, seeds, strict=True)))
        self.last_reasoning_tokens = [thought for _, thought in results]
        return [text for text, _ in results]

    def full_answer(self, prompt: str, *, seed: int | None) -> str:
        text, thought = self._full_answer_one((prompt, seed))
        self.last_reasoning_tokens = [thought]
        return text
