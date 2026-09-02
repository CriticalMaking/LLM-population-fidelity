from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from culture import api_backend
from culture.api_backend import OpenAIBackend
from culture.registry import CULTURE_MODELS
from machine_bias_reproduction.questions import QUESTIONS


class StubCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        answer = self.responses.pop(0) if self.responses else None
        if isinstance(answer, BaseException):
            raise answer
        return answer


class StubClient:
    def __init__(self, responses: list[Any]) -> None:
        self.completions = StubCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class Retryable(Exception):
    status_code = 429


class Fatal(Exception):
    status_code = 400


class APIStatusError(Exception):
    status_code = 400


class BadRequestError(APIStatusError):
    pass


class Unsupported(BadRequestError):
    def __init__(self, param: str, code: str | None = "unsupported_parameter") -> None:
        super().__init__(param)
        self.code = code
        self.param = param
        self.message = f"Unsupported parameter: '{param}' is not supported with this model."


class UnsupportedValue(BadRequestError):
    def __init__(self, param: str, value: float) -> None:
        super().__init__(param)
        self.code = "unsupported_value"
        self.param = param
        self.message = (
            f"Unsupported value: '{param}' does not support {value} with this model. "
            "Only the default (1) value is supported."
        )
        self.type = "invalid_request_error"


def logprob_response(pairs: dict[str, float]) -> Any:
    entries = [
        SimpleNamespace(token=token, logprob=math.log(probability))
        for token, probability in pairs.items()
    ]
    content = [SimpleNamespace(top_logprobs=entries)]
    return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=content))])


def text_response(text: str, reasoning_tokens: int | None = None) -> Any:
    usage = (
        None
        if reasoning_tokens is None
        else SimpleNamespace(
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens)
        )
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=usage,
    )


def backend(
    responses: list[Any],
    question: str = "d_happy",
    model_key: str = "luna",
    model_id: str | None = "luna-test",
) -> OpenAIBackend:
    return OpenAIBackend(
        CULTURE_MODELS[model_key],
        QUESTIONS[question],
        batch_size=1,
        client=StubClient(responses),
        model_id=model_id,
    )


def test_next_token_probabilities_renormalize_over_the_answer_letters() -> None:
    served = backend([logprob_response({" A": 0.2, " B": 0.2, "the": 0.5})])
    result = served.ntp("Answer:")

    assert result["mass"] == pytest.approx(0.4)
    assert result["A"] == pytest.approx(0.5)
    assert result["B"] == pytest.approx(0.5)
    assert result["C"] == 0.0
    assert result["D"] == 0.0


def test_an_answer_outside_the_top_logprobs_reads_as_zero() -> None:
    served = backend([logprob_response({" A": 0.6, "the": 0.4})])
    result = served.ntp("Answer:")

    assert result["mass"] == pytest.approx(0.6)
    assert result["A"] == pytest.approx(1.0)
    assert result["D"] == 0.0


def test_a_reply_with_no_answer_letter_is_degenerate() -> None:
    served = backend([logprob_response({"the": 0.9, "a": 0.1})])

    assert served.ntp("Answer:") == {"mass": 0.0, "degenerate": True}


def test_politics_matches_bare_digits() -> None:
    served = backend([logprob_response({"3": 0.4, "7": 0.4, " left": 0.2})], "d_polpos")
    result = served.ntp("Answer:")

    assert result["mass"] == pytest.approx(0.8)
    assert result["D"] == pytest.approx(0.5)
    assert result["H"] == pytest.approx(0.5)
    assert result["A"] == 0.0


def test_the_probe_reports_mass_without_renormalizing() -> None:
    served = backend([logprob_response({" A": 0.3, "the": 0.7}), logprob_response({"the": 1.0})])

    assert served.ntp_mass_probe(["one", "two"]) == pytest.approx([0.3, 0.0])
    assert served.ntp_mass_probe([]) == []


def test_full_answers_carry_the_seed_the_framework_chose() -> None:
    served = backend([text_response("A. Very happy"), text_response("B. Rather happy")])
    answers = served.full_answer_batch(["one", "two"], seeds=[11, 22])

    assert answers == ["A. Very happy", "B. Rather happy"]
    calls = served._client.completions.calls
    assert [call["seed"] for call in calls] == [11, 22]
    assert all(call["temperature"] == api_backend.FA_TEMPERATURE for call in calls)


def test_prompts_reach_the_api_as_one_user_message_without_the_trailing_newline() -> None:
    served = backend([text_response("A. Very happy")])
    served.full_answer("Question?\nAnswer:\n", seed=None)

    call = served._client.completions.calls[0]
    assert call["messages"] == [{"role": "user", "content": "Question?\nAnswer:"}]
    assert "seed" not in call


def test_mismatched_prompts_and_seeds_are_refused() -> None:
    served = backend([])
    with pytest.raises(ValueError):
        served.full_answer_batch(["one", "two"], seeds=[1])


def test_a_rate_limit_is_retried_and_a_bad_request_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr("culture.api_backend.time.sleep", slept.append)

    served = backend([Retryable(), Retryable(), text_response("A. Very happy")])
    assert served.full_answer("prompt", seed=1) == "A. Very happy"
    assert slept == [1.0, 2.0]
    assert served.describe()["api_retries"] == 2

    fatal = backend([Fatal()])
    with pytest.raises(Fatal):
        fatal.full_answer("prompt", seed=1)


def test_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("culture.api_backend.time.sleep", lambda _: None)
    served = backend([Retryable() for _ in range(api_backend.MAX_API_ATTEMPTS)])

    with pytest.raises(Retryable):
        served.full_answer("prompt", seed=1)
    assert len(served._client.completions.calls) == api_backend.MAX_API_ATTEMPTS


def test_the_backoff_is_capped() -> None:
    delays = [
        min(
            api_backend.BACKOFF_CAP_SECONDS,
            api_backend.BACKOFF_BASE_SECONDS * 2**attempt,
        )
        for attempt in range(12)
    ]
    assert max(delays) == api_backend.BACKOFF_CAP_SECONDS


def test_the_api_model_serves_only_the_base_variant() -> None:
    served = backend([])
    served.set_culture("base")
    with pytest.raises(KeyError):
        served.set_culture("german")


def test_next_token_probability_degrades_to_full_answers() -> None:
    assert OpenAIBackend.ntp_degrades_to_fa is True


def test_token_length_approximates_for_batch_ordering() -> None:
    served = backend([])
    assert served.token_length("a" * 40) == 10
    assert served.token_length("a") == 1


@pytest.mark.parametrize(
    ("model_key", "variable"),
    [("luna", "LUNA_MODEL_ID"), ("terra", "TERRA_MODEL_ID"), ("sol", "SOL_MODEL_ID")],
)
def test_every_served_model_reads_its_own_id_from_the_environment(
    model_key: str,
    variable: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert api_backend.model_id_env(model_key) == variable
    monkeypatch.setenv(variable, f"{model_key}-served-id")

    served = backend([], model_key=model_key, model_id=None)
    assert served.describe()["model_id"] == f"{model_key}-served-id"
    assert served.describe()["model_id_env"] == variable


def test_a_missing_model_id_names_the_variable_to_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TERRA_MODEL_ID", raising=False)

    with pytest.raises(RuntimeError, match="TERRA_MODEL_ID"):
        backend([], model_key="terra", model_id=None)


def test_every_request_carries_the_reasoning_effort_the_run_pinned() -> None:
    served = backend([text_response("A. Very happy"), logprob_response({" A": 0.5})])
    served.full_answer("prompt", seed=1)
    served.ntp("prompt")

    calls = served._client.completions.calls
    assert [call["reasoning_effort"] for call in calls] == ["medium", "medium"]
    assert served.describe()["reasoning_effort"] == api_backend.REASONING_EFFORT


def test_a_model_that_refuses_the_effort_is_called_without_it() -> None:
    served = backend([Unsupported("reasoning_effort"), text_response("A. Very happy")])

    assert served.full_answer("prompt", seed=1) == "A. Very happy"
    calls = served._client.completions.calls
    assert calls[0]["reasoning_effort"] == "medium"
    assert "reasoning_effort" not in calls[1]
    assert served.describe()["reasoning_effort"] is None
    assert served.describe()["reasoning_effort_requested"] == "medium"
    assert served.describe()["reasoning_effort_supported"] is False


def test_an_unknown_effort_is_refused_before_a_billable_call() -> None:
    with pytest.raises(ValueError, match="unknown reasoning effort"):
        OpenAIBackend(
            CULTURE_MODELS["luna"],
            QUESTIONS["d_happy"],
            reasoning_effort="ludicrous",
            client=StubClient([]),
            model_id="luna-test",
        )


def test_reasoning_tokens_are_read_back_per_reply() -> None:
    served = backend(
        [
            text_response("Please choose one:", reasoning_tokens=192),
            text_response("A. Very happy", reasoning_tokens=64),
        ]
    )
    served.full_answer_batch(["one", "two"], seeds=[1, 2])

    assert served.last_reasoning_tokens == [192, 64]
    assert served.describe()["mean_reasoning_tokens"] == 128.0


def test_a_response_without_usage_reads_as_no_reasoning_rather_than_failing() -> None:
    served = backend([text_response("A. Very happy")])
    served.full_answer("prompt", seed=1)

    assert served.last_reasoning_tokens == [0]


@pytest.mark.parametrize(
    "refusal",
    [
        Unsupported("temperature"),
        UnsupportedValue("temperature", 0.7),
        Unsupported("temperature", code=None),
    ],
    ids=["unsupported_parameter", "unsupported_value", "no_code"],
)
def test_a_model_that_refuses_temperature_is_called_without_it(refusal: Exception) -> None:
    served = backend([refusal, text_response("A. Very happy")])

    assert served.full_answer("prompt", seed=1) == "A. Very happy"
    calls = served._client.completions.calls
    assert calls[0]["temperature"] == api_backend.FA_TEMPERATURE
    assert "temperature" not in calls[1]
    assert served.describe()["fa_temperature"] is None
    assert served.describe()["ntp_temperature"] is None


def test_a_model_that_refuses_temperature_still_probes_next_token_probabilities() -> None:
    served = backend([UnsupportedValue("temperature", 0.0), logprob_response({" A": 0.3})])

    assert served.ntp_mass_probe(["prompt"]) == pytest.approx([0.3])
    calls = served._client.completions.calls
    assert calls[0]["temperature"] == api_backend.NTP_TEMPERATURE
    assert "temperature" not in calls[1]
    assert served.describe()["logprobs_supported"] is True


def test_a_bad_request_is_not_retried_for_carrying_a_retryable_class_name() -> None:
    served = backend([BadRequestError("no")])

    with pytest.raises(BadRequestError):
        served.full_answer("prompt", seed=1)
    assert len(served._client.completions.calls) == 1


def test_an_endpoint_without_logprobs_is_recorded_not_read_as_zero_mass() -> None:
    served = backend([Unsupported("logprobs")])

    assert served.ntp_mass_probe(["prompt"]) == [0.0]
    assert served.describe()["logprobs_supported"] is False


def test_describe_records_the_contract_the_api_imposes() -> None:
    described = backend([]).describe()

    assert described["name"] == "openai"
    assert described["model_id"] == "luna-test"
    assert described["prompt_contract"] == "paper-as-single-user-message"
    assert described["seed_semantics"] == "best_effort_api_seed"
    assert described["top_logprobs"] == api_backend.TOP_LOGPROBS
    assert described["ntp_temperature"] == 0.0
    assert described["base_revision"] is None
    assert described["answer_tokens"]["A"] == " A"
