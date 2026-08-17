# Training provenance

The LoRA adapters under `models/culture/<model>/<culture>/` were finetuned in a
separate training repository, soon to be published under the
[CriticalMaking](https://github.com/CriticalMaking) organization.

| | |
|---|---|
| Configs | copied verbatim into `configs/` beside this file |
| Tracking | MLflow, experiment `culture_mllm_training`; run ids below |
| Stack | PEFT 0.19.1, TRL 1.4.0, Transformers 5.8.1, PyTorch 2.12.0 |

`qwen3_vl_8b.yaml` targets the unquantized `Qwen/Qwen3-VL-8B-Thinking` release
and was copied when its adapter was staged, 2026-08-17.

Staged german adapters, one MLflow run each:

| model | MLflow run |
|---|---|
| gemma4_31b | `489588dcbff64bc39a5c138a11e18bd4` |
| gemma4_e4b | `2129abea05274bc093ba5593f5abd5fd` |
| qwen3_vl_8b | `c6578b98ff8843a19d0bcc0d08c7a326` |

`models/culture/ADAPTERS.json` records each staged adapter's SHA-256, LoRA
configuration and health record, written by
`./run_experiment_add_culture.sh adapters`; `health` re-checks on demand, and
an adapter's numbers are only worth reading once its verdict is `healthy`.
