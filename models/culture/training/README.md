# Training provenance

The LoRA adapters under `models/culture/<model>/<culture>/` were finetuned in a
separate training repository, soon to be published under the
[CriticalMaking](https://github.com/CriticalMaking) organization.

| | |
|---|---|
| Configs | copied verbatim into `configs/` beside this file |
| Tracking | MLflow, experiment `culture_mllm_training`; run ids below |
| Stack | PEFT 0.19.1, TRL 1.4.0, Transformers 5.8.1 (5.15.0 for muse_glimmer_30b, qwen3_vl_2b and llama3_2_3b), PyTorch 2.12.0 |

`qwen3_vl_8b.yaml` targets the unquantized `Qwen/Qwen3-VL-8B-Thinking` release
and was copied when its adapter was staged, 2026-08-17.

`muse_glimmer_30b.yaml` targets `meta-models/Muse-Glimmer-30B`, trained QLoRA
4-bit and served NF4 for the same reason; the architecture needs
Transformers 5.15.0, which inference selects through the `muse` extra. Copied
when its adapter was staged, 2026-08-19.

`qwen3_vl_2b.yaml` and `llama3_2_3b.yaml` were copied when their adapters were
staged, 2026-08-21. They are the first two trained against a newer Transformers
than they are served under: both were fitted on 5.15.0, and both load through
the `culture` extra's 5.8.1, which already carries the `qwen3_vl` architecture
and resolves the `TokenizersBackend` tokenizer class the newer release wrote
into their checkpoints. `llama3_2_3b` is the first text-only base in the set —
`meta-llama/Llama-3.2-3B` is a `LlamaForCausalLM` with no vision tower, so the
registry marks it `modality="text"` and inference loads it through
`AutoModelForCausalLM` rather than `AutoModelForImageTextToText`.

Staged german adapters, one MLflow run each:

| model | MLflow run |
|---|---|
| gemma4_31b | `489588dcbff64bc39a5c138a11e18bd4` |
| gemma4_e4b | `2129abea05274bc093ba5593f5abd5fd` |
| qwen3_vl_8b | `c6578b98ff8843a19d0bcc0d08c7a326` |
| qwen3_vl_2b | `3785af11b389426eaee5a26844ec6e61` |
| llama3_2_3b | `c6760babdbcc4ea796f850cf5e356342` |
| muse_glimmer_30b | `57aa7bb649244aed99dd9d438b8c8066` |

`models/culture/ADAPTERS.json` records each staged adapter's SHA-256, LoRA
configuration and health record, written by
`./run_experiment_add_culture.sh adapters`; `health` re-checks on demand, and
an adapter's numbers are only worth reading once its verdict is `healthy`.
