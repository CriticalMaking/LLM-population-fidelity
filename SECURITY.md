# Security Policy

This is a research reproduction, not deployed software. It reads a survey
dataset and runs local models; it serves no traffic, opens no ports and handles
no user data. The realistic security surface is its dependencies.

## Supported Versions

There are no releases. Only the current tip of `main` is supported, and fixes
land there.

| Version | Supported |
| --- | --- |
| `main` (tip) | ✅ |
| Older commits | ❌ — rebase onto `main` |

Dependency versions are pinned in `uv.lock`. Two extras (`culture`,
`inference`) pin a deliberately older stack; see the dispositions below.

## Reporting a Vulnerability

Report privately through GitHub: open the repository's **Security** tab and
choose **Report a vulnerability**. That opens a private advisory visible only
to the maintainers.

> Private reporting must be enabled for the repository (**Settings → Code
> security → Private vulnerability reporting**). If the button is absent, open
> a regular issue that says only that you have a security report and asks for a
> private channel — do not include details in a public issue.

Please include:

- the affected file, dependency or command;
- the version or commit you tested;
- what an attacker gains, and the access they need to start;
- steps to reproduce, ideally a minimal one.

What to expect: acknowledgement within **7 days**, an assessment within **30
days**, and credit in the advisory unless you prefer otherwise. This is a small
research project, not a funded product — there is no bounty and response is
best-effort.

## Scope

**In scope** — the code in this repository: `src/`, `scripts/`, the shell
drivers, and the dependency set pinned in `uv.lock`.

**Out of scope** — the upstream replication package, the model weights and the
adapter checkpoints. None is versioned here; all are obtained separately and
verified against recorded SHA-256 digests (`upstream/MANIFEST.json`,
`models/culture/ADAPTERS.json`). Report issues in those to their own
maintainers.

Also out of scope: findings that require an attacker to already have write
access to the checkout, since that is write access to the code being run.

## Current advisory dispositions

Seven Dependabot alerts against `uv.lock` are dismissed rather than fixed. None
is fixable without giving up something the reproduction depends on, and none is
reachable from any code path this repository executes. The analysis is recorded
here so the dismissals are reviewable rather than private.

Reviewed 2026-08-06 against `uv.lock` as of commit `4a3ab35`, and re-checked
2026-08-16: `uv.lock` had not changed since `71fdae9` wrote it, the four pinned
versions below still held, and the three reachability greps still returned
nothing on a tree that had since grown the base arm and the adapter-health check.

Re-checked 2026-11-17, with 60 alerts open. The 56 against pillow 12.3.0,
certifi 2026.7.22 and filelock 3.32.2 were fixed by upgrading those three alone
in the lock, to 13.0.0, 2026.10.23 and 4.2.12:

```bash
uv lock --upgrade-package pillow --upgrade-package certifi --upgrade-package filelock
```

None of the three enters a score. Every local culture run records its stack in
its `inference_manifest.json` (torch, transformers, peft, accelerate,
bitsandbytes), all five are unchanged, and no run in `outputs/` has to be
regenerated. The test suite gives the same result under both locks. The other
four alerts are 4 to 7 below; 1 to 3 still hold as written.

### The dependency shape that governs all seven

Every affected subtree is an **optional extra**. A plain `uv sync` installs
none of them: `torch` (and through it `setuptools`), `transformers` and
`accelerate` arrive only with `--extra culture` or `--extra muse`, and
`llama-cpp-python` (and through it `diskcache`) only with `--extra inference`.
The analysis package itself — pandas, numpy, scipy, statsmodels, scikit-learn,
matplotlib, seaborn — pulls in none of them.

```bash
uv tree --frozen --invert --package setuptools     # -> torch, extras: culture, muse
uv tree --frozen --invert --package transformers   # -> 5.8.1 extra: culture; 5.15.0 extra: muse
uv tree --frozen --invert --package accelerate     # -> peft and the project, extras: culture, muse
uv tree --frozen --invert --package diskcache      # -> llama-cpp-python, extra: inference
```

### 1. setuptools 81.0.0 — MANIFEST.in exclusion bypass via Unicode normalization

**Dismissed: vulnerable code is not actually used.**

The advisory concerns `FileList` applying `MANIFEST.in` directives while
building a **source distribution**, on normalization-preserving filesystems
(macOS APFS/HFS+). An excluded non-ASCII filename can be packed into the
`.tar.gz` and published.

This repository never reaches that code:

- the build backend is **hatchling**, not setuptools (`pyproject.toml`,
  `[build-system]`);
- there is no `MANIFEST.in`, and nothing here runs `python -m build --sdist`
  or `setup.py sdist`;
- nothing is published to PyPI — the package is consumed from the working tree;
- setuptools is present only because `torch` declares it as a runtime
  dependency.

**The patched release is also unreachable while torch is pinned.** `torch
2.12.0` requires `setuptools<82`, so the advised `setuptools>=83.0.0` cannot be
resolved:

```
$ uv lock --upgrade-package setuptools     # with constraint setuptools>=83.0.0
  × No solution found when resolving dependencies:
  ╰─▶ Because torch==2.12.0 depends on setuptools<82 and setuptools>=83.0.0,
      we can conclude that torch==2.12.0 cannot be used.
```

Taking this fix therefore means taking the torch upgrade below, and the
constraint that blocks it is the same one that keeps the culture adapters
numerically comparable. Given zero exposure, that is not a trade worth making.
`torch 2.13.0` relaxes the bound to `setuptools>=77.0.3`, so this alert closes
by itself whenever torch moves for an unrelated reason.

### 2. torch 2.12.0 — memory corruption via `torch.jit.script`

**Dismissed: vulnerable code is not actually used.**

Nothing in this repository calls `torch.jit`:

```bash
grep -rn "torch\.jit" src/ scripts/ tests/     # no matches
```

The culture extra uses torch only through `transformers` and `peft` for
eager-mode inference with pre-trained LoRA adapters. No model is scripted,
traced, or compiled, and no untrusted TorchScript is loaded.

The `==2.12.0` pin is deliberate and load-bearing: it is the stack the adapters
were trained against, so inference composes the LoRA onto weights in the same
numeric format as training (`pyproject.toml`, `[project.optional-dependencies]
culture`). Upgrading to 2.13.0 would invalidate the provenance of every culture
run already in `outputs/`, which would have to be regenerated before any of
their numbers could be quoted beside the archived ones.

Revisit if this repository ever adopts `torch.compile`, TorchScript export, or
loading of third-party `.pt` artifacts.

### 3. diskcache 5.6.3 — unsafe pickle deserialization

**Dismissed: vulnerable code is not actually used.** No patched release exists —
5.6.3 is the latest on PyPI, so there is nothing to upgrade to.

diskcache arrives via `llama-cpp-python`, which touches it in exactly one place:

```
.venv/lib/python3.11/site-packages/llama_cpp/llama_cache.py:111
    self.cache = diskcache.Cache(cache_dir)      # inside LlamaDiskCache.__init__
```

`LlamaDiskCache` is opt-in — it takes effect only when a caller passes it to
`Llama.set_cache()`. This repository never does:

```bash
grep -rn "set_cache\|LlamaCache\|LlamaDiskCache" src/ scripts/ tests/   # no matches
```

No cache directory is created and none is read, so no pickle is ever
deserialized. The advisory's threat model also presumes an attacker with write
access to that directory — which, for a local research checkout, is already
write access to the code being run.

Revisit if inference ever enables llama.cpp's disk cache.

### 4. setuptools 81.0.0 — Windows path escape via `namespace_packages.txt`

**Dismissed: vulnerable code is not actually used.** GHSA-xmj3-9gf7-h52w, patched
in 84.1.0, which `torch 2.12.0` cannot take for the reason given in 1:

```bash
printf 'torch==2.12.0\nsetuptools>=84.1.0\n' | uv pip compile - --python-version 3.11
# No solution: torch==2.12.0 depends on setuptools<82
```

The advisory concerns `setuptools.wheel.Wheel.install_as_egg`, the legacy path
that unpacks a wheel into an egg directory, and only on Windows; the attacker
chooses where an `__init__.py` lands, not what it holds. Nothing here installs
an egg: uv installs every wheel itself, the build backend is hatchling, and the
reproduction runs on macOS and Linux. Like 1, it closes when torch moves.

### 5. transformers 5.8.1 — path traversal in `save_pretrained` through chat template names

**Dismissed: vulnerable code is not actually used.** GHSA-xrqw-3rrv-vx5w, patched
in 5.10.0.

A Hub repository can put `../` in the keys of a tokenizer's or processor's
`chat_template`, and `save_pretrained()` then writes outside its directory.
Nothing in this repository saves a tokenizer or a processor, and the tokenizers
it loads come from the base model repositories each run records
(`tokenizer_source` in `inference_manifest.json`).

```bash
grep -rn "save_pretrained" src/ scripts/ tests/    # no matches
```

The `==5.8.1` pin is load-bearing in the same way as torch's (2): 310 of the
manifests under `outputs/culture/` record transformers 5.8.1, and the other 62
are `muse_glimmer_30b` on the `muse` extra's 5.15.0, which lies outside the range
of this advisory and of the next. Moving the pin would change the stack behind
every one of those runs.

### 6. transformers 5.8.1 — custom generation code downloaded before trust consent

**Dismissed: vulnerable code is not actually used.** GHSA-x9r9-c232-4q39 lists no
patched release; its range ends at 5.8.1.

`load_custom_generate()` fetches `custom_generate/generate.py` from a Hub
repository into the module cache before it checks `trust_remote_code`. In 5.8.1
it is reached from `from_pretrained` only when `trust_remote_code=True`, and from
`generate` only through `custom_generate=` or one of the decoding modes moved to
the Hub (DoLa, contrastive search, group and constrained beam search). The
backend passes neither argument: next-token probabilities read the logits of one
forward pass, and full answers call `generate` with `do_sample=True`
(`src/culture/backend.py`).

```bash
grep -rnE "custom_generate|trust_remote_code" src/ scripts/ tests/    # no matches
```

### 7. accelerate 1.13.0 — path traversal through sharded checkpoint `weight_map` entries

**Dismissed: vulnerable code is not actually used.** GHSA-4j2p-28q2-5m79 lists no
patched release for 1.14.0 and earlier, and every manifest under
`outputs/culture/` from a local model records accelerate 1.13.0.

The flaw is in `load_checkpoint_in_model` and `load_checkpoint_and_dispatch`,
which follow the shard paths in a checkpoint's index without checking them.
Nothing here imports accelerate, and neither transformers 5.8.1 nor peft 0.19.1
calls either function; both load weights through their own code.

```bash
grep -rnE "import accelerate|from accelerate|load_checkpoint_(in_model|and_dispatch)" src/ scripts/ tests/   # no matches
grep -rlE "load_checkpoint_(in_model|and_dispatch)" .venv/lib/python3.11/site-packages/{transformers,peft}/      # no matches
```

Revisit 5 to 7 if inference ever saves a tokenizer, passes `trust_remote_code`,
or loads a checkpoint through accelerate directly.
