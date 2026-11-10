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

Three Dependabot alerts are open against `uv.lock`. None is fixable without
giving up something the reproduction depends on, and none is reachable from any
code path this repository executes. The analysis is recorded here so the
dismissals are reviewable rather than private.

Reviewed 2026-08-06 against `uv.lock` as of commit `4a3ab35`, and re-checked
2026-08-16: `uv.lock` has not changed since `71fdae9` wrote it, the four pinned
versions below still hold, and the three reachability greps still return nothing
on a tree that has since grown the base arm and the adapter-health check.

### The dependency shape that governs all three

Both affected subtrees are **optional extras**. A plain `uv sync` installs
neither: `torch` (and through it `setuptools`) arrives only with
`--extra culture`, and `llama-cpp-python` (and through it `diskcache`) only with
`--extra inference`. The analysis package itself — pandas, numpy, scipy,
statsmodels, scikit-learn, matplotlib, seaborn — pulls in neither.

```bash
uv tree --frozen --invert --package setuptools   # -> torch, extra: culture
uv tree --frozen --invert --package diskcache    # -> llama-cpp-python, extra: inference
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
