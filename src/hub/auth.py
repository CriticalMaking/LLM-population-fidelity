from __future__ import annotations

import os
from typing import Any

LOGIN_HINT = (
    "No Hugging Face token found. Either:\n"
    "  hf auth login                 (interactive, stores a token)\n"
    "  export HF_TOKEN=hf_...        (write scope needed for push)"
)


def read_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def resolve_api(*, write: bool = False) -> Any:
    from huggingface_hub import HfApi

    token = read_token()
    if token is None:
        raise SystemExit(LOGIN_HINT)

    api = HfApi(token=token)
    try:
        identity = api.whoami(cache=True)
    except Exception as error:
        raise SystemExit(f"Hugging Face token rejected: {error}") from error

    if write:
        permission = ((identity.get("auth") or {}).get("accessToken") or {}).get("role")
        if permission == "read":
            raise SystemExit("Token has 'read' scope; a write-scoped token is required.")
    return api


def describe_identity(api: Any) -> str:
    identity = api.whoami(cache=True)
    name = identity.get("name", "unknown")
    orgs = [entry["name"] for entry in identity.get("orgs", [])]
    suffix = f" (orgs: {', '.join(orgs)})" if orgs else ""
    return f"{name}{suffix}"
