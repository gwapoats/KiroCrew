"""Named iLink connections and backward-compatible protected token bundles."""

from __future__ import annotations

import json
import re
from dataclasses import replace

from kiro_crew.config.sections import WeixinConfig


def valid_account_key(key: str) -> bool:
    return key == "default" or bool(re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", key))


def decode_tokens(value: str) -> dict[str, str]:
    """A legacy token is the default binding; malformed bundles authorize nobody."""
    if not value:
        return {}
    if not value.lstrip().startswith("{"):
        return {"default": value}
    try:
        data = json.loads(value)
        if data.get("version") != 1 or not isinstance(data.get("accounts"), dict):
            return {}
        return {
            key: token
            for key, token in data["accounts"].items()
            if valid_account_key(key) and isinstance(token, str) and token
        }
    except (ValueError, TypeError, AttributeError):
        return {}


def merge_token(previous: str, key: str, token: str) -> str:
    if not valid_account_key(key):
        raise ValueError("Invalid WeChat account key")
    tokens = decode_tokens(previous)
    tokens[key] = token
    if set(tokens) == {"default"}:
        return tokens["default"]
    return json.dumps({"version": 1, "accounts": tokens}, separators=(",", ":"))


def account_section(section: WeixinConfig, key: str) -> WeixinConfig:
    """Resolve a binding without letting a missing/deleted account fall back."""
    if key == "default":
        return replace(section, enabled=section.enabled and section.default_enabled)
    account = section.accounts.get(key)
    if account is None:
        return replace(section, enabled=False, dm_policy="disabled", allowed_user_ids=[])
    return replace(
        section,
        enabled=section.enabled and account.enabled,
        token="",
        account_id=account.account_id,
        base_url=account.base_url,
        dm_policy="allowlist" if section.enabled and account.enabled else "disabled",
        allowed_user_ids=list(account.allowed_user_ids),
    )


def route_id(key: str, user_id: str) -> str:
    return user_id if key == "default" else f"{key}/{user_id}"
