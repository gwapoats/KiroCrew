"""Multiple bindings preserve credentials and route only to the selected bot."""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import make_mocked_request

from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.config.section_builders import _build_weixin_config
from kiro_crew.config.sections import WeixinAccountConfig, WeixinConfig
from kiro_crew.dashboard.handlers import weixin_qr as qr
from kiro_crew.weixin.accounts import account_section, decode_tokens, merge_token
from kiro_crew.weixin.client import ContextTokenStore, WeixinClient
from kiro_crew.weixin.multi import WeixinAccountsTransport, start_accounts
from kiro_crew.weixin.transport import WeixinTransport
from kiro_crew.weixin.transport_dispatch import WeixinDispatcher


def test_token_bundle_preserves_every_binding():
    bundle = merge_token("original", "family1", "first")
    bundle = merge_token(bundle, "family2", "second")
    assert decode_tokens(bundle) == {"default": "original", "family1": "first", "family2": "second"}
    assert decode_tokens(merge_token(bundle, "family1", "replacement"))["default"] == "original"
    assert decode_tokens("{broken") == {}


def test_accounts_round_trip_and_missing_account_denies():
    section = _build_weixin_config(
        {
            "enabled": True,
            "accounts": {"family1": {"account_id": "bot1", "allowed_user_ids": ["person"]}},
        }
    )
    cfg = KiroCrewConfig(weixin=section)
    restored = _build_weixin_config(cfg.to_dict()["weixin"])
    assert restored.accounts["family1"].allowed_user_ids == ["person"]
    assert account_section(restored, "missing").dm_policy == "disabled"
    restored.accounts["family1"].enabled = False
    assert not account_section(restored, "family1").enabled
    assert account_section(restored, "family1").dm_policy == "disabled"


def test_invalid_account_does_not_become_a_live_binding():
    section = _build_weixin_config(
        {
            "accounts": {
                "../escape": {"account_id": "bad"},
                "default": {"account_id": "bad"},
                "valid": {"enabled": "wrong", "account_id": "bot"},
            }
        }
    )
    assert set(section.accounts) == {"valid"}
    assert not section.accounts["valid"].enabled


def test_add_login_preserves_default_and_enrolls_scanning_user(tmp_path, monkeypatch):
    cp, ep = tmp_path / "config.json", tmp_path / ".env"
    cp.write_text(json.dumps({"weixin": {"enabled": True, "account_id": "old-bot"}}))
    ep.write_text("WEIXIN_TOKEN=old-token\nOTHER=untouched\n")
    monkeypatch.setattr(qr, "config_path", lambda: cp)
    monkeypatch.setattr(qr, "env_path", lambda: ep)
    path, serialized = qr._stage_weixin_config(
        account_id="new-bot",
        base_url="https://ilinkai.weixin.qq.com",
        binding_key="family1",
        user_id="new-user",
    )
    qr._commit_credential_and_config(path, serialized, "new-token", binding_key="family1")
    data = json.loads(cp.read_text())["weixin"]
    assert data["account_id"] == "old-bot"
    assert data["accounts"]["family1"]["allowed_user_ids"] == ["new-user"]
    assert decode_tokens(qr._read_env_value("WEIXIN_TOKEN")) == {
        "default": "old-token",
        "family1": "new-token",
    }
    assert qr._read_env_value("OTHER") == "untouched"
    with pytest.raises(ValueError, match="already bound"):
        qr._stage_weixin_config(account_id="old-bot", base_url="", binding_key="family2")


def test_failed_named_login_restores_entire_bundle(tmp_path, monkeypatch):
    cp, ep = tmp_path / "config.json", tmp_path / ".env"
    original = merge_token("old", "family1", "first")
    ep.write_text(f"WEIXIN_TOKEN={original}\n")
    monkeypatch.setattr(qr, "env_path", lambda: ep)
    write = qr._atomic_write

    def fail_config(path, text, **kwargs):
        if path == cp:
            raise OSError("disk full")
        write(path, text, **kwargs)

    monkeypatch.setattr(qr, "_atomic_write", fail_config)
    with pytest.raises(OSError):
        qr._commit_credential_and_config(cp, "{}", "second", binding_key="family2")
    assert qr._read_env_value("WEIXIN_TOKEN") == original


def child(tmp_path, key, user):
    client = SimpleNamespace(send_message=AsyncMock(return_value={"ret": 0}))
    return WeixinTransport(
        client,
        account_id=f"bot-{key}",
        account_key=key,
        ctx_store=ContextTokenStore(str(tmp_path)),
        allowed_user_ids=[user],
    )


def test_outbound_route_is_explicit_and_revocation_is_enforced(tmp_path):
    async def check():
        manager = WeixinAccountsTransport()
        default = child(tmp_path, "default", "same-user")
        family = child(tmp_path, "family1", "same-user")
        manager.children = {"default": default, "family1": family}
        manager.statuses = {k: {"connected": True} for k in manager.children}
        assert await manager.resolve_configured_target("user:family1/same-user") == (
            "family1/same-user",
            None,
        )
        await manager.send_message("family1/same-user", "hello")
        assert family.client.send_message.await_count == 1
        assert default.client.send_message.await_count == 0
        assert not manager.may_send_to("missing/same-user")
        section = WeixinConfig(
            enabled=True,
            accounts={"family1": WeixinAccountConfig(enabled=False, account_id="bot-family1")},
        )
        family.reconfigure(section)
        assert not manager.may_send_to("family1/same-user")
        with pytest.raises(PermissionError):
            await manager.send_message("family1/same-user", "do not send")

    asyncio.run(check())


def test_named_session_keys_stay_separate_even_when_unified(tmp_path):
    cfg = KiroCrewConfig()
    cfg.messaging.dm_scope = "unified"
    dispatchers = [
        WeixinDispatcher(
            sessions=SimpleNamespace(),
            ctx_builder=SimpleNamespace(),
            cfg=cfg,
            account_id=f"bot-{key}",
            account_key=key,
            ctx_store=ContextTokenStore(str(tmp_path)),
        )
        for key in ("family1", "family2")
    ]
    # Inspect the durable bucket without requiring a real session manager.
    for dispatcher in dispatchers:
        dispatcher._conv = SimpleNamespace(current_gen=lambda user: 0)
    keys = [dispatcher._session_key("same-user") for dispatcher in dispatchers]
    assert keys[0] != keys[1]
    assert all(key.startswith("weixin:") for key in keys)


def test_start_stop_multiple_accounts(tmp_path, monkeypatch):
    async def check():
        cfg = KiroCrewConfig(
            weixin=WeixinConfig(
                enabled=True,
                account_id="default-bot",
                allowed_user_ids=["owner"],
                accounts={
                    "family1": WeixinAccountConfig(
                        account_id="family-bot", allowed_user_ids=["relative"]
                    )
                },
            )
        )
        connected = []

        async def connect(transport):
            connected.append(transport._account_key)

        async def disconnect(transport):
            connected.remove(transport._account_key)

        monkeypatch.setattr(WeixinTransport, "connect", connect)
        monkeypatch.setattr(WeixinTransport, "disconnect", disconnect)
        orch = SimpleNamespace(
            _cfg=cfg,
            _weixin_tokens={"default": "d", "family1": "f"},
            sessions=SimpleNamespace(),
            ctx_builder=SimpleNamespace(),
            dashboard_state=None,
            _approval_mode="auto",
        )
        manager = await start_accounts(orch)
        assert connected == ["default", "family1"]
        assert set(manager.dispatchers) == {"default", "family1"}
        await manager.close()
        assert not connected

    asyncio.run(check())


def test_config_response_reports_selected_binding_without_tokens(monkeypatch):
    cfg = KiroCrewConfig(
        weixin=WeixinConfig(
            enabled=True,
            account_id="default-bot",
            accounts={
                "family1": WeixinAccountConfig(
                    account_id="family-bot", allowed_user_ids=["relative"]
                )
            },
        )
    )
    tokens = merge_token("default-secret", "family1", "family-secret")
    monkeypatch.setattr(qr.KiroCrewConfig, "load", lambda: cfg)
    monkeypatch.setattr(cfg, "load_credentials", lambda: {"WEIXIN_TOKEN": tokens})
    monkeypatch.setattr(qr, "is_direct_local_request", lambda request: True)
    request = make_mocked_request("GET", "/api/weixin/config?account=family1")
    response = asyncio.run(qr.weixin_config_get(request))
    body = json.loads(response.body)
    assert body["credential_set"]
    assert body["account_id"] == "family-bot"
    assert body["accounts"] == ["default", "family1"]
    assert "secret" not in response.text


def test_save_named_disable_does_not_disable_default(tmp_path, monkeypatch):
    cp = tmp_path / "config.json"
    cp.write_text(
        json.dumps(
            {
                "weixin": {
                    "enabled": True,
                    "account_id": "default-bot",
                    "accounts": {"family1": {"enabled": True, "account_id": "family-bot"}},
                }
            }
        )
    )
    monkeypatch.setattr(qr, "config_path", lambda: cp)
    monkeypatch.setattr(qr, "is_direct_local_request", lambda request: True)
    monkeypatch.setattr(qr, "_hot_apply_after_write", AsyncMock())
    request = make_mocked_request("PUT", "/api/weixin/config?account=family1")
    request.json = AsyncMock(return_value={"enabled": False})
    response = asyncio.run(qr.weixin_config_save(request))
    assert response.status == 200
    root = json.loads(cp.read_text())["weixin"]
    assert root["enabled"] is True
    assert root["accounts"]["family1"]["enabled"] is False


def test_remote_cannot_start_a_named_login(monkeypatch):
    monkeypatch.setattr(qr, "is_direct_local_request", lambda request: False)
    request = make_mocked_request("POST", "/api/channels/weixin/qr/start?account=family1")
    response = asyncio.run(qr.weixin_qr_start(request))
    assert response.status == 403


def test_superseded_manager_does_not_remain_registered():
    from kiro_crew.slack.gateway_runtime.channel_lifecycle import _forget_superseded_start

    old, newer = WeixinAccountsTransport(), WeixinAccountsTransport()
    state = SimpleNamespace(channel_transports={"weixin": old})
    orch = SimpleNamespace(dashboard_state=state, _weixin_client=old)
    _forget_superseded_start(orch, "weixin", old)
    assert "weixin" not in state.channel_transports
    state.channel_transports["weixin"] = newer
    _forget_superseded_start(orch, "weixin", old)
    assert state.channel_transports["weixin"] is newer


@pytest.mark.parametrize("scrubbed", [False, True])
def test_confirmed_qr_replaces_cached_process_credential(tmp_path, monkeypatch, scrubbed):
    cp, ep = tmp_path / "config.json", tmp_path / ".env"
    cp.write_text(json.dumps({"weixin": {"enabled": True, "account_id": "default-bot"}}))
    ep.write_text("WEIXIN_TOKEN=original\n")
    monkeypatch.setattr(qr, "config_path", lambda: cp)
    monkeypatch.setattr(qr, "env_path", lambda: ep)
    monkeypatch.setattr(qr, "is_direct_local_request", lambda request: True)
    if scrubbed:
        monkeypatch.setenv("_KIROCREW_CREDS_SCRUBBED", "1")
        monkeypatch.delenv("WEIXIN_TOKEN", raising=False)
    else:
        monkeypatch.setenv("WEIXIN_TOKEN", "original")
        monkeypatch.delenv("_KIROCREW_CREDS_SCRUBBED", raising=False)
    client = SimpleNamespace(
        get_qrcode_status=AsyncMock(
            return_value={
                "status": "confirmed",
                "bot_token": "replacement",
                "ilink_bot_id": "family-bot",
                "ilink_user_id": "family-user",
            }
        ),
        close=AsyncMock(),
    )
    monkeypatch.setitem(
        qr._SESSIONS,
        "login",
        {
            "client": client,
            "qrcode": "fake-code",
            "binding_key": "family1",
        },
    )
    request = make_mocked_request("GET", "/api/channels/weixin/qr/status?session_id=login")
    response = asyncio.run(qr.weixin_qr_status(request))
    assert response.status == 200
    expected = {"default": "original", "family1": "replacement"}
    assert decode_tokens(qr._read_env_value("WEIXIN_TOKEN")) == expected
    if scrubbed:
        assert "WEIXIN_TOKEN" not in os.environ
    else:
        assert decode_tokens(os.environ["WEIXIN_TOKEN"]) == expected


def test_closing_client_cancels_its_poll_task():
    async def check():
        client = WeixinClient(token="fake-token")
        entered = asyncio.Event()
        wait = asyncio.Event()

        async def poll():
            entered.set()
            await wait.wait()

        task = asyncio.create_task(poll())
        client._poll_task = task
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            await client.close()
            assert task.done()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def test_transport_registers_poll_for_client_teardown(tmp_path, monkeypatch):
    async def check():
        client = WeixinClient(token="fake-token")
        entered = asyncio.Event()
        wait = asyncio.Event()

        async def blocked_poll(cursor):
            entered.set()
            await wait.wait()
            return {"ret": 0, "msgs": []}

        monkeypatch.setattr(client, "get_updates", blocked_poll)
        transport = WeixinTransport(
            client,
            account_id="fake-bot",
            ctx_store=ContextTokenStore(str(tmp_path)),
        )
        await transport.connect()
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            assert client._poll_task is transport._poll_task
            await client.close()
            assert transport._poll_task.done()
            assert client._session.closed
        finally:
            await transport.disconnect()

    asyncio.run(check())
