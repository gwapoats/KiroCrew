"""One registered Weixin transport with explicit per-binding delivery routes."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from kiro_crew.config.paths import data_home
from kiro_crew.messaging.transport import (
    ConfiguredChannelTarget,
    InboundMessage,
    MessagingTransport,
)
from kiro_crew.weixin.accounts import account_section, route_id
from kiro_crew.weixin.client import ContextTokenStore, TypingTicketCache, WeixinClient
from kiro_crew.weixin.transport import WEIXIN_CAPABILITIES, WeixinTransport
from kiro_crew.weixin.transport_dispatch import WeixinDispatcher

logger = logging.getLogger(__name__)


class WeixinAccountsTransport(MessagingTransport):
    """Keep default routes compatible; named routes are ``account/user``."""

    channel_type = "weixin"
    capabilities = WEIXIN_CAPABILITIES

    def __init__(self) -> None:
        self.children: dict[str, WeixinTransport] = {}
        self.dispatchers: dict[str, WeixinDispatcher] = {}
        self.statuses: dict[str, dict[str, Any]] = {}
        self.on_state_change: Any = None

    def _resolve(self, conversation: str) -> tuple[WeixinTransport, str] | None:
        key, sep, user = conversation.partition("/")
        if not sep:
            key, user = "default", conversation
        child = self.children.get(key)
        return (child, user) if child is not None and user else None

    def configured_targets(self) -> list[ConfiguredChannelTarget]:
        return [
            ConfiguredChannelTarget(
                f"user:{route_id(key, target.target_id.removeprefix('user:'))}",
                f"{key} · {target.label}",
                available=bool(self.statuses.get(key, {}).get("connected")),
            )
            for key, child in self.children.items()
            for target in child.configured_targets()
        ]

    async def resolve_configured_target(self, target_id: str) -> tuple[str, str | None] | None:
        if not target_id.startswith("user:"):
            return None
        conversation = target_id.removeprefix("user:")
        resolved = self._resolve(conversation)
        if resolved is None:
            return None
        child, user = resolved
        if await child.resolve_configured_target(f"user:{user}") is None:
            return None
        return conversation, None

    def may_send_to(
        self, conversation_id: str, thread_id: str | None = None, *, principal: str = ""
    ) -> bool:
        resolved = self._resolve(conversation_id)
        key = conversation_id.partition("/")[0] if "/" in conversation_id else "default"
        return bool(
            resolved
            and self.statuses.get(key, {}).get("connected")
            and resolved[0].may_send_to(resolved[1], principal=principal)
        )

    async def send_message(
        self, conversation_id: str, content: str, thread_id: str | None = None
    ) -> str:
        resolved = self._resolve(conversation_id)
        if resolved is None or not self.may_send_to(conversation_id, principal=resolved[1]):
            raise PermissionError("WeChat binding or recipient is unavailable")
        child, user = resolved
        return await child.send_message(user, content)

    async def resolve_conversation(self, user_id: str) -> str:
        if self._resolve(user_id) is None:
            raise ValueError("An explicit WeChat binding is required")
        return user_id

    async def fetch_history(
        self, conversation_id: str, thread_id: str | None = None
    ) -> list[InboundMessage]:
        return []

    async def receive(self, raw_envelope: Any) -> None:
        raise ValueError("Receive on the account-specific transport")

    def authorize(self, msg: InboundMessage) -> bool:
        resolved = self._resolve(msg.conversation_id)
        return bool(resolved and resolved[0].authorize(msg))

    async def close(self) -> None:
        for status in self.statuses.values():
            status["connected"] = False
        await asyncio.gather(*(child.disconnect() for child in self.children.values()))
        polls = [
            child._poll_task
            for child in self.children.values()
            if child._poll_task is not None and child._poll_task is not asyncio.current_task()
        ]
        if polls:
            await asyncio.gather(*polls, return_exceptions=True)

    async def disconnect(self) -> None:
        await self.close()


async def start_accounts(orch: Any) -> WeixinAccountsTransport:
    from kiro_crew.weixin.gateway import _resolve_approval_mode

    manager = WeixinAccountsTransport()
    section = orch._cfg.weixin
    tokens = orch._weixin_tokens
    seen: set[str] = set()
    for key in ["default", *section.accounts]:
        selected = account_section(section, key)
        status: dict[str, Any] = {"connected": False, "connect_error": ""}
        manager.statuses[key] = status
        token = tokens.get(key, "")
        if not selected.enabled:
            continue
        if not token or not selected.account_id:
            status["connect_error"] = "Account needs QR sign-in"
            continue
        if selected.account_id in seen:
            status["connect_error"] = "Duplicate bot binding; only one poller is allowed"
            continue
        seen.add(selected.account_id)
        client = WeixinClient(
            token=token, base_url=selected.base_url, account_id=selected.account_id
        )
        store = ContextTokenStore(str(data_home()))
        dispatcher = WeixinDispatcher(
            sessions=orch.sessions,
            ctx_builder=orch.ctx_builder,
            cfg=orch._cfg,
            account_id=selected.account_id,
            account_key=key,
            ctx_store=store,
            typing_cache=TypingTicketCache(),
            conv_log=getattr(orch, "conv_log", None),
            approval_mode=_resolve_approval_mode(orch),
        )
        dispatcher.client = client
        child = WeixinTransport(
            client,
            account_id=selected.account_id,
            account_key=key,
            ctx_store=store,
            allowed_user_ids=selected.allowed_user_ids,
            dm_policy=selected.dm_policy,
            dispatch=dispatcher.handle_message,
        )
        dispatcher.transport = child
        manager.children[key] = child
        manager.dispatchers[key] = dispatcher

        def on_state(connected: bool, error: str, *, account: str = key) -> None:
            manager.statuses[account] = {"connected": connected, "connect_error": error[:120]}
            if orch.dashboard_state is not None:
                orch.dashboard_state.weixin_connected = any(
                    row["connected"] for row in manager.statuses.values()
                )

        child.on_state_change = on_state
        try:
            await child.connect()
            on_state(True, "")
        except asyncio.CancelledError:
            await manager.close()
            raise
        except Exception:
            logger.exception("WeChat account %s failed to connect", key)
            await child.disconnect()
            on_state(False, "Connection failed; see gateway log")
    state = orch.dashboard_state
    if state is not None:
        state.register_channel_transport(manager)
        for dispatcher in manager.dispatchers.values():
            dispatcher.dashboard_state = state
        state.weixin_connected = any(row["connected"] for row in manager.statuses.values())
        state.weixin_connect_error = "" if state.weixin_connected else "No WeChat binding connected"
    return manager
