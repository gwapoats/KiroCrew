"""The ``openai`` STT provider: its config, the request it sends, and its failures.

A loopback ``TestServer`` stands in for the OpenAI-compatible server, so every test
sees the request as a real server would and needs no network beyond 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientConnectorError, web
from aiohttp.test_utils import TestServer
from dashboard_handlers_core_coverage_helpers import _req

from kiro_crew import transcribe
from kiro_crew.config import sections_stt
from kiro_crew.config.section_builders import _build_stt_config
from kiro_crew.config.sections import STT_PROVIDER_OPENAI, SttConfig, openai_base_url
from kiro_crew.dashboard.handlers import core as core_mod
from kiro_crew.dashboard.handlers.core_stt_settings import _stt_language_codes
from kiro_crew.dashboard.stt_stream import streaming_providers
from kiro_crew.stt import openai_compat

KEY = "sk-test-openai-compat-0123456789abcdef"
pytestmark = pytest.mark.timeout(30)


def _cfg(url: str = "http://127.0.0.1:1/v1", **fields) -> SttConfig:
    fields.setdefault("openai_model", "whisper-test")
    return SttConfig(provider=STT_PROVIDER_OPENAI, openai_base_url=url, **fields)


def _clip(tmp_path: Path, name: str = "clip.webm", size: int = 400) -> str:
    path = tmp_path / name
    path.write_bytes(b"\x1a\x45\xdf\xa3" * (size // 4))
    return str(path)


def _serve(handler, call):
    """``call(base_url)`` against a loopback server whose transcription route is *handler*."""

    async def main():
        app = web.Application()
        app.router.add_post("/v1/audio/transcriptions", handler)
        server = TestServer(app)
        await server.start_server()
        try:
            return await call(str(server.make_url("/v1")))
        finally:
            await server.close()

    return asyncio.run(main())


def _recorder(seen: dict, *, status: int = 200, payload: object = None, raw: bytes | None = None):
    async def handler(request: web.Request) -> web.StreamResponse:
        form = await request.post()
        upload = form["file"]
        seen["auth"] = request.headers.get("Authorization")
        seen["fields"] = {k: v for k, v in form.items() if k != "file"}
        seen["filename"] = upload.filename
        seen["size"] = len(upload.file.read())
        if raw is not None:
            return web.Response(body=raw, status=status, content_type="text/plain")
        return web.json_response(
            {"text": " hello world "} if payload is None else payload, status=status
        )

    return handler


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setattr(openai_compat, "_api_key", lambda: "")


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr(openai_compat, "_api_key", lambda: KEY)


async def _fail(audio_path: str, cfg: SttConfig) -> openai_compat.SttRequestError:
    """Run the provider through ``transcribe_audio`` and return the failure it kept."""
    assert await transcribe.transcribe_audio(audio_path, cfg) is None
    failure = transcribe.last_failure()
    assert failure is not None
    return failure


# ── config ──────────────────────────────────────────────────────────────


class TestConfig:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1"),
            ("https://api.openai.com/v1", "https://api.openai.com/v1"),
            ("  ", ""),
            ("https://user:pass@example.com/v1", None),
            ("https://example.com/v1?key=x", None),
            ("https://example.com/v1#x", None),
            ("ftp://example.com/v1", None),
            ("http://example.com:99999/v1", None),
            ("http:///v1", None),
            (42, None),
        ],
    )
    def test_base_url_shape(self, value, expected) -> None:
        assert openai_base_url(value) == expected

    def test_a_refused_url_is_dropped_and_never_logged(self, caplog) -> None:
        sections_stt._WARNED_OPENAI_BASE_URLS.clear()
        with caplog.at_level(logging.WARNING):
            cfg = _build_stt_config({"openai_base_url": f"https://me:{KEY}@example.com/v1"})
        assert cfg.openai_base_url == ""
        assert "stt.openai_base_url" in caplog.text
        assert KEY not in caplog.text

    def test_model_and_prompt_are_stripped_strings(self) -> None:
        cfg = _build_stt_config({"openai_model": " m ", "openai_prompt": 7})
        assert (cfg.openai_model, cfg.openai_prompt) == ("m", "")

    def test_auto_language_survives_for_openai(self) -> None:
        cfg = _cfg(language_code="auto")
        assert cfg.effective_language_code == "auto"
        assert _stt_language_codes(cfg)[0] == "auto"


class TestAvailability:
    @pytest.mark.parametrize(
        "url",
        ["http://127.0.0.1:8000/v1", "http://[::1]:8000/v1", "https://example.com/v1"],
    )
    def test_usable_destinations(self, url) -> None:
        assert transcribe.availability_detail(_cfg(url)).ok

    @pytest.mark.parametrize(
        "cfg, fragment",
        [
            (_cfg(""), "Set stt.openai_base_url"),
            (_cfg(openai_model=""), "Set stt.openai_model"),
            (_cfg("http://10.0.0.5:8000/v1"), "plain http"),
            (_cfg("http://localhost:8000/v1"), "not localhost"),
        ],
    )
    def test_refusals_name_the_setting(self, cfg, fragment) -> None:
        detail = transcribe.availability_detail(cfg)
        assert not detail.ok
        assert detail.code == transcribe.CODE_PROVIDER_UNAVAILABLE
        assert fragment in detail.detail

    def test_batch_only(self) -> None:
        assert STT_PROVIDER_OPENAI not in streaming_providers()
        assert transcribe.batch_duration_cap_secs(_cfg()) is None


# ── the request ─────────────────────────────────────────────────────────


class TestRequest:
    def test_fields_header_and_file(self, tmp_path, with_key) -> None:
        seen: dict = {}
        clip = _clip(tmp_path)
        cfg_kw = {"language_code": "zh-CN", "openai_prompt": "Kiro Crew, MCP"}
        text = _serve(
            _recorder(seen),
            lambda url: openai_compat.transcribe_openai(clip, _cfg(url, **cfg_kw)),
        )
        assert text == "hello world"
        assert seen["auth"] == f"Bearer {KEY}"
        assert seen["fields"] == {
            "model": "whisper-test",
            "response_format": "json",
            "language": "zh",
            "prompt": "Kiro Crew, MCP",
        }
        assert (seen["filename"], seen["size"]) == ("recording.webm", 400)

    def test_auto_language_empty_prompt_and_no_key_are_left_out(self, tmp_path, no_key) -> None:
        seen: dict = {}
        clip = _clip(tmp_path, "clip.ogg")
        _serve(_recorder(seen), lambda url: openai_compat.transcribe_openai(clip, _cfg(url)))
        assert seen["auth"] is None
        assert seen["fields"] == {"model": "whisper-test", "response_format": "json"}
        assert seen["filename"] == "recording.ogg"

    def test_transcribe_audio_dispatches_and_redacts(self, tmp_path, no_key) -> None:
        leaked = "AKIAIOSFODNN7EXAMPLE"
        clip = _clip(tmp_path)
        out = _serve(
            _recorder({}, payload={"text": f"my key is {leaked}"}),
            lambda url: transcribe.transcribe_audio(clip, _cfg(url)),
        )
        assert out and leaked not in out
        assert transcribe.last_failure() is None

    def test_silence_is_an_empty_success(self, tmp_path, no_key) -> None:
        clip = _clip(tmp_path)
        out = _serve(
            _recorder({}, payload={"text": "  "}),
            lambda url: transcribe.transcribe_audio(clip, _cfg(url)),
        )
        assert out == ""


# ── failures ────────────────────────────────────────────────────────────


class TestFailures:
    def test_401_names_the_secret_and_never_the_key(self, tmp_path, with_key, caplog) -> None:
        clip = _clip(tmp_path)
        with caplog.at_level(logging.DEBUG):
            failure = _serve(
                _recorder({}, status=401, payload={"error": {"message": f"bad key {KEY}"}}),
                lambda url: _fail(clip, _cfg(url)),
            )
        assert failure.code == openai_compat.CODE_AUTH
        assert failure.http_status == 502
        assert "HTTP 401" in failure.detail and openai_compat.API_KEY_SECRET in failure.detail
        assert KEY not in failure.detail and KEY not in caplog.text

    def test_other_status_carries_the_servers_message(self, tmp_path, no_key) -> None:
        clip = _clip(tmp_path)
        failure = _serve(
            _recorder({}, status=404, payload={"detail": "Model 'x' not found. Available: y"}),
            lambda url: _fail(clip, _cfg(url)),
        )
        assert failure.code == openai_compat.CODE_HTTP_ERROR
        assert "HTTP 404: Model 'x' not found. Available: y" in failure.detail

    def test_a_redirect_is_not_followed(self, tmp_path, with_key) -> None:
        hits: list[str] = []

        async def redirect(request: web.Request) -> web.Response:
            await request.read()
            raise web.HTTPTemporaryRedirect("/v1/elsewhere")

        async def elsewhere(request: web.Request) -> web.Response:
            hits.append(request.headers.get("Authorization", ""))
            return web.json_response({"text": "followed"})

        async def main():
            app = web.Application()
            app.router.add_post("/v1/audio/transcriptions", redirect)
            app.router.add_post("/v1/elsewhere", elsewhere)
            server = TestServer(app)
            await server.start_server()
            try:
                return await _fail(_clip(tmp_path), _cfg(str(server.make_url("/v1"))))
            finally:
                await server.close()

        failure = asyncio.run(main())
        assert failure.code == openai_compat.CODE_HTTP_ERROR
        assert "HTTP 307: redirects are not followed" in failure.detail
        assert hits == []

    @pytest.mark.parametrize("raw", [b"<html>not json</html>", b'{"result": "no text key"}'])
    def test_an_answer_without_text_is_a_bad_response(self, tmp_path, no_key, raw) -> None:
        clip = _clip(tmp_path)
        failure = _serve(_recorder({}, raw=raw), lambda url: _fail(clip, _cfg(url)))
        assert failure.code == openai_compat.CODE_BAD_RESPONSE

    def test_an_oversized_answer_is_a_bad_response(self, tmp_path, no_key, monkeypatch) -> None:
        monkeypatch.setattr(openai_compat, "_MAX_RESPONSE_BYTES", 16)
        clip = _clip(tmp_path)
        failure = _serve(
            _recorder({}, payload={"text": "x" * 64}), lambda url: _fail(clip, _cfg(url))
        )
        assert failure.code == openai_compat.CODE_BAD_RESPONSE

    @pytest.mark.parametrize(
        "os_error, reason",
        [
            (ConnectionRefusedError(errno.ECONNREFUSED, "refused"), "connection refused"),
            (TimeoutError(60, "Operation timed out"), "ClientConnectorError"),
        ],
        ids=["refused", "macos-connect-timeout"],
    )
    def test_connection_error(self, tmp_path, no_key, monkeypatch, os_error, reason) -> None:
        # A non-listening port can time out on macOS; pin the OS error at the HTTP seam.
        url = "http://127.0.0.1:8000/v1"
        connection = SimpleNamespace(host="127.0.0.1", port=8000, ssl=False)
        post = Mock(side_effect=ClientConnectorError(connection, os_error))
        monkeypatch.setattr(openai_compat.aiohttp.ClientSession, "post", post)
        failure = asyncio.run(_fail(_clip(tmp_path), _cfg(url)))
        assert failure.code == openai_compat.CODE_UNREACHABLE
        assert failure.detail == f"Could not reach the speech server at {url[:-3]}: {reason}."
        assert post.call_args.args == (f"{url}/audio/transcriptions",)

    def test_timeout(self, tmp_path, no_key) -> None:
        release = asyncio.Event()

        async def stall(request: web.Request) -> web.Response:
            await request.read()
            await asyncio.wait_for(release.wait(), timeout=10)
            return web.json_response({"text": "late"})

        async def call(url: str):
            try:
                return await _fail(_clip(tmp_path), _cfg(url, timeout_secs=1))
            finally:
                release.set()

        failure = _serve(stall, call)
        assert failure.code == openai_compat.CODE_TIMEOUT
        assert failure.http_status == 504

    def test_an_oversized_recording_is_refused_before_any_request(
        self, tmp_path, no_key, monkeypatch
    ) -> None:
        monkeypatch.setattr(openai_compat, "_MAX_UPLOAD_BYTES", 10)
        seen: dict = {}
        clip = _clip(tmp_path)
        failure = _serve(_recorder(seen), lambda url: _fail(clip, _cfg(url)))
        assert (failure.code, failure.http_status) == (openai_compat.CODE_TOO_LARGE, 413)
        assert seen == {}

    def test_an_unconfigured_provider_is_refused_without_a_request(self, tmp_path) -> None:
        failure = asyncio.run(_fail(_clip(tmp_path), _cfg("http://10.0.0.5/v1")))
        assert (failure.code, failure.http_status) == (openai_compat.CODE_NOT_CONFIGURED, 503)

    def test_the_kept_failure_is_cleared_by_the_next_call(self, tmp_path, no_key) -> None:
        clip = _clip(tmp_path)

        async def twice(url: str):
            await _fail(clip, _cfg("http://10.0.0.5/v1"))
            assert await transcribe.transcribe_audio(clip, _cfg(url)) == "hello world"
            return transcribe.last_failure()

        assert _serve(_recorder({}), twice) is None


class TestBatchEndpoint:
    @pytest.mark.asyncio
    async def test_a_failure_with_a_reason_keeps_its_code_and_status(self, monkeypatch) -> None:
        monkeypatch.setattr(
            core_mod, "availability_detail", lambda _cfg: core_mod.stt.Availability(True)
        )
        monkeypatch.setattr(core_mod, "batch_duration_cap_secs", lambda _cfg: None)

        async def refused(_path, _cfg):
            transcribe._LAST_FAILURE.set(
                openai_compat.SttRequestError(openai_compat.CODE_TIMEOUT, "too slow", 504)
            )
            return None

        monkeypatch.setattr("kiro_crew.transcribe.transcribe_audio", refused)
        field = SimpleNamespace(
            name="audio",
            filename="recording.webm",
            read_chunk=AsyncMock(side_effect=[b"x", b""]),
        )
        req = _req()
        req.multipart = AsyncMock(return_value=SimpleNamespace(next=AsyncMock(return_value=field)))

        resp = await core_mod.api_stt_transcribe(req)

        assert resp.status == 504
        assert json.loads(resp.body) == {"error": "too slow", "code": "stt_openai_timeout"}
