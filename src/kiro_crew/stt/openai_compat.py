"""The ``openai`` STT provider: one recording to an OpenAI-compatible ``/audio/transcriptions``.

Batch only. The server, model and prompt come from ``stt.openai_*``, the key from
the ``STT_OPENAI_API_KEY`` vault entry, never ``.env``. Every failure raises
:class:`SttRequestError`, which :func:`kiro_crew.transcribe.transcribe_audio` turns
into ``None`` and keeps for :func:`kiro_crew.transcribe.last_failure`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from kiro_crew.config.sections_stt import STT_LANGUAGE_AUTO
from kiro_crew.transcribe_ffmpeg import _input_suffix

logger = logging.getLogger(__name__)

API_KEY_SECRET = "STT_OPENAI_API_KEY"
CODE_NOT_CONFIGURED = "stt_openai_not_configured"
CODE_UNREACHABLE = "stt_openai_unreachable"
CODE_TIMEOUT = "stt_openai_timeout"
CODE_AUTH = "stt_openai_auth"
CODE_HTTP_ERROR = "stt_openai_http_error"
CODE_BAD_RESPONSE = "stt_openai_bad_response"
CODE_TOO_LARGE = "stt_openai_audio_too_large"

#: The OpenAI API's per-file limit, and the dashboard upload's.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
#: A transcript is text; a longer answer is not one.
_MAX_RESPONSE_BYTES = 1024 * 1024
_CONNECT_TIMEOUT_SECS = 10.0


class SttRequestError(Exception):
    """A failed transcription: a code, a sentence a caller can show, an HTTP status."""

    def __init__(self, code: str, detail: str, http_status: int = 502) -> None:
        super().__init__(detail)
        self.code, self.detail, self.http_status = code, detail, http_status


def config_problem(stt_config: Any) -> str | None:
    """Why no request can be made, from config alone; ``None`` when one can."""
    url = stt_config.openai_base_url
    if not url:
        return "Set stt.openai_base_url to the server's URL, e.g. http://127.0.0.1:8000/v1."
    if not stt_config.openai_model:
        return "Set stt.openai_model to the model the server should run."
    from kiro_crew.decisions.local_models import is_loopback_endpoint

    if urlsplit(url).scheme != "https" and not is_loopback_endpoint(url):
        return (
            "stt.openai_base_url uses plain http to a host other than a loopback IP "
            "address. Use https, or 127.0.0.1 (not localhost) for a server on this machine."
        )
    return None


def _api_key() -> str:
    """The server's API key from the vault, or ``""``. BLOCKING."""
    from kiro_crew.config.paths import config_dir
    from kiro_crew.secrets.vault import SecretVault

    try:
        secret = SecretVault(config_dir()).get(API_KEY_SECRET)
    except Exception as exc:
        logger.warning("Could not read the %s secret (%s)", API_KEY_SECRET, type(exc).__name__)
        return ""
    return secret.reveal() if secret is not None else ""


def _read_audio(audio_path: str) -> bytes:
    """The recording, refused past the upload limit. BLOCKING."""
    with open(audio_path, "rb") as fh:
        data = fh.read(_MAX_UPLOAD_BYTES + 1)
    if len(data) > _MAX_UPLOAD_BYTES:
        raise SttRequestError(CODE_TOO_LARGE, "The recording is larger than 25 MB.", 413)
    return data


def _excerpt(body: bytes) -> str:
    """The server's own error message, on one short credential-redacted line."""
    from kiro_crew.security import redact_credentials

    text = body[:4096].decode("utf-8", "replace")
    try:
        message = json.loads(text)
        message = message.get("error", message.get("detail"))
        message = message.get("message") if isinstance(message, dict) else message
    except (ValueError, AttributeError):
        message = None
    text = message if isinstance(message, str) else text
    return redact_credentials(" ".join(text.split())[:200])[0]


def _form(data: bytes, audio_path: str, stt_config: Any) -> aiohttp.FormData:
    form = aiohttp.FormData()
    name = "recording" + (_input_suffix(audio_path) or "")
    form.add_field("file", data, filename=name, content_type="application/octet-stream")
    form.add_field("model", stt_config.openai_model)
    form.add_field("response_format", "json")
    code = stt_config.language_code
    if code and code != STT_LANGUAGE_AUTO:
        form.add_field("language", code.split("-", 1)[0].lower())  # ISO 639: zh for zh-CN
    if stt_config.openai_prompt:
        form.add_field("prompt", stt_config.openai_prompt)
    return form


async def transcribe_openai(audio_path: str, stt_config: Any) -> str:
    """The transcript of *audio_path*, ``""`` when nothing was heard."""
    problem = config_problem(stt_config)
    if problem:
        raise SttRequestError(CODE_NOT_CONFIGURED, problem, 503)
    base = stt_config.openai_base_url
    server = "{0.scheme}://{0.netloc}".format(urlsplit(base))
    data = await asyncio.to_thread(_read_audio, audio_path)
    key = await asyncio.to_thread(_api_key)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    total = float(stt_config.timeout_secs)
    timeout = aiohttp.ClientTimeout(total=total, connect=min(_CONNECT_TIMEOUT_SECS, total))
    body = b""
    try:
        # Redirects are not followed, so the key never reaches another host.
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{base}/audio/transcriptions",
                data=_form(data, audio_path, stt_config),
                headers=headers,
                allow_redirects=False,
            ) as resp:
                status = resp.status
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    body += chunk
                    if len(body) > _MAX_RESPONSE_BYTES:
                        break
    except asyncio.TimeoutError:
        detail = f"The speech server at {server} did not answer within {total:g} s."
        raise SttRequestError(CODE_TIMEOUT, detail, 504) from None
    except aiohttp.ClientError as exc:
        refused = isinstance(getattr(exc, "os_error", None), ConnectionRefusedError)
        reason = "connection refused" if refused else type(exc).__name__
        detail = f"Could not reach the speech server at {server}: {reason}."
        raise SttRequestError(CODE_UNREACHABLE, detail) from None

    if status in (401, 403):
        detail = (
            f"The speech server at {server} refused the request (HTTP {status}). "
            f"Check the {API_KEY_SECRET} secret in Settings -> Secrets."
        )
        raise SttRequestError(CODE_AUTH, detail)
    if not 200 <= status < 300:
        reason = "redirects are not followed" if status < 400 else _excerpt(body)
        detail = f"The speech server at {server} answered HTTP {status}"
        raise SttRequestError(CODE_HTTP_ERROR, f"{detail}: {reason}" if reason else f"{detail}.")
    try:
        text = json.loads(body.decode("utf-8")).get("text")
    except (ValueError, AttributeError):
        text = None
    if len(body) > _MAX_RESPONSE_BYTES or not isinstance(text, str):
        raise SttRequestError(
            CODE_BAD_RESPONSE, f"The speech server at {server} sent no transcript."
        )
    return text.strip()
