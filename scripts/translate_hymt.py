"""Hy-MT2 client and optional local llama-server lifecycle management.

The ASR process talks to an OpenAI-compatible llama.cpp server instead of
loading the translation model in every speaker worker.  One shared model can
therefore translate Japanese, English, and Korean in every direction while
llama-server owns GPU batching and VRAM.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable


SUPPORTED_LANGS = ("ja", "en", "ko")
LANGUAGE_NAMES = {"ja": "Japanese", "en": "English", "ko": "Korean"}
TARGETS_BY_SOURCE = {
    source: tuple(lang for lang in SUPPORTED_LANGS if lang != source)
    for source in SUPPORTED_LANGS
}
DEFAULT_URL = "http://127.0.0.1:18081"
DEFAULT_MODEL_NAME = "Hy-MT2-1.8B-Q4_K_M.gguf"


class HyMTError(RuntimeError):
    """The translation service is unavailable or returned an invalid result."""


def translation_targets(source_lang: str) -> tuple[str, ...]:
    """Return the other two trilingual targets in stable display order."""
    return TARGETS_BY_SOURCE.get(source_lang, ())


def _api_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _health_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def llama_server_command(executable: str, model_path: str, host: str,
                         port: int, parallel: int) -> list[str]:
    """Build argv for either classic llama-server or unified llama.app."""
    command = [executable]
    if os.path.basename(executable) in {"llama", "llama.exe"}:
        command.append("serve")
    command += [
        "-m", model_path, "-ngl", "99", "-c", "2048",
        "-np", str(max(1, parallel)), "--host", host,
        "--port", str(port), "--no-warmup",
    ]
    return command


class HyMTClient:
    """Small stdlib-only client for a Hy-MT2 llama.cpp server."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 10.0,
                 opener: Callable = urllib.request.urlopen,
                 max_concurrency: int = 4):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = opener
        # Refiner instances and the live subtitle worker share this client.
        # Keep their aggregate request count within llama-server's slot count.
        self._request_slots = threading.BoundedSemaphore(max(1, max_concurrency))

    def targets_for(self, source_lang: str) -> tuple[str, ...]:
        return translation_targets(source_lang)

    def healthcheck(self, timeout: float = 1.0) -> bool:
        request = urllib.request.Request(_health_url(self.base_url),
                                         headers={"Accept": "application/json"})
        try:
            with self._opener(request, timeout=timeout) as response:
                return 200 <= getattr(response, "status", 200) < 300
        except (OSError, urllib.error.URLError, TimeoutError):
            return False

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang not in SUPPORTED_LANGS:
            raise ValueError(f"unsupported source language: {source_lang!r}")
        if target_lang not in SUPPORTED_LANGS:
            raise ValueError(f"unsupported target language: {target_lang!r}")
        if source_lang == target_lang:
            raise ValueError("source and target languages must differ")
        stripped = text.strip()
        if not stripped:
            return text

        source_name = LANGUAGE_NAMES[source_lang]
        target_name = LANGUAGE_NAMES[target_lang]
        prompt = (
            f"Translate the following {source_name} text into {target_name}. "
            "Note that you should only output the translated result without "
            "any additional explanation. Preserve every Arabic digit sequence "
            f"exactly as written (for example, 1940 must remain 1940):\n{stripped}"
        )
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(256, max(64, len(stripped) * 3)),
            # Tencent's recommended Hy-MT2-1.8B decoding settings.
            "temperature": 0.7,
            "top_p": 0.6,
            "top_k": 20,
            "repeat_penalty": 1.05,
        }
        request = urllib.request.Request(
            _api_url(self.base_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with self._request_slots:
                with self._opener(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            translated = body["choices"][0]["message"]["content"].strip()
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError,
                KeyError, IndexError, TypeError) as exc:
            raise HyMTError(f"Hy-MT2 request failed: {exc}") from exc
        if not translated:
            raise HyMTError("Hy-MT2 returned an empty translation")
        return translated


class ManagedHyMTServer:
    """Start llama-server only when the configured local endpoint is absent."""

    def __init__(self, client: HyMTClient, model_path: str,
                 binary: str = "llama-server", parallel: int = 4):
        self.client = client
        self.model_path = os.path.abspath(model_path)
        self.binary = binary
        self.parallel = max(1, parallel)
        self.process: subprocess.Popen | None = None

    def start(self, wait_s: float = 60.0) -> "ManagedHyMTServer":
        if self.client.healthcheck():
            return self
        parsed = urllib.parse.urlsplit(self.client.base_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise HyMTError(f"remote Hy-MT2 server is unavailable: {self.client.base_url}")
        if not os.path.isfile(self.model_path):
            raise HyMTError(
                f"Hy-MT2 model not found: {self.model_path}; "
                "run scripts/download_models.py --hymt"
            )
        executable = (self.binary if os.path.isfile(self.binary)
                      else shutil.which(self.binary))
        if not executable:
            raise HyMTError(
                f"llama-server not found: {self.binary!r}; install a current llama.cpp build"
            )
        port = parsed.port or 80
        host = parsed.hostname or "127.0.0.1"
        command = llama_server_command(
            executable, self.model_path, host, port, self.parallel)
        self.process = subprocess.Popen(command)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if self.client.healthcheck():
                return self
            if self.process.poll() is not None:
                raise HyMTError(
                    f"llama-server exited during startup with code {self.process.returncode}"
                )
            time.sleep(0.25)
        self.stop()
        raise HyMTError(f"timed out waiting for Hy-MT2 at {self.client.base_url}")

    @property
    def started_here(self) -> bool:
        return self.process is not None

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None


def ensure_hymt_client(base_url: str, timeout: float, model_path: str,
                       binary: str = "llama-server", parallel: int = 4
                       ) -> tuple[HyMTClient, ManagedHyMTServer]:
    """Return a ready client, starting a local server when necessary."""
    client = HyMTClient(base_url, timeout=timeout, max_concurrency=parallel)
    server = ManagedHyMTServer(client, model_path, binary=binary, parallel=parallel)
    server.start()
    return client, server
