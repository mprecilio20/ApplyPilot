"""
Unified LLM client for ApplyPilot.

Auto-detects provider from environment:
  GEMINI_API_KEY  -> Google Gemini (default: gemini-2.0-flash)
  OPENAI_API_KEY  -> OpenAI (default: gpt-4o-mini)
  LLM_URL         -> Local llama.cpp / Ollama compatible endpoint

Set LLM_PROVIDER=claude-cli to route calls through the Claude Code CLI
instead of an HTTP API. This reuses your logged-in Anthropic subscription
(Pro/Max OAuth) for billing -- no API key required. Run `claude login`
once, then set LLM_PROVIDER=claude-cli (default model: sonnet).

LLM_MODEL env var overrides the model name for any provider.
"""

import json
import logging
import os
import shutil
import subprocess
import time

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider() -> tuple[str, str, str]:
    """Return (base_url, model, api_key) based on environment variables.

    Reads env at call time (not module import time) so that load_env() called
    in _bootstrap() is always visible here.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    local_url = os.environ.get("LLM_URL", "")
    model_override = os.environ.get("LLM_MODEL", "")

    if gemini_key and not local_url:
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            model_override or "gemini-2.0-flash",
            gemini_key,
        )

    if openai_key and not local_url:
        return (
            "https://api.openai.com/v1",
            model_override or "gpt-4o-mini",
            openai_key,
        )

    if local_url:
        return (
            local_url.rstrip("/"),
            model_override or "local-model",
            os.environ.get("LLM_API_KEY", ""),
        )

    raise RuntimeError(
        "No LLM provider configured. "
        "Set GEMINI_API_KEY, OPENAI_API_KEY, or LLM_URL in your environment."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds

# Base wait on first 429/503 (doubles each retry, caps at 60s).
# Gemini free tier is 15 RPM = 4s minimum between requests; 10s gives headroom.
_RATE_LIMIT_BASE_WAIT = 10


_GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"


class LLMClient:
    """Thin LLM client supporting OpenAI-compatible and native Gemini endpoints.

    For Gemini keys, starts on the OpenAI-compat layer. On a 403 (which
    happens with preview/experimental models not exposed via compat), it
    automatically switches to the native generateContent API and stays there
    for the lifetime of the process.
    """

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=_TIMEOUT)
        # True once we've confirmed the native Gemini API works for this model
        self._use_native_gemini: bool = False
        self._is_gemini: bool = base_url.startswith(_GEMINI_COMPAT_BASE)

    # -- Native Gemini API --------------------------------------------------

    def _chat_native_gemini(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the native Gemini generateContent API.

        Used automatically when the OpenAI-compat endpoint returns 403,
        which happens for preview/experimental models not exposed via compat.

        Converts OpenAI-style messages to Gemini's contents/systemInstruction
        format transparently.
        """
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                contents.append({"role": "model", "parts": [{"text": text}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"{_GEMINI_NATIVE_BASE}/models/{self.model}:generateContent"
        resp = self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # -- OpenAI-compat API --------------------------------------------------

    def _chat_compat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the OpenAI-compatible endpoint."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

        # 403 on Gemini compat = model not available on compat layer.
        # Raise a specific sentinel so chat() can switch to native API.
        if resp.status_code == 403 and self._is_gemini:
            raise _GeminiCompatForbidden(resp)

        return self._handle_compat_response(resp)

    @staticmethod
    def _handle_compat_response(resp: httpx.Response) -> str:
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message text."""
        # Qwen3 optimization: prepend /no_think to skip chain-of-thought
        # reasoning, saving tokens on structured extraction tasks.
        if "qwen" in self.model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        for attempt in range(_MAX_RETRIES):
            try:
                # Route to native Gemini if we've already confirmed it's needed
                if self._use_native_gemini:
                    return self._chat_native_gemini(messages, temperature, max_tokens)

                return self._chat_compat(messages, temperature, max_tokens)

            except _GeminiCompatForbidden as exc:
                # Model not available on OpenAI-compat layer — switch to native.
                log.warning(
                    "Gemini compat endpoint returned 403 for model '%s'. "
                    "Switching to native generateContent API. "
                    "(Preview/experimental models are often compat-only on native.)",
                    self.model,
                )
                self._use_native_gemini = True
                # Retry immediately with native — don't count as a rate-limit wait
                try:
                    return self._chat_native_gemini(messages, temperature, max_tokens)
                except httpx.HTTPStatusError as native_exc:
                    raise RuntimeError(
                        f"Both Gemini endpoints failed. Compat: 403 Forbidden. "
                        f"Native: {native_exc.response.status_code} — "
                        f"{native_exc.response.text[:200]}"
                    ) from native_exc

            except httpx.HTTPStatusError as exc:
                resp = exc.response
                if resp.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                    # Respect Retry-After header if provided (Gemini sends this).
                    retry_after = (
                        resp.headers.get("Retry-After")
                        or resp.headers.get("X-RateLimit-Reset-Requests")
                    )
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except (ValueError, TypeError):
                            wait = _RATE_LIMIT_BASE_WAIT * (2 ** attempt)
                    else:
                        wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)

                    log.warning(
                        "LLM rate limited (HTTP %s). Waiting %ds before retry %d/%d. "
                        "Tip: Gemini free tier = 15 RPM. Consider a paid account "
                        "or switching to a local model.",
                        resp.status_code, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning(
                        "LLM request timed out, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError("LLM request failed after all retries")

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


class _GeminiCompatForbidden(Exception):
    """Sentinel: Gemini OpenAI-compat returned 403. Switch to native API."""
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Gemini compat 403: {response.text[:200]}")


# ---------------------------------------------------------------------------
# Claude Code CLI client (subscription auth, no API key)
# ---------------------------------------------------------------------------

_CLAUDE_CLI_TIMEOUT = 300  # seconds; CLI can be slow on cold start
_CLAUDE_CLI_MAX_RETRIES = 3
_CLAUDE_CLI_BASE_WAIT = 5


class ClaudeCLIClient:
    """LLM client that shells out to the `claude` CLI.

    This reuses the user's logged-in Anthropic subscription (Pro/Max OAuth)
    for billing instead of an API key. It implements the same chat()/ask()
    interface as LLMClient so it's a drop-in replacement for scoring,
    tailoring, cover letters, and extraction.

    Invocation mirrors the pattern already used by the auto-apply launcher:
    the prompt is piped via stdin and the model is selected with --model.
    Output is requested as JSON so we can reliably pull the result text and
    detect errors.

    Notes / limitations:
      - The CLI has no temperature or max_tokens flags, so those kwargs are
        accepted for interface compatibility but ignored.
      - ANTHROPIC_API_KEY is stripped from the subprocess environment so the
        CLI always uses the interactive subscription login, never an API key.
    """

    def __init__(self, model: str, binary: str = "claude") -> None:
        self.model = model
        self.binary = binary

    # -- prompt assembly ----------------------------------------------------

    @staticmethod
    def _split_messages(messages: list[dict]) -> tuple[str, str]:
        """Split OpenAI-style messages into (system_prompt, user_prompt).

        System messages are concatenated and returned separately (passed via
        --append-system-prompt). Remaining user/assistant turns are flattened
        into a single stdin prompt, labelled by role when more than one turn
        is present.
        """
        system_parts: list[str] = []
        convo: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            else:
                convo.append(msg)

        system_prompt = "\n\n".join(p for p in system_parts if p).strip()

        if len(convo) == 1:
            user_prompt = convo[0].get("content", "")
        else:
            lines: list[str] = []
            for msg in convo:
                role = msg.get("role", "user").upper()
                lines.append(f"{role}: {msg.get('content', '')}")
            user_prompt = "\n\n".join(lines)

        return system_prompt, user_prompt

    # -- subprocess env -----------------------------------------------------

    @staticmethod
    def _clean_env() -> dict:
        env = os.environ.copy()
        # Force subscription (OAuth) auth -- never fall back to API-key billing.
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        # Avoid nested-session detection when running inside Claude Code itself.
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        return env

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat request via the claude CLI and return the response text.

        temperature and max_tokens are ignored (the CLI exposes no such flags)
        but accepted for interface parity with LLMClient.
        """
        system_prompt, user_prompt = self._split_messages(messages)

        cmd = [
            self.binary,
            "--model", self.model,
            "-p",
            "--output-format", "json",
            "--no-session-persistence",
        ]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        cmd += ["-"]  # read the user prompt from stdin

        env = self._clean_env()

        last_err: Exception | None = None
        for attempt in range(_CLAUDE_CLI_MAX_RETRIES):
            try:
                proc = subprocess.run(
                    cmd,
                    input=user_prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    timeout=_CLAUDE_CLI_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                last_err = exc
                if attempt < _CLAUDE_CLI_MAX_RETRIES - 1:
                    wait = _CLAUDE_CLI_BASE_WAIT * (2 ** attempt)
                    log.warning("claude CLI timed out, retry %d/%d in %ds",
                                attempt + 1, _CLAUDE_CLI_MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"claude CLI timed out after {_CLAUDE_CLI_TIMEOUT}s"
                ) from exc

            if proc.returncode != 0:
                stderr = (proc.stderr or proc.stdout or "").strip()
                last_err = RuntimeError(f"claude CLI exited {proc.returncode}: {stderr[:300]}")
                # Retry on transient failures (rate limit / overloaded).
                if attempt < _CLAUDE_CLI_MAX_RETRIES - 1 and self._is_transient(stderr):
                    wait = _CLAUDE_CLI_BASE_WAIT * (2 ** attempt)
                    log.warning("claude CLI transient error, retry %d/%d in %ds: %s",
                                attempt + 1, _CLAUDE_CLI_MAX_RETRIES, wait, stderr[:150])
                    time.sleep(wait)
                    continue
                raise last_err

            return self._parse_output(proc.stdout)

        raise RuntimeError(f"claude CLI failed after all retries: {last_err}")

    @staticmethod
    def _is_transient(text: str) -> bool:
        t = text.lower()
        return any(k in t for k in ("rate limit", "overloaded", "429", "503", "timeout", "temporarily"))

    @staticmethod
    def _parse_output(stdout: str) -> str:
        """Extract the assistant text from `claude -p --output-format json`."""
        raw = (stdout or "").strip()
        if not raw:
            raise RuntimeError("claude CLI returned empty output")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fall back to treating output as plain text.
            return raw
        if isinstance(data, dict):
            if data.get("is_error"):
                raise RuntimeError(f"claude CLI reported error: {str(data.get('result'))[:300]}")
            result = data.get("result")
            if isinstance(result, str):
                return result
        return raw

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:  # noqa: D401 - interface parity, nothing to close
        """No-op: the CLI client holds no persistent connection."""


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: "LLMClient | ClaudeCLIClient | None" = None


def _normalize_provider(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _make_claude_cli_client() -> ClaudeCLIClient:
    binary = shutil.which("claude")
    if not binary:
        raise RuntimeError(
            "LLM_PROVIDER=claude-cli but the 'claude' CLI was not found on PATH. "
            "Install Claude Code from https://claude.ai/code and run `claude login`."
        )
    model = os.environ.get("LLM_MODEL", "") or "sonnet"
    return ClaudeCLIClient(model=model, binary=binary)


def get_client() -> "LLMClient | ClaudeCLIClient":
    """Return (or create) the module-level LLM client singleton.

    Honors LLM_PROVIDER=claude-cli to use the Claude Code CLI (subscription
    auth). Otherwise auto-detects an HTTP provider from the environment.
    """
    global _instance
    if _instance is None:
        provider = _normalize_provider(os.environ.get("LLM_PROVIDER", ""))
        if provider in ("claude-cli", "claude", "claude-code", "anthropic-cli"):
            _instance = _make_claude_cli_client()
            log.info("LLM provider: claude CLI (subscription)  model: %s", _instance.model)
        else:
            base_url, model, api_key = _detect_provider()
            log.info("LLM provider: %s  model: %s", base_url, model)
            _instance = LLMClient(base_url, model, api_key)
    return _instance
