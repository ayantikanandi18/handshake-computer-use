"""
Provider-agnostic LLM client.

The system defaults to a local Ollama model, which means the discovery run costs nothing
and an evaluator can reproduce it without credentials. Swapping to a hosted frontier
model is one environment variable, because the interface here is deliberately tiny:
send messages and a JSON schema, get back a validated object.

That narrowness is the point. Running against a small local model forces the agent
contract to be strict - a closed action grammar, one decision per turn, schema-validated
output with a repair retry. A loop built that way also runs correctly on a large model,
whereas a loop that depends on a large model's tolerance tends to fall apart on anything
smaller. The constraint improved the design rather than compromising it.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_PROVIDER = os.environ.get("CUA_LLM_PROVIDER", "ollama")
DEFAULT_MODEL = os.environ.get("CUA_LLM_MODEL", "qwen2.5:7b-instruct")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> Any:
    """Recover a JSON value from whatever the model actually returned.

    Small models wrap JSON in prose or code fences. Failing the whole run over a
    stray "Here you go:" would be a self-inflicted wound.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"no parseable JSON in: {text[:200]!r}")


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        timeout: float = 600.0,
    ):
        self.provider = provider or DEFAULT_PROVIDER
        self.model = model or DEFAULT_MODEL
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._http.close()

    # --- transport -----------------------------------------------------------

    def complete(self, messages: list[dict[str, str]], schema: dict | None = None) -> str:
        if self.provider == "ollama":
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                # Deterministic decoding. A discovery run that cannot be re-run to the
                # same place is much harder to debug, and nothing here benefits from
                # sampling creativity.
                "options": {"temperature": 0.0, "num_ctx": 8192},
            }
            if schema:
                payload["format"] = schema
            try:
                resp = self._http.post(f"{OLLAMA_URL}/api/chat", json=payload)
            except httpx.ConnectError as exc:
                raise LLMError(
                    f"Cannot reach Ollama at {OLLAMA_URL}. Is `ollama serve` running?"
                ) from exc
            if resp.status_code >= 400:
                raise LLMError(f"{resp.status_code} from Ollama: {resp.text[:300]}")
            return resp.json()["message"]["content"]

        # Any OpenAI-compatible endpoint, including Anthropic via a proxy, Groq, vLLM.
        base = os.environ.get("CUA_LLM_BASE_URL", "").rstrip("/")
        key = os.environ.get("CUA_LLM_API_KEY", "")
        if not base:
            raise LLMError(f"provider '{self.provider}' needs CUA_LLM_BASE_URL")
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.0}
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }
        resp = self._http.post(
            f"{base}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )
        if resp.status_code >= 400:
            raise LLMError(f"{resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]

    # --- typed decisions -----------------------------------------------------

    def decide(self, messages: list[dict[str, str]], model_cls: type[T], repairs: int = 2) -> T:
        """Get one schema-valid decision, re-prompting with the error if it is not.

        The repair loop is what makes a 7B model usable as an agent driver. It is also
        honest engineering for a hosted model: schema violations happen there too, just
        less often.
        """
        convo = list(messages)
        schema = model_cls.model_json_schema()
        last: Exception | None = None
        for attempt in range(repairs + 1):
            raw = self.complete(convo, schema=schema)
            try:
                return model_cls.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                last = exc
                if attempt == repairs:
                    break
                convo = convo + [
                    {"role": "assistant", "content": raw[:1500]},
                    {
                        "role": "user",
                        "content": (
                            f"That did not match the required schema.\nError: {exc}\n"
                            f"Reply with ONLY a JSON object matching this schema:\n"
                            f"{json.dumps(schema)}"
                        ),
                    },
                ]
        raise LLMError(f"model did not produce valid {model_cls.__name__}: {last}")
