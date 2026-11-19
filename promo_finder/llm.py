"""Provider-agnostic LLM client. Swap models with one flag or env var.

Every provider below speaks the OpenAI chat-completions protocol, so one
code path covers all of them. Responses are cached on disk keyed by
(model, prompt) -> reruns are free, deterministic, and work offline.

    --provider groq   --model llama-3.3-70b-versatile     (GROQ_API_KEY)
    --provider gemini --model gemini-2.5-flash            (GEMINI_API_KEY)
    --provider openrouter / openai / anthropic / together / ollama
    --provider custom  (PROMO_LLM_BASE_URL + PROMO_LLM_API_KEY)
    --provider none    rules only, no network
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "llm"

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "claude-haiku-5-5"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "ollama": ("http://localhost:11434/v1", "", "qwen2.5:7b"),
    "custom": (os.environ.get("PROMO_LLM_BASE_URL", ""), "PROMO_LLM_API_KEY", os.environ.get("PROMO_LLM_MODEL", "")),
}


def _load_dotenv():
    f = ROOT / ".env"
    if f.exists():
        for ln in f.read_text().splitlines():
            if "=" in ln and not ln.strip().startswith("#"):
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


class LLM:
    def __init__(self, provider: str = "none", model: str | None = None, temperature: float = 0.2,
                 max_tokens: int = 900, verbose: bool = True):
        _load_dotenv()
        provider = (provider or "none").lower()
        self.provider = provider
        self.enabled = provider != "none"
        self.calls = self.cache_hits = self.tokens_in = self.tokens_out = 0
        self.verbose = verbose
        self.temperature, self.max_tokens = temperature, max_tokens
        if not self.enabled:
            self.model = "rules-only"
            return
        if provider not in PROVIDERS:
            raise SystemExit(f"unknown provider {provider}; choose from {', '.join(PROVIDERS)} or none")
        base, keyvar, default_model = PROVIDERS[provider]
        self.base = os.environ.get("PROMO_LLM_BASE_URL", base) if provider == "custom" else base
        self.key = os.environ.get(keyvar, "") if keyvar else "ollama"
        self.model = model or os.environ.get("PROMO_LLM_MODEL") or default_model
        if not self.key:
            raise SystemExit(f"{keyvar} not set (put it in .env or export it), or use --provider none")
        CACHE.mkdir(parents=True, exist_ok=True)

    @property
    def tag(self):
        return f"{self.provider}:{self.model}"

    def chat(self, system: str, user: str, json_mode: bool = True, temperature: float | None = None) -> str:
        temp = self.temperature if temperature is None else temperature
        payload = {"model": self.model, "temperature": temp, "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode and self.provider not in ("anthropic",):
            payload["response_format"] = {"type": "json_object"}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
        f = CACHE / f"{key}.json"
        if f.exists():
            self.cache_hits += 1
            return json.loads(f.read_text())["content"]
        delay = 4.0
        for attempt in range(8):
            req = urllib.request.Request(self.base.rstrip("/") + "/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.key}",
                                                  "x-api-key": self.key, "anthropic-version": "2023-06-01",
                                                  "User-Agent": "promo-finder/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:400]
                if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                    ra = e.headers.get("retry-after")
                    wait = float(ra) if ra and ra.replace(".", "").isdigit() else delay
                    m = re.search(r"try again in ([\d.]+)(m?s)", body)
                    if m:
                        wait = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) + 0.5
                    if self.verbose:
                        print(f"  [llm] {e.code}, waiting {wait:.1f}s")
                    time.sleep(min(wait, 90))
                    delay = min(delay * 2, 60)
                    continue
                if e.code == 400 and "response_format" in payload:
                    payload.pop("response_format")       # some models reject json mode
                    continue
                raise SystemExit(f"LLM error {e.code}: {body}")
            except urllib.error.URLError as e:
                if attempt < 7:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise SystemExit(f"LLM network error: {e}")
        content = data["choices"][0]["message"]["content"] or ""
        u = data.get("usage") or {}
        self.calls += 1
        self.tokens_in += u.get("prompt_tokens", 0)
        self.tokens_out += u.get("completion_tokens", 0)
        f.write_text(json.dumps({"model": self.model, "content": content, "usage": u}))
        return content

    def chat_json(self, system: str, user: str, **kw) -> dict:
        txt = self.chat(system, user, **kw)
        return parse_json(txt)

    def stats(self):
        return {"model": self.tag, "api_calls": self.calls, "cache_hits": self.cache_hits,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}


def parse_json(txt: str) -> dict:
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    if m:
        txt = m.group(1)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        a, b = txt.find("{"), txt.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(txt[a:b + 1])
            except json.JSONDecodeError:
                pass
    return {"_parse_error": txt[:300]}
