"""Provider-agnostic LLM client. Swap models with one flag or env var.

Every provider below speaks the OpenAI chat-completions protocol, so one
code path covers all of them. Responses are cached on disk keyed by
(model, prompt) -> reruns are free, deterministic, and work offline.

    --provider groq   --model llama-3.3-70b-versatile     (GROQ_API_KEY)
    --provider gemini --model gemini-2.5-flash            (GEMINI_API_KEY)
    --provider anthropic --model claude-haiku-5-5        (ANTHROPIC_API_KEY; official SDK)
    --provider openrouter / openai / together / ollama
    --provider custom  (PROMO_LLM_BASE_URL + PROMO_LLM_API_KEY)
    --provider queue   no API: prompts are written to .cache/queue/, any agent
                       (e.g. a Claude Code Haiku subagent) answers them into
                       <key>.answer.json, rerun picks answers up stage by stage
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
QUEUE = Path(os.environ.get("PROMO_QUEUE_DIR", ROOT / ".cache" / "queue"))

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    # Groq rotates model ids; if the default 404s the error lists what your key can use.
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": ("sdk", "ANTHROPIC_API_KEY", "claude-haiku-5-5"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "ollama": ("http://localhost:11434/v1", "", "qwen2.5:7b"),
    "queue": ("queue", "", "claude-haiku-5-5"),
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
        self.calls = self.cache_hits = self.tokens_in = self.tokens_out = self.pending = 0
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
        self.model = model or os.environ.get("PROMO_LLM_MODEL") or ""
        self._default_model = default_model
        self._auto = not self.model          # no model given -> pick one from the provider
        CACHE.mkdir(parents=True, exist_ok=True)
        if provider == "queue":
            QUEUE.mkdir(parents=True, exist_ok=True)
            return
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise SystemExit("pip install anthropic  (needed for --provider anthropic)")
            self._anthropic = anthropic
            # zero-arg client resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile
            if not any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")) \
                    and not (Path.home() / ".config" / "anthropic").exists():
                raise SystemExit("no Anthropic credentials: set ANTHROPIC_API_KEY (or `ant auth login`), "
                                 "or use --provider queue")
            self.model = self.model or default_model
            self._auto = False
            self.client = anthropic.Anthropic(max_retries=8)
            return
        if not self.key:
            raise SystemExit(f"{keyvar} not set (put it in .env or export it), or use --provider none")
        if self._auto:
            self.model = self.pick_model()

    @property
    def tag(self):
        return f"{self.provider}:{self.model}"

    def _key(self, system, user):
        return hashlib.sha256(json.dumps({"p": "anthropic", "m": self.model, "s": system, "u": user},
                                         sort_keys=True).encode()).hexdigest()[:24]

    def _chat_queue(self, system: str, user: str) -> str:
        key = self._key(system, user)
        f = CACHE / f"{key}.json"
        ans = QUEUE / f"{key}.answer.json"
        if not f.exists() and ans.exists():
            content = ans.read_text().strip()
            f.write_text(json.dumps({"model": self.model + " (agent-answered)", "content": content,
                                     "usage": {"output_chars": len(content)}}))
            (QUEUE / f"{key}.json").unlink(missing_ok=True)
            ans.unlink()
            self.calls += 1
            self.tokens_out += len(content) // 4
            return content
        if f.exists():
            self.cache_hits += 1
            return json.loads(f.read_text())["content"]
        (QUEUE / f"{key}.json").write_text(json.dumps({"key": key, "system": system, "user": user}, indent=1))
        self.pending += 1
        return '{"_pending": true}'

    def _chat_anthropic(self, system: str, user: str) -> str:
        key = self._key(system, user)
        f = CACHE / f"{key}.json"
        if f.exists():
            self.cache_hits += 1
            return json.loads(f.read_text())["content"]
        A = self._anthropic
        try:
            # Haiku 5.5: thinking is adaptive by default; sampling params must stay default (no temperature).
            resp = self.client.messages.create(
                model=self.model, max_tokens=16000,
                system=system + "\nRespond with a single JSON object and nothing else.",
                messages=[{"role": "user", "content": user}])
        except A.AuthenticationError:
            raise SystemExit("Anthropic auth failed: set ANTHROPIC_API_KEY (or run `ant auth login`)")
        except A.NotFoundError:
            raise SystemExit(f"model {self.model} not found for this account")
        except A.BadRequestError as e:
            raise SystemExit(f"Anthropic bad request: {e.message}")
        except A.RateLimitError:
            raise SystemExit("rate limited after retries; rerun later (finished calls are cached)")
        except A.APIStatusError as e:
            raise SystemExit(f"Anthropic API error {e.status_code}: {e.message}")
        except A.APIConnectionError:
            raise SystemExit("network error reaching the Anthropic API")
        if resp.stop_reason == "refusal":
            content = '{"_refusal": true}'
        else:
            content = "".join(b.text for b in resp.content if b.type == "text")
            if resp.stop_reason == "max_tokens" and self.verbose:
                print("  [llm] hit max_tokens; output may be truncated")
        self.calls += 1
        self.tokens_in += resp.usage.input_tokens
        self.tokens_out += resp.usage.output_tokens
        f.write_text(json.dumps({"model": self.model, "content": content,
                                 "usage": {"input_tokens": resp.usage.input_tokens,
                                           "output_tokens": resp.usage.output_tokens}}))
        return content

    def chat(self, system: str, user: str, json_mode: bool = True, temperature: float | None = None) -> str:
        if self.provider == "anthropic":
            return self._chat_anthropic(system, user)
        if self.provider == "queue":
            return self._chat_queue(system, user)
        temp = self.temperature if temperature is None else temperature
        payload = {"model": self.model, "temperature": temp, "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
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
                if e.code in (400, 404) and "model" in body.lower():
                    raise SystemExit(f"model '{self.model}' is not available on {self.provider}.\n"
                                     f"Models your key can use:\n  " + "\n  ".join(self.list_models()) +
                                     f"\nRe-run with --model <id>.")
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

    # names that are not chat models, whatever the provider calls them
    _NOT_CHAT = ("whisper", "tts", "embed", "guard", "moderation", "rerank", "vision-ocr", "distil-whisper",
                 "audio", "image", "safety", "prompt-guard")
    # rough "bigger / newer first" preference, applied to whatever the key can actually use
    _PREFER = ("llama-3.3", "llama-4", "llama-3.1", "70b", "90b", "120b", "mixtral", "qwen", "gemma",
               "instruct", "versatile")

    def pick_model(self) -> str:
        """No --model given: ask the provider what this key can use and choose.

        Providers rotate model ids without notice, so hard-coding a default means
        a 404 one day. Falls back to the documented default if the list call fails.
        """
        ids = [m for m in self.list_models() if not m.startswith("(")]
        chat = [m for m in ids if not any(w in m.lower() for w in self._NOT_CHAT)]
        if not chat:
            return self._default_model
        if self._default_model in chat:
            return self._default_model
        def rank(m):
            low = m.lower()
            return (-sum(1 for i, p in enumerate(self._PREFER) if p in low), len(m))
        best = sorted(chat, key=rank)[0]
        if self.verbose:
            print(f"  [llm] no --model given; using {best} (from {len(chat)} available on {self.provider})")
        return best

    def list_models(self) -> list[str]:
        """Ask the provider which models this key can use (providers rotate ids)."""
        try:
            req = urllib.request.Request(self.base.rstrip("/") + "/models",
                                         headers={"Authorization": f"Bearer {self.key}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return sorted(m["id"] for m in json.loads(r.read()).get("data", []))
        except Exception as e:                                  # noqa: BLE001 - diagnostics only
            return [f"(could not list models: {e})"]

    def chat_json(self, system: str, user: str, **kw) -> dict:
        txt = self.chat(system, user, **kw)
        return parse_json(txt)

    def stats(self):
        return {"model": self.tag, "api_calls": self.calls, "pending": self.pending, "cache_hits": self.cache_hits,
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
