"""Provider-neutral LLM automation for paper reproductions.

The browser never receives stored API keys.  On Windows, optional persistent
keys are protected with DPAPI for the current OS user.  Model output is treated
as untrusted input and normalized before it reaches the dashboard forms.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import html.parser
import io
import ipaddress
import json
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import jsonify, request, send_from_directory


MAX_PAPER_BYTES = 25 * 1024 * 1024
MAX_PAPER_CHARS = 180_000
MAX_REPORT_RUNS = 200
PROVIDER_TIMEOUT_SECONDS = 240

PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com",
        "model": "gpt-5",
        "environment_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "label": "Anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-20250514",
        "environment_key": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com",
        "model": "gemini-2.5-flash",
        "environment_key": "GEMINI_API_KEY",
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "environment_key": "GROQ_API_KEY",
    },
    "openai_compatible": {
        "label": "OpenAI-compatible / local",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "",
        "environment_key": "OPENAI_COMPATIBLE_API_KEY",
    },
}

ANALYSIS_PLAN_KEYS = {
    "representation", "decoder_geometry", "precision_polarization",
    "tangent_metrics", "topology_folding", "factor_probes",
    "causal_interference", "latent_explorer", "factor_traversal",
}
DATASET_SOURCES = {
    "paper_linear", "paper_nonlinear", "synthetic_linear",
    "synthetic_nonlinear", "dsprites", "shapes3d", "mnist",
    "fashion_mnist", "celeba", "uploaded",
}
MODEL_TYPES = {
    "vae", "beta_vae", "beta_vae_full_cov", "ae", "autoencoder",
    "random_decoder", "conv_beta_vae", "factor_vae", "beta_tcvae",
    "slow_vae", "pcl", "weakly_supervised_gan", "custom_vae", "custom_ae",
    "cnn_autoencoder", "cnn_vae", "transformer_autoencoder", "transformer_vae",
    "rnn_autoencoder", "rnn_vae", "mamba_autoencoder", "mamba_vae",
}


def _nullable(kind):
    return {"type": [kind, "null"]}


PROTOCOL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "paper", "dataset", "model", "training", "analysis_plan",
        "seed_plan", "reproduction_limits", "unresolved", "evidence",
    ],
    "properties": {
        "paper": {
            "type": "object", "additionalProperties": False,
            "required": ["title", "citation", "url", "summary", "hypothesis"],
            "properties": {
                "title": {"type": "string"}, "citation": {"type": "string"},
                "url": {"type": "string"}, "summary": {"type": "string"},
                "hypothesis": {"type": "string"},
            },
        },
        "dataset": {
            "type": "object", "additionalProperties": False,
            "required": [
                "source", "dataset_type", "factor_types", "factor_count",
                "latent_dim", "sparsity", "dataset_size", "observation_dim",
                "noise", "train_split", "validation_split", "seed",
                "mixing", "degeneracy", "notes",
            ],
            "properties": {
                "source": {"type": "string"}, "dataset_type": {"type": "string"},
                "factor_types": {"type": "array", "items": {"type": "string"}},
                "factor_count": _nullable("integer"), "latent_dim": _nullable("integer"),
                "sparsity": _nullable("number"), "dataset_size": _nullable("integer"),
                "observation_dim": _nullable("integer"), "noise": _nullable("number"),
                "train_split": _nullable("number"), "validation_split": _nullable("number"),
                "seed": _nullable("integer"), "mixing": {"type": "string"},
                "degeneracy": _nullable("number"), "notes": {"type": "string"},
            },
        },
        "model": {
            "type": "object", "additionalProperties": False,
            "required": ["model_type", "latent_dim", "beta", "custom_model", "notes"],
            "properties": {
                "model_type": {"type": "string"}, "latent_dim": _nullable("integer"),
                # A JSON string keeps the provider schema strict while allowing the
                # dashboard to validate a bounded declarative architecture itself.
                "beta": _nullable("number"), "custom_model": {"type": ["string", "null"]},
                "notes": {"type": "string"},
            },
        },
        "training": {
            "type": "object", "additionalProperties": False,
            "required": [
                "epochs", "training_steps", "learning_rate", "batch_size",
                "optimizer", "adam_beta1", "adam_beta2", "optimizer_epsilon",
                "weight_decay", "reconstruction_loss", "hidden_dim",
                "encoder_depth", "decoder_depth", "encoder_activation",
                "decoder_activation", "bias", "tc_weight", "seed",
                "paper_reproduction", "notes",
            ],
            "properties": {
                "epochs": _nullable("integer"), "training_steps": _nullable("integer"),
                "learning_rate": _nullable("number"), "batch_size": _nullable("integer"),
                "optimizer": {"type": "string"}, "adam_beta1": _nullable("number"),
                "adam_beta2": _nullable("number"), "optimizer_epsilon": _nullable("number"),
                "weight_decay": _nullable("number"), "reconstruction_loss": {"type": "string"},
                "hidden_dim": _nullable("integer"), "encoder_depth": _nullable("integer"),
                "decoder_depth": _nullable("integer"), "encoder_activation": {"type": "string"},
                "decoder_activation": {"type": "string"}, "bias": _nullable("boolean"),
                "tc_weight": _nullable("number"), "seed": _nullable("integer"),
                "paper_reproduction": {"type": "boolean"}, "notes": {"type": "string"},
            },
        },
        "analysis_plan": {"type": "array", "items": {"type": "string"}},
        "seed_plan": {
            "type": "object", "additionalProperties": False,
            "required": ["training_seeds", "metric_seed", "aggregation", "notes"],
            "properties": {
                "training_seeds": {"type": "array", "items": {"type": "integer"}},
                "metric_seed": _nullable("integer"), "aggregation": {"type": "string"},
                "notes": {"type": "string"},
            },
        },
        "reproduction_limits": {"type": "array", "items": {"type": "string"}},
        "unresolved": {"type": "array", "items": {"type": "string"}},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["field", "value", "source_location", "confidence", "status"],
                "properties": {
                    "field": {"type": "string"}, "value": {"type": "string"},
                    "source_location": {"type": "string"}, "confidence": {"type": "number"},
                    "status": {"type": "string"},
                },
            },
        },
    },
}

REPORT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": [
        "title", "abstract", "introduction", "state_of_the_art", "theory",
        "methodology", "results", "discussion", "limitations", "perspectives",
        "conclusion", "key_findings", "references", "data_caveats",
    ],
    "properties": {
        key: {"type": "string"} for key in [
            "title", "abstract", "introduction", "state_of_the_art", "theory",
            "methodology", "results", "discussion", "limitations", "perspectives",
            "conclusion",
        ]
    } | {
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "data_caveats": {"type": "array", "items": {"type": "string"}},
        "references": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["citation", "url"],
                "properties": {"citation": {"type": "string"}, "url": {"type": "string"}},
            },
        },
    },
}


class LLMError(RuntimeError):
    pass


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _windows_dpapi():
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    blob_pointer = ctypes.POINTER(_DataBlob)
    crypt32.CryptProtectData.argtypes = [
        blob_pointer, ctypes.c_wchar_p, blob_pointer, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_ulong, blob_pointer,
    ]
    crypt32.CryptProtectData.restype = ctypes.c_int
    crypt32.CryptUnprotectData.argtypes = [
        blob_pointer, ctypes.POINTER(ctypes.c_wchar_p), blob_pointer,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, blob_pointer,
    ]
    crypt32.CryptUnprotectData.restype = ctypes.c_int
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    return crypt32, kernel32


def _dpapi_protect(value: str) -> str:
    if os.name != "nt":
        raise LLMError("Persistent key protection is available only on Windows; use session-only storage or an environment variable.")
    raw = value.encode("utf-8")
    buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    destination = _DataBlob()
    crypt32, kernel32 = _windows_dpapi()
    ctypes.set_last_error(0)
    ok = crypt32.CryptProtectData(
        ctypes.byref(source), "Interpretability LLM key", None, None, None,
        0x01, ctypes.byref(destination),
    )
    if not ok:
        error = ctypes.WinError(ctypes.get_last_error())
        raise LLMError(
            "Windows DPAPI is unavailable for this process/profile; use session-only key storage or a provider environment variable"
        ) from error
    try:
        protected = ctypes.string_at(destination.pbData, destination.cbData)
        return base64.b64encode(protected).decode("ascii")
    finally:
        kernel32.LocalFree(destination.pbData)


def _dpapi_unprotect(value: str) -> str:
    raw = base64.b64decode(value)
    buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    destination = _DataBlob()
    crypt32, kernel32 = _windows_dpapi()
    ctypes.set_last_error(0)
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 0x01,
        ctypes.byref(destination),
    )
    if not ok:
        error = ctypes.WinError(ctypes.get_last_error())
        raise LLMError("The saved key could not be decrypted by Windows DPAPI for this user") from error
    try:
        return ctypes.string_at(destination.pbData, destination.cbData).decode("utf-8")
    finally:
        kernel32.LocalFree(destination.pbData)


_DPAPI_CAPABILITY = None


def _dpapi_capability() -> tuple[bool, str]:
    global _DPAPI_CAPABILITY
    if _DPAPI_CAPABILITY is not None:
        return _DPAPI_CAPABILITY
    if os.name != "nt":
        _DPAPI_CAPABILITY = (False, "DPAPI is available only on Windows")
        return _DPAPI_CAPABILITY
    try:
        probe = _dpapi_protect("interpretability-dashboard-capability-check")
        if _dpapi_unprotect(probe) != "interpretability-dashboard-capability-check":
            raise LLMError("DPAPI round-trip verification failed")
        _DPAPI_CAPABILITY = (True, "Windows DPAPI current-user protection")
    except Exception as exc:
        _DPAPI_CAPABILITY = (False, str(exc))
    return _DPAPI_CAPABILITY


class ProviderStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.session_keys: dict[str, str] = {}
        self.data = {"active_provider": "openai", "providers": {}}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.data["active_provider"] = str(raw.get("active_provider", "openai"))
                self.data["providers"] = raw.get("providers", {}) if isinstance(raw.get("providers"), dict) else {}
        except Exception:
            self.data = {"active_provider": "openai", "providers": {}}

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def _provider_config(self, provider: str) -> dict:
        defaults = PROVIDERS[provider]
        saved = self.data["providers"].get(provider, {})
        candidate_base_url = str(saved.get("base_url") or defaults["base_url"])
        try:
            base_url = _validate_provider_base_url(provider, candidate_base_url)
        except LLMError:
            # Fail closed for settings written by an older dashboard version.
            # Official providers revert to their fixed origin; the custom
            # provider reverts to the safe loopback default.
            base_url = defaults["base_url"]
        return {
            "provider": provider,
            "model": str(saved.get("model") or defaults["model"]),
            "base_url": base_url,
        }

    def get_key(self, provider: str) -> tuple[str, str]:
        with self.lock:
            if self.session_keys.get(provider):
                return self.session_keys[provider], "session"
            saved = self.data["providers"].get(provider, {})
            if saved.get("protected_key"):
                try:
                    return _dpapi_unprotect(saved["protected_key"]), "protected-local"
                except Exception as exc:
                    raise LLMError(f"The saved {provider} key could not be decrypted for this Windows user: {exc}") from exc
            environment_key = PROVIDERS[provider]["environment_key"]
            if os.environ.get(environment_key):
                return os.environ[environment_key], f"environment:{environment_key}"
            if provider == "openai_compatible":
                return "", "not-required"
            raise LLMError(f"No API key is configured for {PROVIDERS[provider]['label']}.")

    def public(self) -> dict:
        with self.lock:
            providers = {}
            for provider, defaults in PROVIDERS.items():
                config = self._provider_config(provider)
                try:
                    key, source = self.get_key(provider)
                    has_key = bool(key) or provider == "openai_compatible"
                except LLMError:
                    source, has_key = "missing", False
                providers[provider] = {
                    **config, "label": defaults["label"], "has_key": has_key,
                    "key_source": source,
                }
            active = self.data.get("active_provider", "openai")
            if active not in PROVIDERS:
                active = "openai"
            persistence_available, persistence = _dpapi_capability()
            return {
                "active_provider": active, "providers": providers,
                "persistence_available": persistence_available,
                "persistence": persistence if persistence_available else f"session/environment only ({persistence})",
            }

    def update(self, provider: str, model: str, base_url: str, api_key: str, persist: bool):
        if provider not in PROVIDERS:
            raise LLMError("Unknown LLM provider")
        base_url = _validate_provider_base_url(provider, base_url)
        model = model.strip()
        if not model:
            raise LLMError("Model name is required")
        with self.lock:
            saved = dict(self.data["providers"].get(provider, {}))
            saved.update({"model": model[:160], "base_url": base_url})
            if api_key:
                if persist:
                    available, reason = _dpapi_capability()
                    if not available:
                        raise LLMError(reason)
                    saved["protected_key"] = _dpapi_protect(api_key)
                    self.session_keys.pop(provider, None)
                else:
                    self.session_keys[provider] = api_key
                    saved.pop("protected_key", None)
            self.data["providers"][provider] = saved
            self.data["active_provider"] = provider
            self._save()
        return self.public()

    def clear_key(self, provider: str):
        if provider not in PROVIDERS:
            raise LLMError("Unknown LLM provider")
        with self.lock:
            self.session_keys.pop(provider, None)
            saved = dict(self.data["providers"].get(provider, {}))
            saved.pop("protected_key", None)
            self.data["providers"][provider] = saved
            self._save()

    def resolve(self, provider: str | None, model: str | None = None) -> dict:
        provider = provider or self.data.get("active_provider", "openai")
        if provider not in PROVIDERS:
            raise LLMError("Unknown LLM provider")
        config = self._provider_config(provider)
        if model and model.strip():
            config["model"] = model.strip()[:160]
        key, source = self.get_key(provider)
        config.update({"api_key": key, "key_source": source})
        return config


def _validate_provider_base_url(provider: str, base_url: str) -> str:
    """Return a canonical provider URL or reject an unsafe destination."""
    if provider not in PROVIDERS:
        raise LLMError("Unknown LLM provider")
    base_url = str(base_url or "").strip().rstrip("/")
    if not base_url or len(base_url) > 500 or any(char.isspace() or ord(char) < 32 for char in base_url):
        raise LLMError("Provider base URL is invalid")
    if "\\" in base_url:
        raise LLMError("Provider base URL must not contain backslashes")
    try:
        parsed = urllib.parse.urlparse(base_url)
        port = parsed.port
    except ValueError as exc:
        raise LLMError("Provider base URL contains an invalid port") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise LLMError("Provider base URL must be an absolute http or https URL")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise LLMError("Provider base URL must not contain credentials")
    if parsed.query or parsed.fragment or parsed.params:
        raise LLMError("Provider base URL must not contain parameters, a query, or a fragment")
    if provider != "openai_compatible":
        expected = PROVIDERS[provider]["base_url"]
        if base_url != expected:
            raise LLMError(f"{PROVIDERS[provider]['label']} is locked to its official endpoint: {expected}")
        return expected
    if parsed.scheme == "http":
        hostname = parsed.hostname.lower()
        loopback = hostname == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                loopback = False
        if not loopback:
            raise LLMError("Plain HTTP is allowed only for a loopback OpenAI-compatible provider; use HTTPS for remote endpoints")
    # Accessing parsed.port above validates its range. Preserve custom paths
    # such as /v1, while using the normalized scheme/host spelling supplied.
    return base_url


class _NoProviderRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_json(url: str, headers: dict, payload: dict, timeout=PROVIDER_TIMEOUT_SECONDS) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST")
    opener = urllib.request.build_opener(_NoProviderRedirectHandler())
    try:
        with opener.open(req, timeout=timeout) as response:
            raw = response.read(12 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        detail = exc.read(8192).decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            detail = parsed.get("error", {}).get("message") or parsed.get("message") or detail
        except Exception:
            pass
        raise LLMError(f"Provider returned HTTP {exc.code}: {str(detail)[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Could not reach the provider: {exc.reason}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise LLMError("Provider returned a non-JSON response") from exc


def _extract_json_text(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("The model did not return a JSON object")
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise LLMError("The model response must be a JSON object")
    return value


def _gemini_schema(value):
    """Translate nullable JSON Schema unions to Gemini's Schema dialect."""
    if isinstance(value, list):
        return [_gemini_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        # Gemini's responseSchema accepts its OpenAPI-style Schema subset, not
        # JSON Schema's object-closure keywords. Leaving additionalProperties
        # in even a tiny connection-test schema can produce HTTP 400 before
        # the model is invoked.
        if key in {"additionalProperties", "unevaluatedProperties", "$schema", "$defs"}:
            continue
        if key == "type" and isinstance(item, list) and "null" in item:
            concrete = [entry for entry in item if entry != "null"]
            result["type"] = concrete[0] if len(concrete) == 1 else concrete
            result["nullable"] = True
        else:
            result[key] = _gemini_schema(item)
    return result


def call_provider(config: dict, system_prompt: str, user_prompt: str, schema: dict, schema_name: str, max_output_tokens=12000) -> tuple[dict, dict]:
    provider, model, base_url, api_key = config["provider"], config["model"], config["base_url"], config["api_key"]
    if provider == "openai":
        payload = {
            "model": model, "instructions": system_prompt, "input": user_prompt,
            "store": False, "max_output_tokens": max_output_tokens,
            "text": {"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}},
        }
        data = _http_json(f"{base_url}/v1/responses", {"Authorization": f"Bearer {api_key}"}, payload)
        text = data.get("output_text", "")
        if not text:
            parts = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"} and content.get("text"):
                        parts.append(content["text"])
            text = "\n".join(parts)
        return _extract_json_text(text), data.get("usage", {})
    if provider == "anthropic":
        tool_name = f"submit_{schema_name}"
        payload = {
            "model": model, "max_tokens": max_output_tokens, "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "tools": [{"name": tool_name, "description": "Submit the validated structured result.", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        data = _http_json(f"{base_url}/v1/messages", headers, payload)
        for item in data.get("content", []):
            if item.get("type") == "tool_use" and item.get("name") == tool_name:
                if isinstance(item.get("input"), dict):
                    return item["input"], data.get("usage", {})
        raise LLMError("Anthropic did not return the required structured tool result")
    if provider == "gemini":
        quoted_model = urllib.parse.quote(model, safe="-._")
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json", "responseSchema": _gemini_schema(schema),
                "maxOutputTokens": max_output_tokens, "temperature": 0.1,
            },
        }
        data = _http_json(
            f"{base_url}/v1beta/models/{quoted_model}:generateContent",
            {"x-goog-api-key": api_key}, payload,
        )
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "\n".join(str(part.get("text", "")) for part in parts)
        return _extract_json_text(text), data.get("usageMetadata", {})
    json_system_prompt = system_prompt
    if provider == "groq":
        json_system_prompt += "\nReturn only a valid JSON object matching the requested structure."
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": json_system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": 0.1, "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = _http_json(f"{base_url}/chat/completions", headers, payload)
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        label = PROVIDERS.get(provider, {}).get("label", "OpenAI-compatible provider")
        raise LLMError(f"{label} returned an unsupported response shape") from exc
    return _extract_json_text(text), data.get("usage", {})


class _PlainTextHTMLParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self):
        value = "".join(self.parts)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n\s*\n\s*\n+", "\n\n", value)
        return value.strip()


def _host_is_public(hostname: str) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or not _host_is_public(parsed.hostname):
            raise LLMError("Paper URL redirected to a non-public address")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_public_document(url: str) -> tuple[bytes, str, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LLMError("Paper URL must be an absolute http or https URL")
    # An arXiv abstract page does not contain the methods/appendix. Resolve it
    # to the corresponding PDF so a pasted canonical paper URL behaves as users
    # expect without scraping unrelated page chrome.
    if parsed.hostname.lower() in {"arxiv.org", "www.arxiv.org"} and parsed.path.startswith("/abs/"):
        identifier = parsed.path.removeprefix("/abs/").strip("/")
        if identifier:
            parsed = parsed._replace(path=f"/pdf/{identifier}", query="", fragment="")
            url = urllib.parse.urlunparse(parsed)
    if not _host_is_public(parsed.hostname):
        raise LLMError("Paper URL must resolve to a public address")
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    req = urllib.request.Request(url, headers={"User-Agent": "InterpretabilityDashboard/1.0"})
    try:
        with opener.open(req, timeout=60) as response:
            content_type = response.headers.get_content_type()
            data = response.read(MAX_PAPER_BYTES + 1)
            final_url = response.geturl()
    except (urllib.error.URLError, LLMError) as exc:
        raise LLMError(f"Could not download the paper: {exc}") from exc
    if len(data) > MAX_PAPER_BYTES:
        raise LLMError("Paper download exceeds the 25 MB limit")
    return data, content_type, final_url


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise LLMError("PDF extraction requires pypdf; install dashboard requirements first") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        raise LLMError(f"The PDF could not be parsed: {exc}") from exc


def extract_paper(*, uploaded=None, text="", url="") -> tuple[str, dict]:
    source = {"kind": "text", "name": "pasted paper text", "url": "", "sha256": ""}
    content = text.strip()
    if uploaded and uploaded.filename:
        data = uploaded.stream.read(MAX_PAPER_BYTES + 1)
        if len(data) > MAX_PAPER_BYTES:
            raise LLMError("Uploaded paper exceeds the 25 MB limit")
        suffix = Path(uploaded.filename).suffix.lower()
        content = _pdf_text(data) if suffix == ".pdf" or uploaded.mimetype == "application/pdf" else data.decode("utf-8", errors="replace")
        source = {"kind": "upload", "name": Path(uploaded.filename).name[:240], "url": "", "sha256": hashlib.sha256(data).hexdigest()}
    elif url.strip():
        data, content_type, final_url = _download_public_document(url.strip())
        content = _pdf_text(data) if content_type == "application/pdf" or data.startswith(b"%PDF") else ""
        if not content:
            parser = _PlainTextHTMLParser()
            parser.feed(data.decode("utf-8", errors="replace"))
            content = parser.text()
        source = {"kind": "url", "name": final_url.rsplit("/", 1)[-1][:240], "url": final_url, "sha256": hashlib.sha256(data).hexdigest()}
    if len(content.strip()) < 500:
        raise LLMError("Provide a readable PDF, URL, or at least 500 characters of paper text")
    original_chars = len(content)
    if original_chars > MAX_PAPER_CHARS:
        head = content[: int(MAX_PAPER_CHARS * 0.78)]
        tail = content[-int(MAX_PAPER_CHARS * 0.22):]
        content = head + "\n\n[... middle truncated by dashboard ...]\n\n" + tail
    source.update({"characters_used": len(content), "characters_original": original_chars, "truncated": original_chars > len(content)})
    return content, source


def _text(value, limit=20_000) -> str:
    return str(value or "").strip()[:limit]


def _number(value, default, minimum, maximum):
    try:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError
        return max(minimum, min(maximum, value))
    except (TypeError, ValueError):
        return default


def _integer(value, default, minimum, maximum):
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return default


def _boolean(value, default=True):
    return value if isinstance(value, bool) else default


def _list_of_text(value, limit=80, item_limit=1000):
    if not isinstance(value, list):
        return []
    return [_text(item, item_limit) for item in value[:limit] if _text(item, item_limit)]


def _normalize_custom_model(raw) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    allowed_scalar = {"architecture", "activation", "encoder_activation", "decoder_activation", "posterior", "reconstruction_loss", "rnn_type"}
    allowed_bool = {"layer_norm", "bias", "bidirectional"}
    allowed_float = {"dropout"}
    allowed_lists = {
        "encoder_layers", "decoder_layers", "conv_encoder_channels", "conv_encoder_kernel_sizes",
        "conv_encoder_strides", "conv_encoder_paddings", "conv_encoder_dense_layers",
        "conv_decoder_base_shape", "conv_decoder_channels", "conv_decoder_kernel_sizes",
        "conv_decoder_strides", "conv_decoder_paddings", "conv_decoder_dense_layers",
    }
    allowed_int = {
        "sequence_length": (0, 65536), "d_model": (4, 2048), "num_heads": (1, 64),
        "num_layers": (1, 24), "feedforward_dim": (4, 8192), "rnn_hidden_size": (4, 4096),
        "d_state": (1, 256), "d_conv": (2, 16), "expand": (1, 8),
    }
    result = {}
    for key in allowed_scalar:
        if key in raw:
            result[key] = _text(raw[key], 80).lower()
    for key in allowed_bool:
        if key in raw:
            result[key] = bool(raw[key])
    for key in allowed_float:
        if key in raw:
            result[key] = _number(raw[key], 0.0, 0.0, 0.9)
    for key, (minimum, maximum) in allowed_int.items():
        if key in raw:
            result[key] = _integer(raw[key], minimum, minimum, maximum)
    for key in allowed_lists:
        if isinstance(raw.get(key), list):
            minimum = 0 if "paddings" in key else 1
            result[key] = [_integer(item, 1, minimum, 8192) for item in raw[key][:12]]
    return result


def normalize_protocol(raw: dict, source: dict) -> dict:
    """Convert untrusted model output into the dashboard's supported schema."""
    if not isinstance(raw, dict):
        raise LLMError("Protocol response is not a JSON object")
    paper_raw = raw.get("paper") if isinstance(raw.get("paper"), dict) else {}
    dataset_raw = raw.get("dataset") if isinstance(raw.get("dataset"), dict) else {}
    model_raw = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    training_raw = raw.get("training") if isinstance(raw.get("training"), dict) else {}
    seed_raw = raw.get("seed_plan") if isinstance(raw.get("seed_plan"), dict) else {}

    dataset_source = _text(dataset_raw.get("source"), 80).lower().replace("-", "_")
    aliases = {"3dshapes": "shapes3d", "3d_shapes": "shapes3d", "fashionmnist": "fashion_mnist", "custom": "uploaded"}
    dataset_source = aliases.get(dataset_source, dataset_source)
    if dataset_source not in DATASET_SOURCES:
        dataset_source = "uploaded"
    dataset_type = _text(dataset_raw.get("dataset_type"), 80).lower().replace(" ", "_") or "mixed"
    if dataset_type not in {"linear", "circle", "multiple_circles", "sphere", "torus", "swiss_roll", "dsprites", "mixed"}:
        dataset_type = "mixed"
    factor_types = _list_of_text(dataset_raw.get("factor_types"), limit=32, item_limit=60)
    factor_count = _integer(dataset_raw.get("factor_count"), len(factor_types) or 2, 1, 32)
    if not factor_types:
        factor_types = ["linear"] * factor_count
    if len(factor_types) < factor_count:
        factor_types.extend([factor_types[-1]] * (factor_count - len(factor_types)))
    factor_types = factor_types[:factor_count]
    latent_dim = _integer(model_raw.get("latent_dim", dataset_raw.get("latent_dim")), factor_count, 1, 256)

    model_type = _text(model_raw.get("model_type"), 80).lower().replace("-", "_")
    model_aliases = {"beta_vae": "beta_vae", "β_vae": "beta_vae", "full_covariance_beta_vae": "beta_vae_full_cov", "ae": "autoencoder"}
    model_type = model_aliases.get(model_type, model_type)
    if model_type not in MODEL_TYPES:
        model_type = "custom_vae" if model_raw.get("custom_model") else "beta_vae"

    optimizer = _text(training_raw.get("optimizer"), 40).lower()
    if optimizer not in {"adam", "adamw", "sgd", "adagrad"}:
        optimizer = "adam"
    reconstruction = _text(training_raw.get("reconstruction_loss"), 80).lower()
    reconstruction_aliases = {"bce": "bernoulli_logits", "bernoulli": "bernoulli_logits", "mse": "mse_sum", "l1": "l1_sum"}
    reconstruction = reconstruction_aliases.get(reconstruction, reconstruction)
    if reconstruction not in {"mse_sum", "mse_mean", "bernoulli_logits", "l1_sum"}:
        reconstruction = "mse_sum"
    activation_values = {"relu", "gelu", "tanh", "silu"}
    encoder_activation = _text(training_raw.get("encoder_activation"), 40).lower()
    decoder_activation = _text(training_raw.get("decoder_activation"), 40).lower()
    if encoder_activation not in activation_values:
        encoder_activation = "relu"
    if decoder_activation not in activation_values:
        decoder_activation = encoder_activation

    training_seed = _integer(training_raw.get("seed"), 1, -2_147_483_648, 2_147_483_647)
    training_seeds = []
    for item in seed_raw.get("training_seeds", []) if isinstance(seed_raw.get("training_seeds"), list) else []:
        seed = _integer(item, None, -2_147_483_648, 2_147_483_647)
        if seed is not None and seed not in training_seeds:
            training_seeds.append(seed)
    if not training_seeds:
        training_seeds = [training_seed]

    paper = {
        "title": _text(paper_raw.get("title"), 500) or "Untitled paper reproduction",
        "citation": _text(paper_raw.get("citation"), 1000),
        "url": _text(paper_raw.get("url") or source.get("url"), 1000),
        "summary": _text(paper_raw.get("summary"), 8000),
        "hypothesis": _text(paper_raw.get("hypothesis"), 4000),
    }
    dataset = {
        "source": dataset_source, "dataset_type": dataset_type,
        "factor_types": factor_types, "factor_count": factor_count,
        "latent_dim": latent_dim,
        "sparsity": _number(dataset_raw.get("sparsity"), 1.0, 0.0, 1.0),
        "dataset_size": _integer(dataset_raw.get("dataset_size"), 5000, 100, 2_000_000),
        "observation_dim": _integer(dataset_raw.get("observation_dim"), 12, 2, 100_000),
        "noise": _number(dataset_raw.get("noise"), 0.0, 0.0, 100.0),
        "train_split": _number(dataset_raw.get("train_split"), 0.8, 0.1, 0.98),
        "validation_split": _number(dataset_raw.get("validation_split"), 0.1, 0.0, 0.45),
        "seed": _integer(dataset_raw.get("seed"), 42, -2_147_483_648, 2_147_483_647),
        "mixing": _text(dataset_raw.get("mixing"), 40).lower() or "orthogonal",
        "degeneracy": _number(dataset_raw.get("degeneracy"), 1.0, 0.000001, 1.0),
        "notes": _text(dataset_raw.get("notes"), 6000),
    }
    validation_split = min(dataset["validation_split"], 1.0 - dataset["train_split"])
    dataset["validation_split"] = validation_split

    custom_model = _normalize_custom_model(model_raw.get("custom_model"))
    model = {
        "model_type": model_type, "latent_dim": latent_dim,
        "beta": _number(model_raw.get("beta"), 1.0, 0.0, 100.0),
        "custom_model": custom_model,
        "notes": _text(model_raw.get("notes"), 6000),
    }
    training = {
        "epochs": _integer(training_raw.get("epochs"), 40, 1, 2000),
        "training_steps": _integer(training_raw.get("training_steps"), 0, 0, 10_000_000),
        "learning_rate": _number(training_raw.get("learning_rate"), 0.001, 0.000001, 1.0),
        "batch_size": _integer(training_raw.get("batch_size"), 256, 8, 8192),
        "optimizer": optimizer,
        "adam_beta1": _number(training_raw.get("adam_beta1"), 0.9, 0.0, 0.999999),
        "adam_beta2": _number(training_raw.get("adam_beta2"), 0.999, 0.0, 0.999999),
        "optimizer_epsilon": _number(training_raw.get("optimizer_epsilon"), 1e-8, 1e-12, 0.1),
        "weight_decay": _number(training_raw.get("weight_decay"), 0.0, 0.0, 10.0),
        "reconstruction_loss": reconstruction,
        "hidden_dim": _integer(training_raw.get("hidden_dim"), 128, 4, 4096),
        "encoder_depth": _integer(training_raw.get("encoder_depth"), 2, 0, 8),
        "decoder_depth": _integer(training_raw.get("decoder_depth"), 2, 0, 8),
        "encoder_activation": encoder_activation, "decoder_activation": decoder_activation,
        "bias": _boolean(training_raw.get("bias"), True),
        "tc_weight": _number(training_raw.get("tc_weight"), 6.0, 0.0, 10_000.0),
        "seed": training_seed, "paper_reproduction": True,
        "notes": _text(training_raw.get("notes"), 6000),
    }
    analysis_plan = [key for key in _list_of_text(raw.get("analysis_plan"), limit=20, item_limit=80) if key in ANALYSIS_PLAN_KEYS]
    if not analysis_plan:
        analysis_plan = ["representation", "decoder_geometry", "precision_polarization", "factor_probes", "latent_explorer"]

    evidence = []
    for item in raw.get("evidence", [])[:100] if isinstance(raw.get("evidence"), list) else []:
        if not isinstance(item, dict):
            continue
        status = _text(item.get("status"), 30).lower()
        if status not in {"reported", "inferred", "assumed", "missing"}:
            status = "inferred"
        evidence.append({
            "field": _text(item.get("field"), 300), "value": _text(item.get("value"), 1000),
            "source_location": _text(item.get("source_location"), 1000),
            "confidence": _number(item.get("confidence"), 0.5, 0.0, 1.0), "status": status,
        })

    # The dashboard needs concrete control values, but a normalized fallback is
    # not paper evidence. Mark every such fallback independently of whether the
    # model remembered to list it as unresolved.
    fallback_candidates = {
        "dataset.source": (dataset_raw.get("source"), dataset["source"]),
        "dataset.factor_count": (dataset_raw.get("factor_count"), dataset["factor_count"]),
        "dataset.dataset_size": (dataset_raw.get("dataset_size"), dataset["dataset_size"]),
        "dataset.train_split": (dataset_raw.get("train_split"), dataset["train_split"]),
        "dataset.seed": (dataset_raw.get("seed"), dataset["seed"]),
        "model.model_type": (model_raw.get("model_type"), model["model_type"]),
        "model.latent_dim": (model_raw.get("latent_dim", dataset_raw.get("latent_dim")), model["latent_dim"]),
        "model.beta": (model_raw.get("beta"), model["beta"]),
        "training.learning_rate": (training_raw.get("learning_rate"), training["learning_rate"]),
        "training.batch_size": (training_raw.get("batch_size"), training["batch_size"]),
        "training.optimizer": (training_raw.get("optimizer"), training["optimizer"]),
        "training.reconstruction_loss": (training_raw.get("reconstruction_loss"), training["reconstruction_loss"]),
        "training.seed": (training_raw.get("seed"), training["seed"]),
    }
    if not training_raw.get("epochs") and not training_raw.get("training_steps"):
        fallback_candidates["training.epochs"] = (training_raw.get("epochs"), training["epochs"])
    fallback_fields = []
    unresolved = _list_of_text(raw.get("unresolved"), 80, 1200)
    reproduction_limits = _list_of_text(raw.get("reproduction_limits"), 80, 1200)
    for field, (original, fallback) in fallback_candidates.items():
        if original is not None and original != "":
            continue
        fallback_fields.append(field)
        note = f"{field} was not reported; dashboard fallback {fallback!r} is review-required"
        if not any(field in item for item in unresolved):
            unresolved.append(note)
        if not any(field == item.get("field") for item in evidence):
            evidence.append({
                "field": field, "value": str(fallback),
                "source_location": "not reported; dashboard normalization fallback",
                "confidence": 0.0, "status": "assumed",
            })
    if fallback_fields:
        fallback_limit = "Dashboard control fallbacks are not paper claims: " + ", ".join(fallback_fields)
        if fallback_limit not in reproduction_limits:
            reproduction_limits.append(fallback_limit)

    paper_protocol = {
        "paper_title": paper["title"], "paper_url": paper["url"],
        "model_type": model_type, "latent_dim": latent_dim, "beta": model["beta"],
        **training, "custom_model": custom_model,
        "dataset_source": dataset_source, "paper_reproduction": True,
        "seed_source": "effective_training_configuration",
        "paper_result_scope": _text(seed_raw.get("aggregation"), 1000) or f"aggregate of {len(training_seeds)} training seed(s)",
        "llm_extraction": {
            "review_required": True, "source_sha256": source.get("sha256", ""),
            "evidence_count": len(evidence), "fallback_fields": fallback_fields,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return {
        "paper": paper, "dataset": dataset, "model": model, "training": training,
        "analysis_plan": analysis_plan,
        "seed_plan": {
            "training_seeds": training_seeds[:200],
            "metric_seed": _integer(seed_raw.get("metric_seed"), 0, -2_147_483_648, 2_147_483_647),
            "aggregation": _text(seed_raw.get("aggregation"), 2000),
            "notes": _text(seed_raw.get("notes"), 4000),
        },
        "reproduction_limits": reproduction_limits,
        "unresolved": unresolved,
        "evidence": evidence,
        "paper_protocol": paper_protocol,
        "source": source,
        "review_required": True,
    }


PROTOCOL_SYSTEM_PROMPT = """You are a meticulous machine-learning reproduction methodologist. Extract only what the supplied paper supports. Distinguish reported values from inferences, assumptions, and missing information. Never invent a hyperparameter. When a value is absent, use null in the schema, add the field to unresolved, and explain the practical reproduction choice in reproduction_limits. Map requested analyses to the dashboard keys: representation, decoder_geometry, precision_polarization, tangent_metrics, topology_folding, factor_probes, causal_interference, latent_explorer, factor_traversal. Return the required structured object only. The result is a draft for human review and must not claim that an experiment has run."""


def protocol_prompt(paper_text: str, notes: str) -> str:
    return f"""<paper>
{paper_text}
</paper>

<dashboard_capabilities>
Dataset sources: {', '.join(sorted(DATASET_SOURCES))}.
Model types: {', '.join(sorted(MODEL_TYPES))}.
Optimizers: adam, adamw, sgd, adagrad.
Reconstruction losses: mse_sum, mse_mean, bernoulli_logits, l1_sum.
Activations: relu, gelu, tanh, silu.
Custom architectures may specify MLP, convolutional, Transformer, RNN/GRU/LSTM, or Mamba fields. Sequence models use sequence_length; Transformers use d_model, num_heads, num_layers, and feedforward_dim; recurrent models use rnn_type, rnn_hidden_size, num_layers, and bidirectional; Mamba uses d_model, num_layers, d_state, d_conv, and expand. In the structured response, custom_model must be a JSON-encoded string containing that declarative object, or null when no custom architecture is reported.
</dashboard_capabilities>

<user_notes>{notes}</user_notes>

Extract the paper identity, dataset, architecture, training protocol, seeds, evaluation plan, and missing information. Every nontrivial value must have an evidence entry with a section/table/page or a concise location description. Confidence is 0 to 1; status is reported, inferred, assumed, or missing. Preserve the distinction between training seed and metric seed. Prefer the exact experiment used for the paper's principal result; describe additional conditions in notes and reproduction_limits."""


def _safe_json_load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compact_value(value, depth=0):
    if depth > 6:
        return "[depth limited]"
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"latent_vectors", "projection", "factor_values", "factor_active", "mutual_information_matrix", "importance_matrix", "training_votes", "evaluation_votes", "score_matrix", "jacobians"}:
                continue
            result[str(key)[:160]] = _compact_value(item, depth + 1)
        return result
    if isinstance(value, list):
        if len(value) > 30:
            numeric = [float(item) for item in value if isinstance(item, (int, float)) and math.isfinite(float(item))]
            if len(numeric) == len(value):
                return {"n": len(numeric), "mean": statistics.fmean(numeric), "min": min(numeric), "max": max(numeric)}
            return [_compact_value(item, depth + 1) for item in value[:12]] + [f"[{len(value) - 12} more items omitted]"]
        return [_compact_value(item, depth + 1) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)[:500]


def _compact_run(directory: Path) -> dict:
    config = _safe_json_load(directory / "config.json") or {}
    training = _safe_json_load(directory / "training_config.json") or {}
    analysis = _safe_json_load(directory / "analysis.json") or {}
    result = {
        "id": directory.name,
        "identity": {key: config.get(key) for key in ["name", "run_label", "paper_title", "paper_citation", "paper_url", "hypothesis", "dataset_source", "dataset_variant", "dataset_size", "latent_dim", "effective_training_seed", "tags"]},
        "paper_protocol": _compact_value(config.get("paper_protocol", {})),
        "training": _compact_value(training),
        "analysis": _compact_value({key: analysis.get(key) for key in ["model_type", "model_label", "training_seed", "training_budget", "split_counts", "test_reconstruction_error", "polarization_summary"]}),
    }
    for name in ["paper_metrics.json", "scientific_metrics.json", "zietlow_metrics.json", "summary.json"]:
        value = _safe_json_load(directory / name)
        if value is not None:
            result[name.removesuffix(".json")] = _compact_value(value)
    return result


def _flatten_numbers(value, prefix="", output=None):
    output = output if output is not None else {}
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_numbers(item, f"{prefix}.{key}" if prefix else str(key), output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        output[prefix] = float(value)
    return output


def _aggregate_runs(runs: list[dict]) -> dict:
    values = {}
    for run in runs:
        for key, value in _flatten_numbers(run).items():
            if any(token in key for token in ["seed", "dataset_size", "latent_dim", "completed_steps", "requested_steps", "epoch"]):
                continue
            values.setdefault(key, []).append(value)
    result = {}
    for key, items in values.items():
        if len(items) < 2:
            continue
        result[key] = {
            "n": len(items), "mean": statistics.fmean(items),
            "sample_sd": statistics.stdev(items) if len(items) > 1 else 0.0,
            "min": min(items), "max": max(items),
        }
    return dict(sorted(result.items())[:500])


REPORT_SYSTEM_PROMPT = """You are writing a rigorous computational-reproducibility paper. Use only the supplied saved artifacts and aggregate statistics. Never fabricate a metric, significance test, citation, implementation detail, or paper result. Clearly distinguish the original paper's claim, the dashboard observation, interpretation, and limitations. If evidence is absent, state that it was not measured. Bibliography entries must come only from citations explicitly present in the supplied artifacts; omit rather than reconstruct or guess an entry. Write detailed, publication-quality prose in plain text without Markdown headings and without LaTeX commands. Do not claim exact reproduction unless the evidence explicitly supports it. Return the required structured object only."""


def report_prompt(context: dict, notes: str) -> str:
    return f"""<experiment_artifacts>
{json.dumps(context, ensure_ascii=False, indent=2)}
</experiment_artifacts>

<user_notes>{notes}</user_notes>

Write a self-contained research paper with an abstract, introduction, state of the art, theory of every reported metric, methodology, results with exact observed numbers, discussion, limitations, perspectives, and conclusion. Explain implementation departures and seed provenance. References must be copied only from citations already present in the artifacts; when a URL is unknown use an empty string. Data caveats must enumerate missing conditions, incomplete suites, assumptions, and non-equivalent implementations. Key findings should be concise and numerically grounded."""


def normalize_report(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise LLMError("Report response is not a JSON object")
    section_names = [
        "title", "abstract", "introduction", "state_of_the_art", "theory",
        "methodology", "results", "discussion", "limitations", "perspectives", "conclusion",
    ]
    result = {key: _text(raw.get(key), 60_000) for key in section_names}
    if not result["title"]:
        result["title"] = "Computational Reproduction Study"
    result["key_findings"] = _list_of_text(raw.get("key_findings"), 40, 3000)
    result["data_caveats"] = _list_of_text(raw.get("data_caveats"), 40, 3000)
    references = []
    for item in raw.get("references", [])[:100] if isinstance(raw.get("references"), list) else []:
        if isinstance(item, dict) and _text(item.get("citation"), 2000):
            references.append({"citation": _text(item.get("citation"), 2000), "url": _text(item.get("url"), 1000)})
    result["references"] = references
    return result


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _latex_paragraphs(value: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", value) if part.strip()]
    return "\n\n".join(_latex_escape(part.replace("\n", " ")) for part in paragraphs)


def render_arxiv_tex(report: dict, context: dict) -> str:
    sections = [
        ("Introduction", "introduction"), ("State of the Art", "state_of_the_art"),
        ("Theoretical Background and Metrics", "theory"), ("Methodology", "methodology"),
        ("Results", "results"), ("Discussion", "discussion"),
        ("Limitations", "limitations"), ("Perspectives", "perspectives"),
        ("Conclusion", "conclusion"),
    ]
    section_text = "\n".join(f"\\section{{{_latex_escape(title)}}}\n{_latex_paragraphs(report[key])}" for title, key in sections)
    findings = "\n".join(f"\\item {_latex_escape(item)}" for item in report["key_findings"])
    caveats = "\n".join(f"\\item {_latex_escape(item)}" for item in report["data_caveats"])
    references = "\n".join(
        f"\\bibitem{{ref{index}}} {_latex_escape(item['citation'])}" + (f" \\url{{{_latex_escape(item['url'])}}}" if item["url"] else "")
        for index, item in enumerate(report["references"], 1)
    ) or r"\bibitem{dashboard} Interpretability Dashboard experiment artifacts and saved provenance."
    generated = datetime.now(timezone.utc).strftime("%d %B %Y")
    return r"""\documentclass[11pt]{article}
\usepackage[letterpaper,margin=0.78in]{geometry}
\usepackage{amsmath,amssymb,booktabs,microtype,hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\title{""" + _latex_escape(report["title"]) + r"""}
\author{Automated draft from verified experiment artifacts}
\date{""" + generated + r"""}
\begin{document}
\maketitle
\begin{abstract}
""" + _latex_paragraphs(report["abstract"]) + r"""
\end{abstract}
\section*{Key Findings}
\begin{itemize}
""" + findings + r"""
\end{itemize}
""" + section_text + r"""
\section{Data and Reproduction Caveats}
\begin{itemize}
""" + caveats + r"""
\end{itemize}
\section*{Artifact Provenance}
This draft was generated from """ + str(context.get("run_count", 0)) + r""" saved run artifact(s). The language model was not allowed to execute training or modify measurements. All claims require human review against the original paper and run files.
\begin{thebibliography}{99}
""" + references + r"""
\end{thebibliography}
\end{document}
"""


def _report_markdown(report: dict, context: dict) -> str:
    blocks = [f"# {report['title']}", "## Abstract", report["abstract"], "## Key findings", "\n".join(f"- {item}" for item in report["key_findings"])]
    for title, key in [
        ("Introduction", "introduction"), ("State of the art", "state_of_the_art"),
        ("Theory and metrics", "theory"), ("Methodology", "methodology"),
        ("Results", "results"), ("Discussion", "discussion"),
        ("Limitations", "limitations"), ("Perspectives", "perspectives"), ("Conclusion", "conclusion"),
    ]:
        blocks.extend([f"## {title}", report[key]])
    blocks.extend(["## Data and reproduction caveats", "\n".join(f"- {item}" for item in report["data_caveats"]), "## References"])
    blocks.append("\n".join(f"- {item['citation']}" + (f" — {item['url']}" if item["url"] else "") for item in report["references"]))
    blocks.append(f"\nGenerated from {context.get('run_count', 0)} saved run artifact(s). Human verification is required.")
    return "\n\n".join(blocks).strip() + "\n"


def _compile_tex(report_directory: Path, tex_path: Path) -> tuple[str | None, str | None]:
    executable = shutil.which("pdflatex")
    if not executable:
        return None, "pdflatex is not installed; the arXiv-compatible .tex source is available."
    try:
        for _ in range(2):
            process = subprocess.run(
                [executable, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                cwd=report_directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=180, check=False,
            )
            if process.returncode != 0:
                return None, f"LaTeX compilation failed: {process.stdout[-2000:]}"
        pdf_path = tex_path.with_suffix(".pdf")
        return (pdf_path.name if pdf_path.exists() else None), None
    except Exception as exc:
        return None, f"LaTeX compilation failed: {exc}"


def register_llm_assistant(app, experiments_dir):
    dashboard_root = Path(__file__).resolve().parent
    geometry_root = Path(experiments_dir) / "geometry_lab"
    reports_root = dashboard_root / "ai_reports"
    drafts_root = dashboard_root / "ai_protocol_drafts"
    reports_root.mkdir(parents=True, exist_ok=True)
    drafts_root.mkdir(parents=True, exist_ok=True)
    store = ProviderStore(dashboard_root / ".llm_credentials.json")

    @app.before_request
    def llm_same_origin_guard():
        """Do not let an unrelated web page spend a locally configured key."""
        if not request.path.startswith("/api/llm/") or request.method in {"GET", "HEAD", "OPTIONS"}:
            return None
        if request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
            return jsonify({"success": False, "error": "Cross-site LLM requests are not allowed"}), 403
        origin = request.headers.get("Origin", "").strip()
        if not origin:  # Native clients and local scripts do not send Origin.
            return None
        parsed = urllib.parse.urlparse(origin)
        request_port = urllib.parse.urlparse(f"http://{request.host}").port or 80
        origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or origin_port != request_port:
            return jsonify({"success": False, "error": "LLM requests must originate from this local dashboard"}), 403
        return None

    def error_response(exc, status=400):
        return jsonify({"success": False, "error": str(exc)}), status

    def provider_config(payload):
        provider = _text(payload.get("provider"), 80) or None
        model = _text(payload.get("model"), 160) or None
        return store.resolve(provider, model)

    @app.route("/api/llm/settings", methods=["GET"])
    def llm_settings_get():
        return jsonify({"success": True, **store.public()})

    @app.route("/api/llm/settings", methods=["POST"])
    def llm_settings_update():
        try:
            raw = request.get_json(force=True) or {}
            provider = _text(raw.get("provider"), 80)
            settings = store.update(
                provider=provider,
                model=_text(raw.get("model"), 160),
                base_url=_text(raw.get("base_url"), 500),
                api_key=str(raw.get("api_key") or "").strip(),
                persist=bool(raw.get("persist", False)),
            )
            return jsonify({"success": True, **settings})
        except Exception as exc:
            return error_response(exc)

    @app.route("/api/llm/settings/<provider>/key", methods=["DELETE"])
    def llm_settings_clear_key(provider):
        try:
            store.clear_key(provider)
            return jsonify({"success": True, **store.public()})
        except Exception as exc:
            return error_response(exc)

    @app.route("/api/llm/test", methods=["POST"])
    def llm_test():
        try:
            raw = request.get_json(force=True) or {}
            config = provider_config(raw)
            schema = {
                "type": "object", "additionalProperties": False,
                "required": ["status"], "properties": {"status": {"type": "string"}},
            }
            started = time.time()
            result, usage = call_provider(
                config,
                "Return the required structured object. Set status to connected.",
                "Test this provider connection.", schema, "connection_test", max_output_tokens=100,
            )
            if _text(result.get("status"), 80).lower() != "connected":
                raise LLMError("Provider responded, but did not follow the structured test schema")
            return jsonify({
                "success": True, "provider": config["provider"], "model": config["model"],
                "latency_seconds": round(time.time() - started, 3), "usage": _compact_value(usage),
            })
        except Exception as exc:
            return error_response(exc)

    @app.route("/api/llm/analyze-paper", methods=["POST"])
    def llm_analyze_paper():
        try:
            paper_text, source = extract_paper(
                uploaded=request.files.get("paper"),
                text=request.form.get("paper_text", ""),
                url=request.form.get("paper_url", ""),
            )
            notes = _text(request.form.get("notes", ""), 10_000)
            config = provider_config(request.form)
            raw, usage = call_provider(
                config, PROTOCOL_SYSTEM_PROMPT, protocol_prompt(paper_text, notes),
                PROTOCOL_SCHEMA, "paper_reproduction_protocol", max_output_tokens=14000,
            )
            draft = normalize_protocol(raw, source)
            draft_id = f"protocol_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
            artifact = {
                "schema_version": 1, "id": draft_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "provider": config["provider"], "model": config["model"],
                "usage": _compact_value(usage), "draft": draft,
            }
            (drafts_root / f"{draft_id}.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
            warnings = []
            if source.get("truncated"):
                warnings.append(f"Paper text was truncated from {source['characters_original']:,} to {source['characters_used']:,} characters.")
            if draft["unresolved"]:
                warnings.append(f"{len(draft['unresolved'])} field(s) remain unresolved and require review.")
            assumed = sum(1 for item in draft["evidence"] if item["status"] in {"assumed", "inferred", "missing"})
            if assumed:
                warnings.append(f"{assumed} evidence item(s) are inferred, assumed, or missing rather than explicitly reported.")
            return jsonify({
                "success": True, "draft_id": draft_id, "draft": draft,
                "provider": config["provider"], "model": config["model"],
                "usage": _compact_value(usage), "warnings": warnings,
            })
        except LLMError as exc:
            return error_response(exc)
        except Exception as exc:
            return error_response(f"Paper analysis failed: {exc}", 500)

    def experiment_directory(exp_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,240}", exp_id or ""):
            raise LLMError("Invalid experiment identifier")
        directory = (geometry_root / exp_id).resolve()
        if geometry_root.resolve() not in directory.parents or not (directory / "config.json").exists():
            raise LLMError(f"Experiment not found: {exp_id}")
        return directory

    def report_directories(raw: dict) -> list[Path]:
        scope = _text(raw.get("scope"), 40).lower() or "active"
        exp_id = _text(raw.get("experiment_id"), 240)
        if scope == "active":
            return [experiment_directory(exp_id)]
        if scope == "selected":
            ids = raw.get("experiment_ids", [])
            if not isinstance(ids, list) or not ids:
                raise LLMError("Select at least one experiment")
            return [experiment_directory(_text(item, 240)) for item in ids[:MAX_REPORT_RUNS]]
        configs = []
        for directory in geometry_root.iterdir():
            if not directory.is_dir() or directory.name.startswith("_") or not (directory / "config.json").exists():
                continue
            config = _safe_json_load(directory / "config.json") or {}
            configs.append((directory, config))
        if scope == "run_label":
            active = _safe_json_load(experiment_directory(exp_id) / "config.json") or {}
            label = _text(raw.get("run_label") or active.get("run_label"), 240)
            if not label:
                raise LLMError("The active experiment has no run label")
            directories = [directory for directory, config in configs if config.get("run_label") == label]
        elif scope == "all_completed":
            directories = [directory for directory, _ in configs if (directory / "analysis.json").exists()]
            directories.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        else:
            raise LLMError("Report scope must be active, selected, run_label, or all_completed")
        if not directories:
            raise LLMError("No experiment artifacts match the requested report scope")
        if len(directories) > MAX_REPORT_RUNS:
            raise LLMError(f"The report scope contains {len(directories)} runs; narrow it to at most {MAX_REPORT_RUNS}")
        return directories

    @app.route("/api/llm/report", methods=["POST"])
    def llm_generate_report():
        try:
            raw = request.get_json(force=True) or {}
            config = provider_config(raw)
            directories = report_directories(raw)
            runs = [_compact_run(directory) for directory in directories]
            aggregate = _aggregate_runs(runs)
            if len(runs) <= 40:
                prompt_runs = runs
                omitted = 0
            else:
                prompt_runs = []
                for run in runs[:60]:
                    prompt_runs.append({key: run.get(key) for key in ["id", "identity", "analysis", "paper_metrics", "zietlow_metrics"] if key in run})
                omitted = len(runs) - len(prompt_runs)
            context = {
                "run_count": len(runs), "run_summaries": prompt_runs,
                "run_summaries_omitted": omitted, "aggregate_numeric_metrics": aggregate,
                "scope": _text(raw.get("scope"), 40) or "active",
            }
            notes = _text(raw.get("notes"), 12_000)
            generated, usage = call_provider(
                config, REPORT_SYSTEM_PROMPT, report_prompt(context, notes),
                REPORT_SCHEMA, "reproduction_research_report", max_output_tokens=24000,
            )
            report = normalize_report(generated)
            report_id = f"report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
            directory = reports_root / report_id
            directory.mkdir(parents=True, exist_ok=False)
            tex_path = directory / "paper.tex"
            md_path = directory / "paper.md"
            json_path = directory / "report.json"
            tex_path.write_text(render_arxiv_tex(report, context), encoding="utf-8")
            md_path.write_text(_report_markdown(report, context), encoding="utf-8")
            metadata = {
                "schema_version": 1, "id": report_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "provider": config["provider"], "model": config["model"],
                "usage": _compact_value(usage), "experiment_ids": [item.name for item in directories],
                "context": context, "report": report,
            }
            json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            pdf_name, compile_warning = _compile_tex(directory, tex_path)
            files = {
                "tex": f"/api/llm/reports/{report_id}/paper.tex",
                "markdown": f"/api/llm/reports/{report_id}/paper.md",
                "json": f"/api/llm/reports/{report_id}/report.json",
            }
            if pdf_name:
                files["pdf"] = f"/api/llm/reports/{report_id}/{pdf_name}"
            return jsonify({
                "success": True, "report_id": report_id, "title": report["title"],
                "abstract": report["abstract"], "key_findings": report["key_findings"],
                "data_caveats": report["data_caveats"], "run_count": len(runs),
                "provider": config["provider"], "model": config["model"],
                "files": files, "compile_warning": compile_warning,
            })
        except LLMError as exc:
            return error_response(exc)
        except Exception as exc:
            return error_response(f"Report generation failed: {exc}", 500)

    @app.route("/api/llm/reports/<report_id>/<filename>", methods=["GET"])
    def llm_report_download(report_id, filename):
        if not re.fullmatch(r"report_[A-Za-z0-9_-]{1,100}", report_id or ""):
            return error_response("Invalid report identifier", 404)
        allowed = {"paper.tex", "paper.md", "paper.pdf", "report.json"}
        if filename not in allowed:
            return error_response("Unknown report artifact", 404)
        directory = reports_root / report_id
        if not (directory / filename).exists():
            return error_response("Report artifact not found", 404)
        return send_from_directory(directory, filename, as_attachment=True, download_name=f"{report_id}_{filename}")

    @app.route("/api/llm/reports", methods=["GET"])
    def llm_reports_list():
        reports = []
        for directory in sorted(reports_root.glob("report_*"), key=lambda item: item.stat().st_mtime, reverse=True)[:100]:
            metadata = _safe_json_load(directory / "report.json") or {}
            reports.append({
                "id": directory.name, "generated_at": metadata.get("generated_at"),
                "title": (metadata.get("report") or {}).get("title"),
                "run_count": len(metadata.get("experiment_ids") or []),
                "has_pdf": (directory / "paper.pdf").exists(),
            })
        return jsonify({"success": True, "reports": reports})
