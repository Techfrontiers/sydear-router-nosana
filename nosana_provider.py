"""Nosana provider prototype for the SYDEAR Router Gateway (Decentralize AI entry).

Stdlib only. Nosana's hosted inference API is OpenAI-compatible, so this module
talks the same shape the gateway already speaks:

  GET  {base}/models            -> dynamic model list (never hardcoded)
  POST {base}/chat/completions  -> chat completion, OpenAI shape
  GET  https://api.nosana.com/credits/llm/usage -> per-request records (cross-check)

Gateway naming convention is preserved: a model is addressed as
"nosana/<model-id>", exactly like "minimax-portal/<model-id>".

A receipt is produced for every routed request. It records what the gateway
saw (tokens, estimated cost from the model's own pricing) AND the provider's
own record (Nosana requestId + costNano from the usage API), so anyone can
verify a receipt against Nosana's ledger instead of taking our word for it.

No key is needed to run the tests: they run against mock_nosana.py on
localhost. A live test needs a real Nosana API key (the user's step).
"""

import http.client
import json
import ssl
import time
import urllib.parse
import uuid

INFERENCE_BASE = "https://inference.nosana.com/v1"
USAGE_BASE = "https://api.nosana.com"


class NosanaError(Exception):
    """An error in Nosana's own OpenAI shape. `code` is Nosana's code string."""

    def __init__(self, status, code, message):
        super().__init__("%s %s: %s" % (status, code, message))
        self.status = status
        self.code = code
        self.message = message


def _request(method, url, api_key, body=None, timeout=120):
    """One HTTPS request via http.client (no env-proxy interference).

    Returns parsed JSON on 2xx. Raises NosanaError preserving the provider's
    own status + code + message on any other status.
    """
    parts = urllib.parse.urlsplit(url)
    host = parts.netloc
    path = parts.path
    if parts.query:
        path += "?" + parts.query
    conn = http.client.HTTPSConnection(host, timeout=timeout,
                                       context=ssl.create_default_context()) \
        if parts.scheme == "https" else \
        http.client.HTTPConnection(host, timeout=timeout)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    payload = json.dumps(body) if body is not None else None
    try:
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
    finally:
        conn.close()
    try:
        data = json.loads(raw.decode("utf-8")) if raw else {}
    except ValueError:
        data = {}
    if 200 <= resp.status < 300:
        return data
    err = data.get("error", {}) if isinstance(data, dict) else {}
    raise NosanaError(resp.status, err.get("code", "unknown"),
                      err.get("message", "HTTP %s" % resp.status))


def split_model(provider_model):
    """'nosana/<model-id>' -> ('nosana', '<model-id>'). Raises ValueError."""
    provider, sep, model = provider_model.partition("/")
    if provider != "nosana" or not sep or not model:
        raise ValueError("expected 'nosana/<model-id>', got %r" % provider_model)
    return provider, model


def list_models(base_url=INFERENCE_BASE, api_key=""):
    """Dynamic model list. Only models with a GPU serving them appear."""
    data = _request("GET", base_url.rstrip("/") + "/models", api_key)
    return data.get("data", [])


def chat(base_url, api_key, model, messages, timeout=120):
    """POST /v1/chat/completions. Returns the raw OpenAI-shaped response."""
    return _request("POST", base_url.rstrip("/") + "/chat/completions", api_key,
                    {"model": model, "messages": messages}, timeout=timeout)


def fetch_usage(api_key, usage_base=USAGE_BASE, limit=20):
    """Per-request records from Nosana's own ledger (cross-check source)."""
    url = usage_base.rstrip("/") + "/credits/llm/usage?limit=%d" % limit
    return _request("GET", url, api_key)


def estimate_cost_usd(prompt_tokens, completion_tokens, pricing):
    """Cost in USD from the model's own /v1/models pricing ($/token).

    This is an ESTIMATE. The authoritative figure is Nosana's costNano from
    the usage API (see make_receipt). Credit-unit conversion is NOT assumed
    here: 1 credit's dollar value is unverified, so receipts carry USD
    estimates plus the raw costNano, never a converted credit figure.
    """
    p = float((pricing or {}).get("prompt", 0) or 0)
    c = float((pricing or {}).get("completion", 0) or 0)
    return round(prompt_tokens * p + completion_tokens * c, 8)


def make_receipt(gateway_request_id, model, pricing, completion_data,
                 usage_record=None):
    """Build one verifiable receipt for a routed request."""
    usage = (completion_data or {}).get("usage", {}) or {}
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    rec = (usage_record or {})
    return {
        "gateway_request_id": gateway_request_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": "nosana",
        "model": model,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cost_usd_estimate": estimate_cost_usd(pt, ct, pricing),
        "nosana_request_id": rec.get("requestId"),
        "nosana_cost_nano": rec.get("costNano"),
        "nosana_status": rec.get("status"),
        "verifiable_via": USAGE_BASE + "/credits/llm/usage",
    }


def new_request_id():
    return "gw-" + uuid.uuid4().hex[:16]
