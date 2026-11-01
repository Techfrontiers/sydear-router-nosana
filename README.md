# SYDEAR Router Gateway on Nosana

Nosana provider prototype for a self-hosted AI gateway, built for the
Decentralize AI Hackathon Round 2 (HackerNoon x Nosana x Arweave x MEXC).

## What it is

The gateway routes chat requests by `provider/model` naming. Nosana's
hosted inference API is OpenAI-compatible, so the gateway gains a
decentralized-GPU provider by changing the base URL and key, reading the
live model list (never hardcoding a model id), and billing per token
against credits.

Every routed request produces a receipt: gateway request id, timestamp,
provider, model, prompt and completion tokens, cost estimate from the
model's own pricing, plus Nosana's own record (requestId, costNano,
status) fetched from the usage API, so anyone can verify a receipt
against the provider's ledger.

## Files

- `nosana_provider.py` — provider module (stdlib only). Model addressed
  as `nosana/<model-id>`. No key is stored anywhere; the key is passed
  in by the caller.
- `mock_nosana.py` — localhost mock of the three endpoints, with the
  same error shapes (401 invalid_api_key, 402 insufficient_credits,
  404 model_not_found).
- `test_nosana_provider.py` — 10 offline checks. Run:
  `python3 test_nosana_provider.py`

## Live status (2026-11-01)

- Live `/v1/models` check passed: 2 models served at check time
  (`qwen/qwen3.8-27b` for chat, `nvidia/nemotron-3-embed-1b` for
  embeddings). The served list changes with GPU availability.
- Live chat attempt returned 402 `insufficient_credits` on a key with
  zero usage history. The key itself is valid; spendable credits were
  not attached yet. A live run is pending the participant credit claim.
- No secrets are committed in this repo. No live gateway code is
  included; integration into the running gateway is a separate step.
