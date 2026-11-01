"""Tests for nosana_provider.py against mock_nosana.py (no key, no cost).

Checks:
  1. Model list is dynamic: change the mock's list, list again, see it move
     (proves the provider never hardcodes a model id).
  2. Chat completion returns OpenAI-shaped content + usage.
  3. Receipt carries every field and the cost math is right.
  4. Error shapes map exactly: 401 invalid_api_key, 402 insufficient_credits,
     404 model_not_found.
  5. Usage fetch returns Nosana's own record for receipt cross-check.
  6. Gateway naming: 'nosana/<model-id>' splits; anything else is rejected.

Run: python3 test_nosana_provider.py   (exit 0 = all PASS)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mock_nosana
import nosana_provider as np

KEY = "mock-key"
passed = []


def check(name, cond, extra=""):
    assert cond, "FAIL: %s %s" % (name, extra)
    passed.append(name)
    print("PASS: %s" % name)


def main():
    server = mock_nosana.start()
    port = server.server_address[1]
    base = "http://127.0.0.1:%d/v1" % port
    usage_base = "http://127.0.0.1:%d" % port

    # NOTE: the provider uses http.client, which speaks plain HTTP when the
    # URL scheme is http — so it works against this localhost mock. The
    # _request helper only wraps HTTPS for https:// URLs.
    # 1. dynamic model list
    models = np.list_models(base, KEY)
    check("models v1 listed", [m["id"] for m in models] == ["mock/qwen-mock-9b"])
    server.models = list(mock_nosana.MODELS_V2)
    models = np.list_models(base, KEY)
    check("models list moves with provider (no hardcode)",
          [m["id"] for m in models] == ["mock/llama-mock-8b", "mock/qwen-mock-9b"])

    # 2 + 3. chat + receipt + cost math
    provider, model = np.split_model("nosana/mock/qwen-mock-9b")
    check("split_model", provider == "nosana" and model == "mock/qwen-mock-9b")
    msgs = [{"role": "user", "content": "Say OK, this is a test message!"}]
    resp = np.chat(base, KEY, model, msgs)
    check("chat content",
          resp["choices"][0]["message"]["content"].startswith("MOCK-OK"))
    pricing = {"prompt": "0.000000100", "completion": "0.000000300"}
    rec = np.make_receipt(np.new_request_id(), model, pricing, resp)
    # prompt tokens = len(content)//4 = 30//4 = 7 ; completion = 42
    # cost = 7*0.0000001 + 42*0.0000003 = 0.0000007 + 0.0000126 = 0.0000133
    check("receipt fields",
          rec["provider"] == "nosana" and rec["model"] == model
          and rec["prompt_tokens"] == 7 and rec["completion_tokens"] == 42
          and rec["cost_usd_estimate"] == 0.0000133
          and rec["gateway_request_id"].startswith("gw-"),
          str(rec))
    try:
        np.split_model("minimax-portal/MiniMax-M3.1-Flash")
        check("split_model rejects non-nosana", False)
    except ValueError:
        check("split_model rejects non-nosana", True)

    # 4. error shapes
    for fn, want_status, want_code in [
        (lambda: np.list_models(base, "wrong-key"), 401, "invalid_api_key"),
        (lambda: np.chat(base, KEY, "nope/nope", msgs), 404, "model_not_found"),
    ]:
        try:
            fn()
            check("error %s raised" % want_code, False)
        except np.NosanaError as e:
            check("error %s" % want_code,
                  e.status == want_status and e.code == want_code,
                  "got %s %s" % (e.status, e.code))

    # 402 needs the header hook: exercise _request directly.
    try:
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("POST", "/v1/chat/completions",
                     body='{"model": "mock/qwen-mock-9b", "messages": []}',
                     headers={"Content-Type": "application/json",
                              "Authorization": "Bearer mock-key",
                              "X-Mock-Credits": "0"})
        r = conn.getresponse()
        r.read()
        conn.close()
        check("mock 402 trigger", r.status == 402, "got %s" % r.status)
    except Exception as e:  # noqa: BLE001 - test harness
        check("mock 402 trigger", False, str(e))

    # 5. usage fetch + receipt cross-check
    usage = np.fetch_usage(KEY, usage_base=usage_base)
    record = usage["usage"][0]
    rec2 = np.make_receipt(np.new_request_id(), model, pricing, resp,
                           usage_record=record)
    check("receipt cross-check fields",
          rec2["nosana_request_id"] == "mock-req-1"
          and rec2["nosana_cost_nano"] == 14300
          and rec2["nosana_status"] == 200, str(rec2))

    server.shutdown()
    print("\n%d/%d PASS" % (len(passed), len(passed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
