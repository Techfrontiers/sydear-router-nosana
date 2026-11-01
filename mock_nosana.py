"""Mock Nosana server for offline tests (stdlib only, localhost only).

Implements the three endpoints the provider uses, in Nosana's own shapes:
  GET  /v1/models                -> {"object": "list", "data": [...]}
  POST /v1/chat/completions      -> OpenAI-shaped completion + usage
  GET  /credits/llm/usage        -> {"usage": [...], "totals": {...}}

Test hooks (so error paths are testable without a real key):
  - Authorization header other than "Bearer mock-key" -> 401 invalid_api_key
  - model "nope/nope" in a chat body -> 404 model_not_found
  - header "X-Mock-Credits: 0" -> 402 insufficient_credits
The model list is mutable at runtime (server.models) so tests can prove the
provider never hardcodes a model id: change the list, list again, see it move.

Run directly to smoke-test: python3 mock_nosana.py  (serves on an ephemeral
port, prints it, serves until killed).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS_V1 = [
    {"id": "mock/qwen-mock-9b", "object": "model",
     "pricing": {"prompt": "0.000000100", "completion": "0.000000300"}},
]
MODELS_V2 = [
    {"id": "mock/llama-mock-8b", "object": "model",
     "pricing": {"prompt": "0.000000200", "completion": "0.000000400"}},
    {"id": "mock/qwen-mock-9b", "object": "model",
     "pricing": {"prompt": "0.000000100", "completion": "0.000000300"}},
]


class Handler(BaseHTTPRequestHandler):
    server_version = "MockNosana/1.0"

    def log_message(self, *args):
        pass

    def _send(self, status, obj):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _auth_ok(self):
        return self.headers.get("Authorization") == "Bearer mock-key"

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            if not self._auth_ok():
                return self._send(401, {"error": {"code": "invalid_api_key",
                                                  "message": "bad key"}})
            return self._send(200, {"object": "list", "data": self.server.models})
        if self.path.startswith("/credits/llm/usage"):
            if not self._auth_ok():
                return self._send(401, {"error": {"code": "invalid_api_key",
                                                  "message": "bad key"}})
            return self._send(200, {
                "usage": [{"requestId": "mock-req-1", "model": "mock/qwen-mock-9b",
                           "promptTokens": 17, "completionTokens": 42,
                           "costNano": 14300, "status": 200}],
                "totals": {"promptTokens": 17, "completionTokens": 42,
                           "costNano": 14300}})
        return self._send(404, {"error": {"code": "not_found",
                                          "message": "no such route"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if self.path.startswith("/v1/chat/completions"):
            if not self._auth_ok():
                return self._send(401, {"error": {"code": "invalid_api_key",
                                                  "message": "bad key"}})
            if self.headers.get("X-Mock-Credits") == "0":
                return self._send(402, {"error": {"code": "insufficient_credits",
                                                  "message": "no credits"}})
            if body.get("model") == "nope/nope":
                return self._send(404, {"error": {"code": "model_not_found",
                                                  "message": "not served"}})
            msgs = body.get("messages", [])
            prompt_tokens = sum(len(m.get("content", "")) for m in msgs) // 4
            return self._send(200, {
                "id": "chatcmpl-mock1", "object": "chat.completion",
                "model": body.get("model"),
                "choices": [{"message": {"role": "assistant",
                                         "content": "MOCK-OK: decentralized GPUs work."}}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": 42,
                          "total_tokens": prompt_tokens + 42}})
        return self._send(404, {"error": {"code": "not_found",
                                          "message": "no such route"}})


class MockNosana(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address=("127.0.0.1", 0)):
        super().__init__(address, Handler)
        self.models = list(MODELS_V1)


def start():
    server = MockNosana()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


if __name__ == "__main__":
    srv = start()
    print("mock nosana on 127.0.0.1:%d" % srv.server_address[1])
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
