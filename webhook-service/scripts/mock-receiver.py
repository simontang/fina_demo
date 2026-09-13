#!/usr/bin/env python3
"""Mock webhook receiver: logs every delivery and verifies Standard Webhooks
signatures when RECEIVER_WEBHOOK_SECRET is set (whsec_... format).

Usage:
  python3 mock-receiver.py [--port 5909] [--out /tmp/received.log]

Each delivery is appended to --out as one JSON line:
  {"time":..., "webhook-id":..., "webhook-timestamp":..., "signature-valid":true, "body":...}
"""
import argparse
import base64
import hashlib
import hmac
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--port", type=int, default=5909)
args_parser.add_argument("--out", default="/tmp/webhook-received.log")
cli = args_parser.parse_args()

secret_b64 = None
_env_secret = __import__("os").environ.get("RECEIVER_WEBHOOK_SECRET", "")
if _env_secret:
    secret_b64 = _env_secret.removeprefix("whsec_")


def verify(webhook_id: str, timestamp: str, body: bytes, signature_header: str) -> bool:
    if not secret_b64 or not signature_header:
        return False
    signed_content = f"{webhook_id}.{timestamp}.".encode() + body
    expected = base64.b64encode(
        hmac.new(base64.b64decode(secret_b64), signed_content, hashlib.sha256).digest()
    ).decode()
    for part in signature_header.split(" "):
        if part.startswith("v1,") and hmac.compare_digest(part[3:], expected):
            return True
    return False


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        wid = self.headers.get("webhook-id", "")
        ts = self.headers.get("webhook-timestamp", "")
        sig = self.headers.get("webhook-signature", "")
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "path": self.path,
            "webhook-id": wid,
            "webhook-timestamp": ts,
            "webhook-signature": sig,
            "signature-valid": verify(wid, ts, body, sig),
            "body": body.decode(errors="replace"),
        }
        with open(cli.out, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"mock receiver listening on :{cli.port}, log={cli.out}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", cli.port), Handler).serve_forever()
