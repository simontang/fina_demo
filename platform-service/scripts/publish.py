#!/usr/bin/env python3
"""Publish a factory event through the platform-service webhook facade.

Usage:
  python3 scripts/publish.py --tenant hankel --event-type job.completed --payload '{"jobId":"1"}'
  python3 scripts/publish.py --tenant hankel --event-type job.completed --channel vip --payload '{"jobId":"1"}'

Env:
  PLATFORM_SERVICE_URL (default http://localhost:5707)
  PLATFORM_SERVICE_API_KEY (optional; set only when the service enforces X-Api-Key)
"""
import argparse
import json
import os
import urllib.request

base = os.environ.get("PLATFORM_SERVICE_URL", "http://localhost:5707").rstrip("/")
key = os.environ.get("PLATFORM_SERVICE_API_KEY", "")

parser = argparse.ArgumentParser()
parser.add_argument("--tenant", required=True)
parser.add_argument("--event-type", required=True)
parser.add_argument("--payload", default="{}")
parser.add_argument("--channel", action="append", default=[], help="Svix channel filter; repeat for multiple")
args = parser.parse_args()

headers = {
    "X-Tenant-Id": args.tenant,
    "Content-Type": "application/json",
}
if key:
    headers["X-Api-Key"] = key

body = {"eventType": args.event_type, "payload": json.loads(args.payload)}
channels = [channel.strip() for channel in args.channel if channel.strip()]
if channels:
    body["channels"] = channels
payload = json.dumps(body).encode()
req = urllib.request.Request(f"{base}/api/v1/webhooks/publish", data=payload, method="POST", headers=headers)
with urllib.request.urlopen(req) as resp:
    print(f"HTTP {resp.status}: {resp.read().decode()}")
