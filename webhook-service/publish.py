#!/usr/bin/env python3
"""Publish an event to the Outpost webhook service.

Usage:
  python3 publish.py --tenant hankel --topic job.completed --data '{"jobId":"1"}'

Env:
  WEBHOOK_SERVICE_URL (default http://localhost:5708)
  WEBHOOK_SERVICE_API_KEY (default webhook-demo-key)
"""
import argparse
import json
import os
import urllib.request

base = os.environ.get("WEBHOOK_SERVICE_URL", "http://localhost:5708").rstrip("/")
key = os.environ.get("WEBHOOK_SERVICE_API_KEY", "webhook-demo-key")

parser = argparse.ArgumentParser()
parser.add_argument("--tenant", required=True)
parser.add_argument("--topic", required=True)
parser.add_argument("--data", default="{}")
args = parser.parse_args()

payload = json.dumps({
    "tenant_id": args.tenant,
    "topic": args.topic,
    "data": json.loads(args.data),
}).encode()

req = urllib.request.Request(
    f"{base}/api/v1/publish",
    data=payload,
    method="POST",
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    },
)
with urllib.request.urlopen(req) as resp:
    body = resp.read().decode()
    print(f"HTTP {resp.status}: {body}")
