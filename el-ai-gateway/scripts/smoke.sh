#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:5708}"
KEY="${KEY:-dev_key}"
FILE="${1:-/tmp/voice.wav}"

echo "== health =="
curl -sS "$BASE/health"; echo

echo "== upload =="
UPLOAD=$(curl -sS -X POST "$BASE/api/v1/files?path=voice" \
  -H "Authorization: Bearer $KEY" \
  -F "file=@${FILE}")
echo "$UPLOAD"
UUID=$(printf '%s' "$UPLOAD" | python3 -c 'import sys,json;print(json.load(sys.stdin)["uuid"])')

echo "== start task =="
TASK=$(curl -sS -X POST "$BASE/api/v1/voice-tagging" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"uuid\":\"$UUID\"}")
echo "$TASK"
TASK_ID=$(printf '%s' "$TASK" | python3 -c 'import sys,json;print(json.load(sys.stdin)["taskId"])')

echo "== task status =="
curl -sS "$BASE/api/v1/voice-tagging/$TASK_ID" -H "Authorization: Bearer $KEY"; echo

echo "== feedback =="
curl -sS -X POST "$BASE/api/v1/voice-tagging/$TASK_ID/feedback" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"content":"Smoke test feedback."}'; echo
