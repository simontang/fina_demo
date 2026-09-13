#!/usr/bin/env bash
# Webhook module end-to-end smoke against the platform-service facade:
# provision destination -> start signature-verifying receiver -> publish ->
# wait for delivery -> verify Standard Webhooks signature.
#
# Prereq: platform-service (5707) running with a reachable svix-server.
# Env: PLATFORM_SERVICE_URL, PLATFORM_SERVICE_API_KEY, RECEIVER_PORT.
set -euo pipefail

BASE="${PLATFORM_SERVICE_URL:-http://localhost:5707}"
TENANT="whsmoke-$(date +%m%d%H%M%S)"
RECEIVER_PORT="${RECEIVER_PORT:-5909}"
DEST_URL="http://host.docker.internal:${RECEIVER_PORT}/catch"
RCV="$(mktemp /tmp/webhook-received-XXXXXX.log)"
RCV_PID=""

cleanup() {
  [[ -n "$RCV_PID" ]] && kill "$RCV_PID" 2>/dev/null || true
  rm -f "$RCV"
}
trap cleanup EXIT

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1" >&2; exit 1; }
AUTH=()
[[ -n "${PLATFORM_SERVICE_API_KEY:-}" ]] && AUTH=(-H "X-Api-Key: ${PLATFORM_SERVICE_API_KEY}")

echo "== provision destination for ${TENANT} =="
PROVISION=$(bash "$(dirname "$0")/provision-destination.sh" "$TENANT" "$DEST_URL" "job.completed")
echo "$PROVISION" | grep -q "whsec_" || fail "no signing secret in provision response: $PROVISION"
SECRET=$(echo "$PROVISION" | python3 -c "import json,sys;print(json.load(sys.stdin)['secret'])")
pass "destination registered (secret ${SECRET:0:12}...)"

echo "== start signature-verifying receiver on :${RECEIVER_PORT} =="
RECEIVER_WEBHOOK_SECRET="$SECRET" python3 "$(dirname "$0")/mock-receiver.py" \
  --port "$RECEIVER_PORT" --out "$RCV" > /dev/null 2>&1 &
RCV_PID=$!
sleep 1
kill -0 "$RCV_PID" || fail "receiver did not start"
pass "receiver up"

echo "== publish job.completed =="
PLATFORM_SERVICE_URL="$BASE" PLATFORM_SERVICE_API_KEY="${PLATFORM_SERVICE_API_KEY:-}" \
  python3 "$(dirname "$0")/publish.py" \
  --tenant "$TENANT" --topic job.completed --data '{"jobId":"smoke-1","status":"done"}' \
  | grep -q "HTTP 2" || fail "publish failed"
pass "event accepted"

echo "== wait for delivery =="
FOUND=0
for _ in $(seq 1 30); do
  if [[ -s "$RCV" ]] && grep -q "smoke-1" "$RCV"; then FOUND=1; break; fi
  sleep 1
done
[[ "$FOUND" == "1" ]] || fail "no delivery within 30s"
pass "delivery received"

echo "== verify signature and facade observability =="
python3 - "$RCV" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
hit = [r for r in rows if "smoke-1" in r.get("body", "")]
assert hit, "smoke delivery not found"
assert hit[-1]["signature-valid"] is True, f"signature invalid: {hit[-1]}"
print(f"signature-valid: True (webhook-id={hit[-1]['webhook-id']})")
PY
MSGS=$(curl -sf -H "X-Tenant-Id: ${TENANT}" ${AUTH[@]+"${AUTH[@]}"} "${BASE}/api/v1/webhooks/messages?limit=5")
echo "$MSGS" | grep -q "job.completed" || fail "facade messages missing topic: $MSGS"
pass "facade messages list observable"

echo ""
echo "ALL WEBHOOK SMOKE TESTS PASSED"
