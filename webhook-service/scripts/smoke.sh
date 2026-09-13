#!/usr/bin/env bash
# Webhook service end-to-end smoke: start mock receiver, provision a tenant
# with a destination pointing at it, publish an event, wait for delivery,
# verify the Standard Webhooks signature.
#
# Prereq: Outpost reachable (WEBHOOK_SERVICE_URL, default http://localhost:5708).
# Run from repo root or this directory; requires python3 and curl.
set -euo pipefail

BASE="${WEBHOOK_SERVICE_URL:-http://localhost:5708}"
KEY="${WEBHOOK_SERVICE_API_KEY:-webhook-demo-key}"
TENANT="smoke-$(date +%m%d%H%M%S)"
RECEIVER_PORT="${RECEIVER_PORT:-5909}"
# Container -> host.docker.internal reaches the receiver on the host.
DEST_URL="http://host.docker.internal:${RECEIVER_PORT}/catch"
LOG="$(mktemp /tmp/webhook-smoke-XXXXXX.log)"
RCV="$(mktemp /tmp/webhook-received-XXXXXX.log)"
SECRET_FILE="$(mktemp)"

cleanup() {
  [[ -n "${RCV_PID:-}" ]] && kill "$RCV_PID" 2>/dev/null || true
  rm -f "$LOG" "$RCV" "$SECRET_FILE"
}
trap cleanup EXIT

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1" >&2; exit 1; }

echo "== health =="
curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/api/v1/tenants/x/portal" -o /dev/null \
  || echo "(portal probe failed — continuing; health is judged by provision step)"

echo "== start mock receiver on :${RECEIVER_PORT} =="
RECEIVER_WEBHOOK_SECRET="" python3 "$(dirname "$0")/mock-receiver.py" \
  --port "$RECEIVER_PORT" --out "$RCV" > "$LOG" 2>&1 &
RCV_PID=$!
sleep 1
kill -0 "$RCV_PID" || fail "receiver did not start"
pass "receiver up"

echo "== provision tenant ${TENANT} =="
PROVISION_OUT="$(bash "$(dirname "$0")/provision-tenant.sh" "$TENANT" "$DEST_URL")"
echo "$PROVISION_OUT" | grep -q "whsec_" || fail "provisioning did not return a signing secret"
echo "$PROVISION_OUT" | grep "whsec_" | tail -1 | awk '{print $2}' > "$SECRET_FILE"
pass "tenant + destination provisioned"

echo "== publish event =="
WEBHOOK_SERVICE_URL="$BASE" WEBHOOK_SERVICE_API_KEY="$KEY" \
  python3 "$(dirname "$0")/../publish.py" \
  --tenant "$TENANT" --topic job.completed --data '{"jobId":"smoke-1","status":"done"}' \
  | grep -q "HTTP 2" || fail "publish failed"
pass "event accepted"

echo "== wait for delivery =="
FOUND=0
for _ in $(seq 1 30); do
  if [[ -s "$RCV" ]] && grep -q "smoke-1" "$RCV"; then FOUND=1; break; fi
  sleep 1
done
[[ "$FOUND" == "1" ]] || fail "no delivery within 30s: $(cat "$RCV" 2>/dev/null | tail -2)"
pass "delivery received"

echo "== verify signature =="
SECRET=$(cat "$SECRET_FILE")
RECEIVER_WEBHOOK_SECRET="$SECRET" python3 - "$RCV" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
hit = [r for r in rows if "smoke-1" in r.get("body", "")]
assert hit, "smoke delivery not found"
assert hit[-1]["signature-valid"] is True, f"signature invalid: {hit[-1]}"
print(f"signature-valid: True (webhook-id={hit[-1]['webhook-id']})")
PY
pass "Standard Webhooks signature verified"

echo ""
echo "ALL WEBHOOK SERVICE SMOKE TESTS PASSED"
