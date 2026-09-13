#!/usr/bin/env bash
# Provision a tenant on the Outpost webhook service: create tenant,
# register a webhook destination with a signing secret, and print a
# tenant portal link.
#
# Usage: scripts/provision-tenant.sh <tenant-id> <destination-url> [topics]
# Env:   WEBHOOK_SERVICE_URL (default http://localhost:5708)
#        WEBHOOK_SERVICE_API_KEY (default webhook-demo-key)
set -euo pipefail

TENANT="${1:?usage: provision-tenant.sh <tenant-id> <destination-url> [topics]}"
DEST_URL="${2:?usage: provision-tenant.sh <tenant-id> <destination-url> [topics]}"
TOPICS="${3:-gate.passed,job.completed,decision.captured,import.completed,run.published}"

BASE="${WEBHOOK_SERVICE_URL:-http://localhost:5708}"
KEY="${WEBHOOK_SERVICE_API_KEY:-webhook-demo-key}"
AUTH=(-H "Authorization: Bearer ${KEY}" -H "Content-Type: application/json")

# ITOA64 secret in whsec_ format for Standard Webhooks signing
SECRET="whsec_$(openssl rand -base64 24)"

echo "== create/update tenant ${TENANT} =="
curl -sf -X PUT "${AUTH[@]}" "${BASE}/api/v1/tenants/${TENANT}" -d '{}' | head -c 400; echo

echo "== register destination =="
curl -sf -X POST "${AUTH[@]}" "${BASE}/api/v1/tenants/${TENANT}/destinations" -d "{
  \"type\": \"webhook\",
  \"topics\": $(python3 -c "import json,sys;print(json.dumps(sys.argv[1].split(',')))" "${TOPICS}"),
  \"config\": {\"url\": \"${DEST_URL}\"},
  \"credentials\": {\"secret\": \"${SECRET}\"}
}" | head -c 600; echo

echo "== portal link =="
curl -sf "${AUTH[@]}" "${BASE}/api/v1/tenants/${TENANT}/portal" | head -c 400; echo

echo ""
echo "destination signing secret (give to the receiver for verification):"
echo "  ${SECRET}"
