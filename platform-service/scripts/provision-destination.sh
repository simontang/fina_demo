#!/usr/bin/env bash
# Register a webhook destination for a tenant via the platform-service
# facade. Returns the endpoint id and the whsec_ signing secret that the
# receiver needs for Standard Webhooks signature verification.
#
# Usage: scripts/provision-destination.sh <tenant-id> <destination-url> [topics]
# Env:   PLATFORM_SERVICE_URL (default http://localhost:5707)
#        PLATFORM_SERVICE_API_KEY (optional)
set -euo pipefail

TENANT="${1:?usage: provision-destination.sh <tenant-id> <destination-url> [topics]}"
DEST_URL="${2:?usage: provision-destination.sh <tenant-id> <destination-url> [topics]}"
TOPICS="${3:-gate.passed,job.completed,decision.captured,import.completed,run.published}"

BASE="${PLATFORM_SERVICE_URL:-http://localhost:5707}"
AUTH=()
[[ -n "${PLATFORM_SERVICE_API_KEY:-}" ]] && AUTH=(-H "X-Api-Key: ${PLATFORM_SERVICE_API_KEY}")

curl -sf -X POST -H "X-Tenant-Id: ${TENANT}" "${AUTH[@]}" \
  -H "Content-Type: application/json" \
  -d "{
    \"url\": \"${DEST_URL}\",
    \"topics\": $(python3 -c "import json,sys;print(json.dumps(sys.argv[1].split(',')))" "${TOPICS}"),
    \"description\": \"provisioned by provision-destination.sh\"
  }" "${BASE}/api/v1/webhooks/destinations"
