#!/usr/bin/env bash
# Register a webhook destination for a tenant via the platform-service
# facade. Returns the endpoint id and the whsec_ signing secret that the
# receiver needs for Standard Webhooks signature verification.
#
# Usage: scripts/provision-destination.sh <tenant-id> <destination-url> [filter-types] [channels]
# Env:   PLATFORM_SERVICE_URL (default http://localhost:5707)
#        PLATFORM_SERVICE_API_KEY (optional)
set -euo pipefail

TENANT="${1:?usage: provision-destination.sh <tenant-id> <destination-url> [filter-types] [channels]}"
DEST_URL="${2:?usage: provision-destination.sh <tenant-id> <destination-url> [filter-types] [channels]}"
FILTER_TYPES="${3:-gate.passed,job.completed,decision.captured,import.completed,run.published}"
CHANNELS="${4:-}"

BASE="${PLATFORM_SERVICE_URL:-http://localhost:5707}"
AUTH=()
[[ -n "${PLATFORM_SERVICE_API_KEY:-}" ]] && AUTH=(-H "X-Api-Key: ${PLATFORM_SERVICE_API_KEY}")

curl -sf -X POST -H "X-Tenant-Id: ${TENANT}" ${AUTH[@]+"${AUTH[@]}"} \
  -H "Content-Type: application/json" \
  -d "$(python3 -c '
import json, sys
url, filter_types, channels = sys.argv[1:4]
body = {
    "url": url,
    "filterTypes": [v.strip() for v in filter_types.split(",") if v.strip()],
    "description": "provisioned by provision-destination.sh",
}
if channels.strip():
    body["channels"] = [v.strip() for v in channels.split(",") if v.strip()]
print(json.dumps(body))
' "${DEST_URL}" "${FILTER_TYPES}" "${CHANNELS}")" "${BASE}/api/v1/webhooks/destinations"
