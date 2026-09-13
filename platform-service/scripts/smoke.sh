#!/usr/bin/env bash
# File Service end-to-end smoke test.
# Usage: FILE_SERVICE_URL=http://localhost:5707 [FILE_SERVICE_API_KEY=...] scripts/smoke.sh
set -euo pipefail

BASE="${FILE_SERVICE_URL:-http://localhost:5707}"
# Fresh tenants per run — the smoke assumes empty per-tenant state.
TENANT_A="${TENANT_A:-smoke-a-$(date +%m%d%H%M%S)}"
TENANT_B="${TENANT_B:-smoke-b-$(date +%m%d%H%M%S)}"
AUTH=()
if [[ -n "${FILE_SERVICE_API_KEY:-}" ]]; then
  AUTH=(-H "X-Api-Key: ${FILE_SERVICE_API_KEY}")
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1" >&2; exit 1; }

jqget() { python3 -c "import json,sys;d=json.load(sys.stdin);print(eval(sys.argv[1]))" "$2" 2>/dev/null; }

echo "== health =="
curl -sf "${BASE}/actuator/health" | grep -q '"UP"' || fail "health"
pass "health"

V1_CONTENT='name,amount\nfoo,1\n'
V2_CONTENT='name,amount\nfoo,1\nbar,2\n'
printf 'name,amount\nfoo,1\n' > "$TMP/data.csv"

echo "== upload v1 (${TENANT_A:-tenant-a}) =="
R=$(curl -sf -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} \
  -F "file=@${TMP}/data.csv" -F "path=demo/docs" -F "fileCategory=smoke" -F "usage=import" \
  "${BASE}/api/v1/files/upload")
V=$(echo "$R" | jqget - "d['version']")
D=$(echo "$R" | jqget - "d['deduplicated']")
[[ "$V" == "1" && "$D" == "False" ]] || fail "upload v1 got version=$V deduplicated=$D"
pass "upload v1 version=1"

echo "== identical re-upload is deduplicated, version stays 1 =="
R=$(curl -sf -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} \
  -F "file=@${TMP}/data.csv" -F "path=demo/docs" \
  "${BASE}/api/v1/files/upload")
V=$(echo "$R" | jqget - "d['version']")
D=$(echo "$R" | jqget - "d['deduplicated']")
[[ "$V" == "1" && "$D" == "True" ]] || fail "dedupe got version=$V deduplicated=$D"
pass "identical upload deduplicated"

echo "== modified content appends version 2, old row untouched =="
printf 'name,amount\nfoo,1\nbar,2\n' > "$TMP/data.csv"
R=$(curl -sf -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} \
  -F "file=@${TMP}/data.csv" -F "path=demo/docs" \
  "${BASE}/api/v1/files/upload")
V=$(echo "$R" | jqget - "d['version']")
[[ "$V" == "2" ]] || fail "expected version 2, got $V"
pass "re-upload appends version 2"
U2=$(echo "$R" | jqget - "d['uuid']")
U1=$(docker exec file-service-pg psql -U document -d postgres -t -A -c \
  "select uuid from file_objects where tenant_id='${TENANT_A}' and filename='data.csv' and version=1 limit 1" 2>/dev/null)

echo "== download by path returns latest version =="
BODY=$(curl -sf -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} "${BASE}/api/v1/files/$U2/download")
echo "$BODY" | grep -q "bar,2" || fail "latest download missing v2 content"
echo "$BODY" | grep -q "foo,1" || fail "latest download missing base content"
pass "uuid download returns v2"

echo "== download specific version 1 =="
BODY=$(curl -sf -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} "${BASE}/api/v1/files/$U1/download")
[[ "$BODY" == "$(printf 'name,amount\nfoo,1\n')" ]] || fail "version=1 content mismatch: $BODY"
pass "v1 uuid still serves v1 (immutable)"

echo "== bom=true prepends UTF-8 BOM =="
BOM=$(curl -sf -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} \
  "${BASE}/api/v1/files/$U1/download?bom=true" | xxd -p -l 3)
[[ "$BOM" == "efbbbf" ]] || fail "expected BOM efbbbf, got $BOM"
pass "bom=true prepends UTF-8 BOM"

echo "== pseudo-directory listing =="
R=$(curl -sf -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} "${BASE}/api/v1/files?prefix=demo")
echo "$R" | grep -q '"docs"' || fail "listing should contain docs dir: $R"
R=$(curl -sf -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} "${BASE}/api/v1/files?prefix=demo/docs")
COUNT=$(echo "$R" | python3 -c "import json,sys;print(len(json.load(sys.stdin)['files']))")
[[ "$COUNT" -ge 2 ]] || fail "expected >=2 files under demo/docs, got $COUNT"
pass "prefix listing works without folder entities"

echo "== cross-tenant isolation: same path in TENANT_B =="
printf 'tenant,b\nonly-b,9\n' > "$TMP/data.csv"
R=$(curl -sf -H "X-Tenant-Id: ${TENANT_B:-tenant-b}" ${AUTH[@]+"${AUTH[@]}"} \
  -F "file=@${TMP}/data.csv" -F "path=demo/docs" \
  "${BASE}/api/v1/files/upload")
V=$(echo "$R" | jqget - "d['version']")
[[ "$V" == "1" ]] || fail "tenant-b expected own version 1, got $V"
UB=$(docker exec file-service-pg psql -U document -d postgres -t -A -c \
  "select uuid from file_objects where tenant_id='${TENANT_B}' and filename='data.csv' order by id desc limit 1" 2>/dev/null)
BODY=$(curl -sf -H "X-Tenant-Id: ${TENANT_B:-tenant-b}" ${AUTH[@]+"${AUTH[@]}"} "${BASE}/api/v1/files/$UB/download")
echo "$BODY" | grep -q "only-b" || fail "tenant-b should read its own content"
C=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} "${BASE}/api/v1/files/$UB/download")
[[ "$C" == "404" ]] || fail "tenant-a reached tenant-b object (got $C)"
pass "tenants isolated (same path, distinct uuids)"

echo "== HEAD metadata =="
HDR=$(curl -s -I -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} "${BASE}/api/v1/files/$U2")
echo "$HDR" | grep -qi "^etag:" || fail "HEAD missing ETag"
pass "HEAD metadata (ETag)"

echo "== presign (download url per storage reachability) =="
B=$(curl -sf -X POST -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} \
  -H "Content-Type: application/json" -d "{\"uuid\":\"$U2\",\"ttlSeconds\":300}" \
  "${BASE}/api/v1/files/presign")
K=$(echo "$B" | jqget - "d['kind']")
U=$(echo "$B" | jqget - "d['url']")
case "$K" in
  direct)
    curl -sf -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} "$U" | grep -q foo \
      || fail "direct url download failed"
    pass "presign → own download url (internal storage)" ;;
  presigned)
    echo "$U" | grep -q "X-Amz-Signature" || fail "presigned url missing signature"
    pass "presign → storage presigned url (reachable storage)" ;;
  *) fail "unexpected kind=$K" ;;
esac

echo "== soft delete (uuid pins one version) =="
R=$(curl -sf -X DELETE -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} "${BASE}/api/v1/files/$U2")
N=$(echo "$R" | jqget - "d['deleted']")
[[ "$N" == "1" ]] || fail "expected 1 soft-deleted row, got $N"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} \
  "${BASE}/api/v1/files/$U2/download")
[[ "$CODE" == "404" ]] || fail "expected 404 after delete, got $CODE"
BODY=$(curl -sf -H "X-Tenant-Id: ${TENANT_A:-tenant-a}" ${AUTH[@]+"${AUTH[@]}"} "${BASE}/api/v1/files/$U1/download")
echo "$BODY" | grep -q foo || fail "sibling version lost after delete"
pass "soft delete + sibling version intact"

echo ""
echo "ALL FILE SERVICE SMOKE TESTS PASSED"
