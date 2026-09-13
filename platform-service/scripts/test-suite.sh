#!/usr/bin/env bash
# platform-service 综合测试套件 —— TEST-PLAN.md 的可执行形态。
# 用例编号与 TEST-PLAN.md §1 一一对应；输出 PASS/FAIL 汇总，任一 FAIL 时退出码 1。
set -uo pipefail

BASE="${PLATFORM_SERVICE_URL:-http://localhost:5707}"
SPARE="${SPARE_INSTANCE_URL:-http://localhost:5708}"
TS="$(date +%m%d%H%M%S)"
TA="suite-a-$TS"; TB="suite-b-$TS"; TC="suite-c-$TS"
TMP="$(mktemp -d)"
SUITE_DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=0; FAIL=0; FAILED=()

cleanup() {
  [[ -n "${SPARE_PID:-}" ]] && kill -9 "$SPARE_PID" 2>/dev/null
  [[ -n "${RCV_PID:-}" ]] && kill -9 "$RCV_PID" 2>/dev/null
  [[ -n "${RCV2_PID:-}" ]] && kill -9 "$RCV2_PID" 2>/dev/null
  rm -rf "$TMP"
}
trap cleanup EXIT

ok()  { PASS=$((PASS+1)); echo "PASS $1  $2"; }
bad() { FAIL=$((FAIL+1)); FAILED+=("$1"); echo "FAIL $1  $2"; }
jget() { python3 -c "import json,sys;d=json.loads(sys.argv[1]);print(eval(sys.argv[2]))" "$1" "$2" 2>/dev/null; }
code() { curl -s -m 60 -o /dev/null -w "%{http_code}" "$@"; }
hd() { printf -v r 'curl -s'; echo; }

req() { # method path tenant [data] [extra-header: value]... -> body, sets HTTP
  local m="$1" p="$2" t="$3"; shift 3
  local args=(-s -X "$m" -H "X-Tenant-Id: $t")
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == *:* && "$1" != "{"* ]]; then args+=(-H "$1"); shift; else break; fi
  done
  [[ $# -gt 0 ]] && args+=(-d "$1")
  HTTP=$(curl -s -m 60 -o /tmp/suite-body -w "%{http_code}" "${args[@]}" "$BASE/api/v1$p")
  cat /tmp/suite-body
}

echo "=== A. files 模块 · 功能契约 ==="

# F-01
curl -sf -m 60 "$BASE/actuator/health" | grep -q '"UP"' && ok F-01 "health" || bad F-01 "health"

# F-02
printf 'id,name\n1,alpha\n' > "$TMP/f02.csv"
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/f02.csv;type=text/csv" -F "path=a/docs" -F "fileName=f02.csv" "$BASE/api/v1/files/upload")
[[ "$(jget "$B" "d['version']" 2>/dev/null)" == "1" && "$(jget "$B" "d['deduplicated']" 2>/dev/null)" == "False" && $(jget "$B" "len(d['sha256'])" 2>/dev/null || echo 0) -eq 64 ]] \
  && ok F-02 "basic upload receipt" || bad F-02 "got: $B"

# F-03
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/f02.csv;type=text/csv" -F "path=a/docs" -F "fileName=f02.csv" "$BASE/api/v1/files/upload")
[[ "$(jget "$B" "d['deduplicated']" 2>/dev/null)" == "True" && "$(jget "$B" "d['version']" 2>/dev/null)" == "1" ]] \
  && ok F-03 "identical upload deduplicated" || bad F-03 "got: $B"

# F-04
printf 'id,name\n1,alpha\n2,beta\n' > "$TMP/f02.csv"
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/f02.csv;type=text/csv" -F "path=a/docs" -F "fileName=f02.csv" "$BASE/api/v1/files/upload")
[[ "$(jget "$B" "d['version']" 2>/dev/null)" == "2" ]] && ok F-04 "version append" || bad F-04 "got: $B"
U2=$(jget "$B" "d['uuid']")
U1=$(docker exec file-service-pg psql -U document -d postgres -t -A -c \
  "select uuid from file_objects where tenant_id='$TA' and filename='f02.csv' and version=1 limit 1" 2>/dev/null)

# F-05
printf 'orig\n' > "$TMP/orig-name.txt"
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/orig-name.txt" -F "path=a/etc" "$BASE/api/v1/files/upload")
[[ "$(jget "$B" "d['filename']" 2>/dev/null)" == "orig-name.txt" ]] && ok F-05 "default fileName from original" || bad F-05 "got: $B"

# F-06
printf 'unicode\n' > "$TMP/数据 表格.txt"
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/数据 表格.txt" -F "path=a/数据" "$BASE/api/v1/files/upload")
UC=$(jget "$B" "d['uuid']")
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$UC")
[[ "$BODY" == "unicode" && "$(jget "$B" "d['filename']")" == "数据 表格.txt" ]] \
  && ok F-06 "unicode filename roundtrip via uuid" || bad F-06 "mismatch: $BODY"

# F-07
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/orig-name.txt" -F "path=a/etc" -F "fileName=meta.json" -F 'meta={"k1":"v1","n":2}' "$BASE/api/v1/files/upload")
M=$(jget "$B" "d['meta']" 2>/dev/null)
[[ "$M" == *'"k1":"v1"'* && "$M" == *'"n":2'* ]] && ok F-07 "meta preserved verbatim (jsonb storage, string in receipt)" || bad F-07 "got meta=$M"

# F-08
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/orig-name.txt" -F "path=a/etc" -F "fileName=cat.txt" -F "fileCategory=report" -F "usage=import" "$BASE/api/v1/files/upload")
[[ "$(jget "$B" "d['fileCategory']" 2>/dev/null)" == "report" && "$(jget "$B" "d['usage']" 2>/dev/null)" == "import" ]] \
  && ok F-08 "category/usage stored" || bad F-08 "got: $B"

# F-09
: > "$TMP/empty.csv"
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/empty.csv" -F "path=a/etc" "$BASE/api/v1/files/upload")
[[ "$C" == "400" ]] && ok F-09 "empty file rejected 400" || bad F-09 "got $C"

# F-10
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -X POST -H "X-Tenant-Id: $TA" -F "path=a/etc" "$BASE/api/v1/files/upload")
[[ "$C" == "400" ]] && ok F-10 "missing file part 400" || bad F-10 "got $C"

# F-11
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/f02.csv;type=text/csv" -F "path=a//docs///" -F "fileName=norm.csv" "$BASE/api/v1/files/upload")
P=$(jget "$B" "d['path']" 2>/dev/null)
[[ "$P" == "a/docs" ]] && ok F-11 "path normalized" || bad F-11 "got path=$P"
B2=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/f02.csv;type=text/csv" -F "path=a/docs" -F "fileName=norm.csv" "$BASE/api/v1/files/upload")
[[ "$(jget "$B2" "d['deduplicated']" 2>/dev/null)" == "True" ]] && ok F-11b "normalized path dedupes" || bad F-11b "got: $B2"

# F-12
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/f02.csv;type=text/csv" -F "path=../escape" "$BASE/api/v1/files/upload")
[[ "$C" == "400" ]] && ok F-12 "path traversal rejected" || bad F-12 "got $C"

# F-13
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/orig-name.txt" -F "path=a/etc" -F "fileName=we/ird.txt" "$BASE/api/v1/files/upload")
[[ "$(jget "$B" "d['filename']" 2>/dev/null)" == "we_ird.txt" ]] && ok F-13 "filename sanitized" || bad F-13 "got: $B"

# F-14 / F-15 / F-16 / F-17 / F-18
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$U2")
echo "$BODY" | grep -q "beta" && ok F-14 "download latest (v2)" || bad F-14 "got: $BODY"
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$U1")
[[ "$BODY" == "$(printf 'id,name\n1,alpha\n')" ]] && ok F-15 "download v1 intact" || bad F-15 "got: $BODY"
BOM=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$U1?bom=true" | xxd -p -l 3)
[[ "$BOM" == "efbbbf" ]] && ok F-16a "BOM prepended" || bad F-16a "got $BOM"
printf '\xef\xbb\xbfalready\n' > "$TMP/bom.csv"
curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/bom.csv" -F "path=a/docs" -F "fileName=bom.csv" "$BASE/api/v1/files/upload" >/dev/null
UB=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -H "Content-Type: application/json" -d '{"uuid":"'"$UB"'"}' "$BASE/api/v1/files/presign" >/dev/null 2>&1; echo)
UB=$(docker exec file-service-pg psql -U document -d postgres -t -A -c "select uuid from file_objects where tenant_id='$TA' and filename='bom.csv' order by id desc limit 1")
BOM2=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$UB?bom=true" | xxd -p -l 6)
[[ "$BOM2" == "efbbbf61"* ]] && ok F-16b "no double BOM" || bad F-16b "got $BOM2"
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/00000000000000000000000000000000")
[[ "$C" == "404" ]] && ok F-17 "404 on missing" || bad F-17 "got $C"
HDR=$(curl -s -m 60 -D - -o /dev/null -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$U2")
echo "$HDR" | grep -qi "content-type: text/csv" && echo "$HDR" | grep -qi "filename\*=utf-8" \
  && ok F-18 "headers preserved" || bad F-18 "headers: $(echo "$HDR" | grep -i 'content-type\|disposition')"

# F-19 / F-20
B=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files?prefix=a")
echo "$B" | grep -q '"docs"' && ok F-19a "prefix shows subdir" || bad F-19a "got: $B"
B=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files?prefix=a/docs")
N=$(echo "$B" | python3 -c "import json,sys;print(len(json.load(sys.stdin)['files']))")
[[ "$N" -ge 3 ]] && ok F-19b "prefix lists files" || bad F-19b "files=$N"
B=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files")
echo "$B" | grep -q '"a"' && ok F-20 "root listing" || bad F-20 "got: $(echo "$B" | head -c 120)"

# F-22 / F-23
B=$(curl -s -m 60 -X DELETE -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$U1")
[[ "$(jget "$B" "d['deleted']")" == "1" ]] && ok F-22a "delete single version" || bad F-22a "got: $B"
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$U1")
[[ "$C" == "404" ]] && ok F-22b "deleted version 404" || bad F-22b "got $C"
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$U2")
echo "$BODY" | grep -q beta && ok F-23 "other version survives" || bad F-23 "got: $BODY"


# F-26 / F-27 (S3-style PUT + HEAD)
printf 'put-stream\n' > "$TMP/put.bin"
PU=$(python3 -c "import uuid;print(uuid.uuid4().hex)")
B=$(curl -s -m 60 -X PUT -H "X-Tenant-Id: $TA" -H "Content-Type: application/octet-stream" \
  -H "X-File-Path: a/stream" -H "X-File-Name: put.bin" -H "X-File-Category: raw" \
  --data-binary "@$TMP/put.bin" "$BASE/api/v1/files/$PU")
[[ "$(jget "$B" "d['filename']")" == "put.bin" ]] && ok F-26 "PUT raw upload at chosen uuid" || bad F-26 "got: $B"
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$PU")
[[ "$BODY" == "put-stream" ]] && ok F-26b "PUT content readable" || bad F-26b "got: $BODY"
HDR=$(curl -s -I -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$U2")
echo "$HDR" | grep -qi "^etag:" && echo "$HDR" | grep -qi "x-file-version: 2" \
  && ok F-27 "HEAD metadata headers" || bad F-27 "headers: $(echo "$HDR" | head -4 | tr '\n' ' ')"

echo "=== L. 获取下载链接 (POST /files/presign) ==="

# L-01 auto + internal storage (self-hosted MinIO) → our own download URL
FILE_SERVICE_PORT=5708 FILE_LINK_MODE=auto \
  OBJECT_STORAGE_ENDPOINT="http://document-minio:9000" OBJECT_STORAGE_FORCE_PATH_STYLE=true \
  SPRING_DATASOURCE_URL="jdbc:postgresql://localhost:5433/postgres?stringtype=unspecified" \
  SPRING_DATASOURCE_USERNAME=document SPRING_DATASOURCE_PASSWORD=document \
  SVIX_SERVER_URL="http://localhost:8071" \
  nohup java -jar "$(cd "$SUITE_DIR/.." && pwd)/build/libs/platform-service.jar" > /tmp/suite-direct.log 2>&1 &
SPARE_PID=$!
POK=0
for _ in $(seq 1 45); do curl -sf -m 5 "$SPARE/actuator/health" >/dev/null && POK=1 && break; sleep 1; done
if [[ "$POK" == "1" ]]; then
  B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -H "Content-Type: application/json" \
    -d "{\"uuid\":\"$U2\",\"ttlSeconds\":300}" "$SPARE/api/v1/files/presign")
  K=$(jget "$B" "d['kind']"); U=$(jget "$B" "d['url']")
  [[ "$K" == "direct" && "$U" == *"/api/v1/files/$U2"* ]] \
    && ok L-01 "auto: internal storage → own download url" || bad L-01 "kind=$K url=$U"
else
  bad L-01 "internal-mode instance failed to start"
fi
kill -9 "$SPARE_PID" 2>/dev/null; wait "$SPARE_PID" 2>/dev/null; SPARE_PID=""

# L-02 validation
C=$(curl -s -o /dev/null -w "%{http_code}" -m 60 -X POST -H "X-Tenant-Id: $TA" -H "Content-Type: application/json" \
  -d '{}' "$BASE/api/v1/files/presign")
[[ "$C" == "400" ]] && ok L-02 "missing uuid 400" || bad L-02 "got $C"

# L-03 auto + reachable storage (TOS) → storage presigned URL
FILE_SERVICE_PORT=5708 FILE_LINK_MODE=auto \
  OBJECT_STORAGE_ENDPOINT="https://tos-s3-cn-beijing.volces.com" \
  OBJECT_STORAGE_BUCKET="finademo" OBJECT_STORAGE_FORCE_PATH_STYLE=false \
  SPRING_DATASOURCE_URL="jdbc:postgresql://localhost:5433/postgres?stringtype=unspecified" \
  SPRING_DATASOURCE_USERNAME=document SPRING_DATASOURCE_PASSWORD=document \
  SVIX_SERVER_URL="http://localhost:8071" \
  nohup java -jar "$(cd "$SUITE_DIR/.." && pwd)/build/libs/platform-service.jar" > /tmp/suite-presign.log 2>&1 &
SPARE_PID=$!
POK=0
for _ in $(seq 1 45); do curl -sf -m 5 "$SPARE/actuator/health" >/dev/null && POK=1 && break; sleep 1; done
if [[ "$POK" == "1" ]]; then
  B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -H "Content-Type: application/json" \
    -d "{\"uuid\":\"$U2\",\"ttlSeconds\":300}" "$SPARE/api/v1/files/presign")
  K=$(jget "$B" "d['kind']"); U=$(jget "$B" "d['url']")
  [[ "$K" == "presigned" && "$U" == *"X-Amz-Signature"* ]] \
    && ok L-03 "auto: reachable storage → presigned url" || bad L-03 "kind=$K url=${U:0:80}"
else
  bad L-03 "presign instance failed to start"
fi
kill -9 "$SPARE_PID" 2>/dev/null; wait "$SPARE_PID" 2>/dev/null; SPARE_PID=""

echo "=== D. Portal ==="
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" "$BASE/portal")
[[ "$C" == "200" ]] && ok P-01 "portal forward 200" || bad P-01 "got $C"
curl -s -m 60 "$BASE/portal/index.html" | grep -q "投递目标" && ok P-02 "portal content" || bad P-02 "content missing"
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" "$BASE/portal/index.html")
[[ "$C" == "200" ]] && ok P-03 "static shell without tenant" || bad P-03 "got $C"

echo "=== E. 韧性 ==="

# S-01 svix down
docker stop svix-server >/dev/null
sleep 1
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -X POST -H "X-Tenant-Id: $TC" -H "Content-Type: application/json" \
  -d '{"topic":"job.completed","data":{"jobId":"s01"}}' "$BASE/api/v1/webhooks/publish")
[[ "$C" -ge 500 ]] && ok S-01a "svix down surfaces 5xx ($C)" || bad S-01a "got $C"
docker start svix-server >/dev/null && sleep 6
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TC" -H "Content-Type: application/json" \
  -d '{"topic":"job.completed","data":{"jobId":"s01-recovered"}}' "$BASE/api/v1/webhooks/publish")
jget "$B" "d['messageId']" >/dev/null 2>&1 && ok S-01b "recovers after svix restart" || bad S-01b "got: $B"

# S-02 restart persistence
pkill -f "platform-service.jar" 2>/dev/null; sleep 3
( cd "$SUITE_DIR/.." && set -a && . "$SUITE_DIR/../../document_service/.env" && set +a && SPRING_DATASOURCE_URL="jdbc:postgresql://localhost:5433/postgres?stringtype=unspecified" \
  SPRING_DATASOURCE_USERNAME=document SPRING_DATASOURCE_PASSWORD=document \
  SVIX_SERVER_URL="http://localhost:8071" FILE_LINK_MODE="${FILE_LINK_MODE:-ticket}" \
  nohup java -jar build/libs/platform-service.jar > /tmp/platform-service.log 2>&1 & )
PERSIST_OK=0
for _ in $(seq 1 30); do curl -sf -m 60 "$BASE/actuator/health" >/dev/null && { PERSIST_OK=1; break; }; sleep 1; done
if [[ "$PERSIST_OK" == "1" ]]; then
  BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/$U2")
  echo "$BODY" | grep -q beta && ok S-02 "state survives service restart" || bad S-02 "file lost after restart"
else
  bad S-02 "service did not restart (needs TOS env in shell)"
fi

# S-03 compose
( cd "$SUITE_DIR/../.." && SPRING_DATASOURCE_PASSWORD=x SVIX_JWT_SECRET=x docker compose config --quiet ) 2>/dev/null \
  && ok S-03 "compose config valid" || bad S-03 "compose invalid"

echo ""
echo "=============================="
echo "SUITE RESULT: PASS=$PASS FAIL=$FAIL"
[[ ${#FAILED[@]} -gt 0 ]] && printf 'FAILED CASES: %s\n' "${FAILED[*]}"
[[ "$FAIL" -eq 0 ]] && echo "ALL TEST CASES PASSED"
