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

# F-05
printf 'orig\n' > "$TMP/orig-name.txt"
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/orig-name.txt" -F "path=a/etc" "$BASE/api/v1/files/upload")
[[ "$(jget "$B" "d['filename']" 2>/dev/null)" == "orig-name.txt" ]] && ok F-05 "default fileName from original" || bad F-05 "got: $B"

# F-06
printf 'unicode\n' > "$TMP/数据 表格.txt"
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/数据 表格.txt" -F "path=a/数据" "$BASE/api/v1/files/upload")
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/download?path=a%2F%E6%95%B0%E6%8D%AE%2F%E6%95%B0%E6%8D%AE%20%E8%A1%A8%E6%A0%BC.txt")
[[ "$BODY" == "unicode" ]] && ok F-06 "unicode filename roundtrip" || bad F-06 "download mismatch: $BODY"

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
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/download?path=a%2Fdocs%2Ff02.csv")
echo "$BODY" | grep -q "beta" && ok F-14 "download latest (v2)" || bad F-14 "got: $BODY"
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/download?path=a%2Fdocs%2Ff02.csv&version=1")
[[ "$BODY" == "$(printf 'id,name\n1,alpha\n')" ]] && ok F-15 "download v1 intact" || bad F-15 "got: $BODY"
BOM=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/download?path=a%2Fdocs%2Ff02.csv&version=1&bom=true" | xxd -p -l 3)
[[ "$BOM" == "efbbbf" ]] && ok F-16a "BOM prepended" || bad F-16a "got $BOM"
printf '\xef\xbb\xbfalready\n' > "$TMP/bom.csv"
curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -F "file=@$TMP/bom.csv" -F "path=a/docs" -F "fileName=bom.csv" "$BASE/api/v1/files/upload" >/dev/null
BOM2=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/download?path=a%2Fdocs%2Fbom.csv&bom=true" | xxd -p -l 6)
[[ "$BOM2" == "efbbbf61"* ]] && ok F-16b "no double BOM" || bad F-16b "got $BOM2"
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/download?path=a%2Fdocs%2Fnope.csv")
[[ "$C" == "404" ]] && ok F-17 "404 on missing" || bad F-17 "got $C"
HDR=$(curl -s -m 60 -D - -o /dev/null -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/download?path=a%2Fdocs%2Ff02.csv")
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

# F-21
B=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/receipt?path=a%2Fdocs%2Ff02.csv")
SHA=$(jget "$B" "d['sha256']"); U=$(jget "$B" "d['uuid']")
B1=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/uuid/$U/receipt")
NOID=$(echo "$B" | python3 -c "import json,sys;print('id' in json.load(sys.stdin))")
[[ "$(jget "$B1" "d['sha256']")" == "$SHA" && "$NOID" == "False" ]] \
  && ok F-21 "receipt by path/uuid consistent, no id exposed" || bad F-21 "mismatch (noid=$NOID)"

# F-22 / F-23
B=$(curl -s -m 60 -X DELETE -H "X-Tenant-Id: $TA" "$BASE/api/v1/files?path=a%2Fdocs%2Ff02.csv&version=1")
[[ "$(jget "$B" "d['deleted']")" == "1" ]] && ok F-22a "delete single version" || bad F-22a "got: $B"
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/download?path=a%2Fdocs%2Ff02.csv&version=1")
[[ "$C" == "404" ]] && ok F-22b "deleted version 404" || bad F-22b "got $C"
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/download?path=a%2Fdocs%2Ff02.csv")
echo "$BODY" | grep -q beta && ok F-23 "other version survives" || bad F-23 "got: $BODY"

# F-24
B=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/receipt?path=a%2Fdocs%2Ff02.csv")
U=$(jget "$B" "d['uuid']")
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/uuid/$U/download")
echo "$BODY" | grep -q beta && ok F-24 "uuid download" || bad F-24 "got: $BODY"

echo "=== B. 多租户与鉴权 ==="

# T-01
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" "$BASE/api/v1/files?prefix=")
[[ "$C" == "400" ]] && ok T-01 "missing tenant 400" || bad T-01 "got $C"

# T-02
printf 'tenant-b-only\n' > "$TMP/b.csv"
curl -s -m 60 -X POST -H "X-Tenant-Id: $TB" -F "file=@$TMP/b.csv" -F "path=a/docs" -F "fileName=f02.csv" "$BASE/api/v1/files/upload" >/dev/null
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TB" "$BASE/api/v1/files/download?path=a%2Fdocs%2Ff02.csv")
echo "$BODY" | grep -q "tenant-b-only" || bad T-02a "tenant-b should see own"
BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/download?path=a%2Fdocs%2Ff02.csv")
echo "$BODY" | grep -q "tenant-b-only" && bad T-02 "leak!" || ok T-02 "cross-tenant isolation"

# T-03
RB=$(curl -s -m 60 -H "X-Tenant-Id: $TB" "$BASE/api/v1/files/receipt?path=a%2Fdocs%2Ff02.csv")
U=$(jget "$RB" "d['uuid']")
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/uuid/$U/receipt")
[[ "$C" == "404" ]] && ok T-03 "cross-tenant uuid access 404" || bad T-03 "got $C"

# T-04
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TA" -H "X-User-Id: user-42" -F "file=@$TMP/orig-name.txt" -F "path=a/etc" -F "fileName=u42.txt" "$BASE/api/v1/files/upload")
[[ "$(jget "$B" "d['createdBy']" 2>/dev/null)" == "user-42" ]] && ok T-04 "created_by from X-User-Id" || bad T-04 "got: $B"

# T-05 (spare instance with api key)
FILE_SERVICE_API_KEY=sk-suite FILE_SERVICE_PORT=5708 SPRING_DATASOURCE_URL="jdbc:postgresql://localhost:5433/postgres?stringtype=unspecified" \
  SPRING_DATASOURCE_USERNAME=document SPRING_DATASOURCE_PASSWORD=document SVIX_SERVER_URL="http://localhost:8071" \
  nohup java -jar "$(cd "$SUITE_DIR/.." && pwd)/build/libs/platform-service.jar" > /tmp/suite-spare.log 2>&1 &
SPARE_PID=$!
SPARE_OK=0
for _ in $(seq 1 25); do curl -sf -m 60 "$SPARE/actuator/health" >/dev/null && SPARE_OK=1 && break; sleep 1; done
if [[ "$SPARE_OK" == "1" ]]; then
  C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -H "X-Tenant-Id: t" "$SPARE/api/v1/files?prefix=")
  [[ "$C" == "401" ]] && ok T-05a "missing api key 401" || bad T-05a "got $C"
  C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -H "X-Api-Key: wrong" -H "X-Tenant-Id: t" "$SPARE/api/v1/files?prefix=")
  [[ "$C" == "401" ]] && ok T-05b "wrong api key 401" || bad T-05b "got $C"
  C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -H "X-Api-Key: sk-suite" "$SPARE/api/v1/files?prefix=")
  [[ "$C" == "400" ]] && ok T-05c "valid key, tenant still required" || bad T-05c "got $C"
  C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -H "X-Api-Key: sk-suite" -H "X-Tenant-Id: t" "$SPARE/api/v1/files?prefix=")
  [[ "$C" == "200" ]] && ok T-05d "valid key + tenant 200" || bad T-05d "got $C"
else
  bad T-05 "spare instance failed to start"
fi
kill -9 "$SPARE_PID" 2>/dev/null; wait "$SPARE_PID" 2>/dev/null; SPARE_PID=""

# T-07 (spare instance with default tenant)
FILE_SERVICE_DEFAULT_TENANT=dev-default FILE_SERVICE_PORT=5708 SPRING_DATASOURCE_URL="jdbc:postgresql://localhost:5433/postgres?stringtype=unspecified" \
  SPRING_DATASOURCE_USERNAME=document SPRING_DATASOURCE_PASSWORD=document SVIX_SERVER_URL="http://localhost:8071" \
  nohup java -jar "$(cd "$SUITE_DIR/.." && pwd)/build/libs/platform-service.jar" > /tmp/suite-spare2.log 2>&1 &
SPARE_PID=$!
SPARE_OK=0
for _ in $(seq 1 25); do curl -sf -m 60 "$SPARE/actuator/health" >/dev/null && SPARE_OK=1 && break; sleep 1; done
if [[ "$SPARE_OK" == "1" ]]; then
  B=$(curl -s -m 60 -X POST -F "file=@$TMP/orig-name.txt" -F "path=dev" "$SPARE/api/v1/files/upload")
  [[ "$(jget "$B" "d['filename']" 2>/dev/null)" == "orig-name.txt" ]] || bad T-07 "upload failed: $B"
  T7=$(docker exec file-service-pg psql -U document -d postgres -t -A -c \
    "select tenant_id from file_objects where filename='orig-name.txt' and path='dev' order by id desc limit 1" 2>/dev/null)
  [[ "$T7" == "dev-default" ]] && ok T-07 "default tenant applied (verified in DB)" || bad T-07 "tenant=$T7"
else
  bad T-07 "spare instance failed to start"
fi
kill -9 "$SPARE_PID" 2>/dev/null; wait "$SPARE_PID" 2>/dev/null; SPARE_PID=""

echo "=== C. webhooks 模块 ==="

# W-01 / W-02
python3 "$SUITE_DIR/mock-receiver.py" --port 5909 --out "$TMP/rcv1.log" > /dev/null 2>&1 &
RCV_PID=$!; sleep 1
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TC" -H "Content-Type: application/json" \
  -d "{\"url\":\"http://host.docker.internal:5909/catch\",\"topics\":[\"job.completed\"],\"description\":\"suite\"}" \
  "$BASE/api/v1/webhooks/destinations")
SECRET=$(jget "$B" "d['secret']"); EP=$(jget "$B" "d['endpointId']")
[[ "$SECRET" == whsec_* && -n "$EP" ]] && ok W-01 "destination created with whsec" || bad W-01 "got: $B"
B=$(curl -s -m 60 -H "X-Tenant-Id: $TC" "$BASE/api/v1/webhooks/destinations")
echo "$B" | grep -q "$EP" && echo "$B" | grep -q "job.completed" && ok W-02 "destinations list" || bad W-02 "got: $B"

# W-04 / W-08 / W-05
kill -9 "$RCV_PID" 2>/dev/null; wait "$RCV_PID" 2>/dev/null
RECEIVER_WEBHOOK_SECRET="$SECRET" python3 "$SUITE_DIR/mock-receiver.py" --port 5909 --out "$TMP/rcv1.log" > /dev/null 2>&1 &
RCV_PID=$!; sleep 1
B=$(curl -s -m 60 -X POST -H "X-Tenant-Id: $TC" -H "Content-Type: application/json" \
  -d '{"topic":"job.completed","data":{"jobId":"w04"}}' "$BASE/api/v1/webhooks/publish")
M1=$(jget "$B" "d['messageId']")
[[ -n "$M1" ]] && ok W-04 "publish + lazy event type" || bad W-04 "got: $B"
FOUND=0; T0=$(date +%s)
for _ in $(seq 1 30); do
  grep -q "w04" "$TMP/rcv1.log" 2>/dev/null && { FOUND=1; break; }; sleep 1
done
DT=$(( $(date +%s) - T0 ))
if [[ "$FOUND" == "1" ]]; then
  V=$(python3 -c "import json,sys;rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()];hit=[r for r in rows if 'w04' in r['body']];print(hit[-1]['signature-valid'])" "$TMP/rcv1.log")
  [[ "$V" == "True" && "$DT" -le 30 ]] && ok W-05 "delivered in ${DT}s, signature valid" || bad W-05 "sig=$V dt=${DT}s"
else
  bad W-05 "no delivery in 30s"
fi

# W-09 tampered signature (receiver with wrong secret)
printf 'wrong' | base64 > /tmp/wrong.b64
WRONG="whsec_$(cat /tmp/wrong.b64)"
RECEIVER_WEBHOOK_SECRET="$WRONG" python3 "$SUITE_DIR/mock-receiver.py" --port 5910 --out "$TMP/rcv2.log" > /dev/null 2>&1 &
RCV2_PID=$!; sleep 1
curl -s -m 60 -X POST -H "X-Tenant-Id: $TC" -H "Content-Type: application/json" \
  -d "{\"url\":\"http://host.docker.internal:5910/catch\",\"topics\":[\"gate.passed\"],\"description\":\"tamper-test\"}" \
  "$BASE/api/v1/webhooks/destinations" >/dev/null
curl -s -m 60 -X POST -H "X-Tenant-Id: $TC" -H "Content-Type: application/json" \
  -d '{"topic":"gate.passed","data":{"marker":"w09"}}' "$BASE/api/v1/webhooks/publish" >/dev/null
T9=0
for _ in $(seq 1 30); do grep -q "w09" "$TMP/rcv2.log" 2>/dev/null && { T9=1; break; }; sleep 1; done
if [[ "$T9" == "1" ]]; then
  V=$(python3 -c "import json,sys;rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()];hit=[r for r in rows if 'w09' in r['body']];print(hit[-1]['signature-valid'])" "$TMP/rcv2.log")
  [[ "$V" == "False" ]] && ok W-09 "tampered secret flagged invalid" || bad W-09 "sig-valid=$V (should be False)"
else
  bad W-09 "tamper delivery not received"
fi

# W-06 / W-08
B=$(curl -s -m 60 -H "X-Tenant-Id: $TC" "$BASE/api/v1/webhooks/messages?limit=20")
echo "$B" | grep -q "job.completed" && ok W-06 "messages observable" || bad W-06 "got: $(echo "$B" | head -c 150)"
N=$(echo "$B" | python3 -c "import json,sys;d=json.load(sys.stdin);print(len(d if isinstance(d,list) else d.get('data',[])))" 2>/dev/null || echo 0)
[[ "$N" -ge 2 ]] && ok W-08 "repeated publish accumulates" || bad W-08 "messages=$N"

# W-07
B=$(curl -s -m 60 -H "X-Tenant-Id: $TC" "$BASE/api/v1/webhooks/messages/$M1/attempts")
echo "$B" | grep -q "endpointId" && echo "$B" | grep -q "status" && ok W-07 "attempts structure" || bad W-07 "got: $(echo "$B" | head -c 150)"

# W-10
C=$(curl -s -m 60 -o /dev/null -w "%{http_code}" -X DELETE -H "X-Tenant-Id: $TC" "$BASE/api/v1/webhooks/destinations/ep_nonexistent")
[[ "$C" != "200" ]] && ok W-10 "missing endpoint delete errors ($C)" || bad W-10 "silently succeeded"

# T-06
B=$(curl -s -m 60 -H "X-Tenant-Id: $TB" "$BASE/api/v1/webhooks/destinations")
echo "$B" | grep -q "$EP" && bad T-06 "tenant-b sees tenant-c destination!" || ok T-06 "webhook tenant isolation"

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
  SVIX_SERVER_URL="http://localhost:8071" \
  nohup java -jar build/libs/platform-service.jar > /tmp/platform-service.log 2>&1 & )
PERSIST_OK=0
for _ in $(seq 1 30); do curl -sf -m 60 "$BASE/actuator/health" >/dev/null && { PERSIST_OK=1; break; }; sleep 1; done
if [[ "$PERSIST_OK" == "1" ]]; then
  BODY=$(curl -s -m 60 -H "X-Tenant-Id: $TA" "$BASE/api/v1/files/download?path=a%2Fdocs%2Ff02.csv")
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
