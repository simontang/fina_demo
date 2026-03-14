#!/usr/bin/env bash
# Test Data Source API and Metric API (metrics-server)
# Run on deploy@demo.alphafina.cn when metrics_server is up on 5704.
# Usage: ./scripts/test-metrics-api.sh [BASE_URL]
#   BASE_URL default: http://127.0.0.1:5704
#   Or use: https://demo.alphafina.cn/api/metrics (nginx must strip prefix: proxy_pass http://127.0.0.1:5704/)

set -e
BASE_URL="${1:-http://127.0.0.1:5704}"

echo "=== Metrics API tests (BASE_URL=$BASE_URL) ==="

# 1) Data Source API
echo ""
echo "--- 1. GET /api/v1/datasources ---"
curl -s -w "\nHTTP: %{http_code}\n" "$BASE_URL/api/v1/datasources" | head -80

echo ""
echo "--- 2. GET /api/v1/datasources/active ---"
curl -s -w "\nHTTP: %{http_code}\n" "$BASE_URL/api/v1/datasources/active" | head -40

# 2) Metric API (need a valid dsId from step 1; using 1 for demo)
DS_ID="${DS_ID:-1}"
echo ""
echo "--- 3. GET /api/v1/datasources/${DS_ID}/metrics/index ---"
curl -s -w "\nHTTP: %{http_code}\n" "$BASE_URL/api/v1/datasources/${DS_ID}/metrics/index" | head -80

echo ""
echo "--- 4. GET /api/v1/datasources/${DS_ID}/metrics/net_sales_amt/detail ---"
curl -s -w "\nHTTP: %{http_code}\n" "$BASE_URL/api/v1/datasources/${DS_ID}/metrics/net_sales_amt/detail" | head -60

echo ""
echo "--- 5. POST /api/v1/metrics/query (semantic query) ---"
curl -s -X POST "$BASE_URL/api/v1/metrics/query" \
  -H "Content-Type: application/json" \
  -d '{"datasourceId":'"${DS_ID}"',"metrics":["net_sales_amt"],"groupBy":["DocDate__month"],"filters":[{"dimension":"DocDate","operator":"BETWEEN","values":["2025-01-01","2025-12-31"]}],"limit":10}' \
  -w "\nHTTP: %{http_code}\n" | head -50

echo ""
echo "=== Done ==="
