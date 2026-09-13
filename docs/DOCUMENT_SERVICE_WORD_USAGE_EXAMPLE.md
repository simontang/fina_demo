# Document Service Word Usage Example

This example uses the public Document Service endpoint and a real Word file:

- Base URL: `https://ada.alphafina.cn/api/documents`
- File: `business_line_rules_for_review.docx`
- File type: Microsoft Word 2007+ (`.docx`)
- File size: `8977` bytes
- Tested engine: `textin`

The example below uses real responses captured from the deployed service.

## 1. Health Check

Request:

```bash
curl -k https://ada.alphafina.cn/api/documents/health
```

Actual response:

```json
{"status":"ok"}
```

## 2. Check Engines

Request:

```bash
curl -k https://ada.alphafina.cn/api/documents/v1/engines
```

Actual response summary:

```text
datalab           available=true   requires_public_url=false
mineru            available=true   requires_public_url=true
textin            available=true   requires_public_url=false
qwen_ocr          available=true   requires_public_url=true
paddleocr_remote  available=true   requires_public_url=false
```

For this Word example, `textin` was used because it supports `docx` and does
not require a public presigned URL.

## 3. Upload Word File

Request:

```bash
curl -k -X POST \
  -F "file=@business_line_rules_for_review.docx;type=application/vnd.openxmlformats-officedocument.wordprocessingml.document" \
  https://ada.alphafina.cn/api/documents/v1/assets
```

Actual HTTP status:

```text
201
```

Actual response:

```json
{
  "asset_id": "asset_dd2340bc58d2401eb45e50ea7696b624",
  "filename": "business_line_rules_for_review.docx",
  "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "size_bytes": 8977,
  "storage_key": "assets/asset_dd2340bc58d2401eb45e50ea7696b624/business_line_rules_for_review.docx",
  "kind": "original"
}
```

The `asset_id` is used as `source_asset_id` in the parse request.

## 4. Create Parse Run

Request:

```bash
curl -k -X POST \
  -H "Content-Type: application/json" \
  https://ada.alphafina.cn/api/documents/v1/runs \
  -d '{
    "operation": "document.parse",
    "engine": "textin",
    "inputs": {
      "source_asset_id": "asset_dd2340bc58d2401eb45e50ea7696b624"
    },
    "params": {
      "output_formats": ["markdown", "json"],
      "mode": "balanced",
      "language_hint": ["zh", "en"]
    }
  }'
```

Actual HTTP status:

```text
202
```

Actual response:

```json
{
  "run_id": "run_824f3a03cd3a4096a20f1596f5ee9f81",
  "operation": "document.parse",
  "requested_engine": "textin",
  "selected_engine": null,
  "status": "queued",
  "inputs": {
    "source_asset_id": "asset_dd2340bc58d2401eb45e50ea7696b624"
  },
  "params": {
    "output_formats": ["markdown", "json"],
    "mode": "balanced",
    "language_hint": ["zh", "en"],
    "page_ranges": null
  },
  "outputs": {},
  "error_code": null,
  "error_message": null
}
```

## 5. Poll Run Status

Request:

```bash
curl -k https://ada.alphafina.cn/api/documents/v1/runs/run_824f3a03cd3a4096a20f1596f5ee9f81
```

Observed poll states:

```text
poll=1 status=running engine=textin
poll=2 status=running engine=textin
poll=3 status=succeeded engine=textin
```

Actual final response:

```json
{
  "run_id": "run_824f3a03cd3a4096a20f1596f5ee9f81",
  "operation": "document.parse",
  "requested_engine": "textin",
  "selected_engine": "textin",
  "status": "succeeded",
  "inputs": {
    "source_asset_id": "asset_dd2340bc58d2401eb45e50ea7696b624"
  },
  "params": {
    "output_formats": ["markdown", "json"],
    "mode": "balanced",
    "language_hint": ["zh", "en"],
    "page_ranges": null
  },
  "outputs": {
    "preprocessed_pdf": "asset_f64669c2427b4b44a38a00260df96cfe",
    "document_ir": "asset_4b1564c59dce4ca9b80571961f9934bb",
    "markdown": "asset_fcf5cb8b8554452e978b7898efcea5d0",
    "engine_raw": "asset_4aedc81f8a5c475f89a2d2a0af3f944e",
    "engine": "textin",
    "summary": {
      "block_count": 7,
      "table_count": 0,
      "has_markdown": true
    }
  },
  "error_code": null,
  "error_message": null
}
```

Notes:

- `preprocessed_pdf` is present because the service rendered the Word document
  to PDF before sending it to the parsing engine.
- `markdown`, `document_ir`, and `engine_raw` are stored as document assets.
- `status=succeeded` means the upload, object storage, worker, engine call,
  result normalization, and output persistence all completed.

## 6. Download Markdown Output

Request:

```bash
curl -k \
  https://ada.alphafina.cn/api/documents/v1/runs/run_824f3a03cd3a4096a20f1596f5ee9f81/outputs/markdown
```

Actual HTTP status:

```text
200
```

Actual markdown size:

```text
13281 bytes
```

Actual markdown preview:

```markdown
# TUV A／P／M／I 业务线判定规则

业务方Review版｜基于当前deep-agent skill与2026-04-28 golden 回归结果整理

提示：本文档用于业务方确认分类口径，不是程序代码说明。当前规则坚持“正文证据优先、对象＋动作＋标准＋交付物优先”，历史人工桶只用于学习可复用模式，不作为单条记录的硬编码依据。

## 1．使用范围

当前判定空间覆盖A／P／M／I四个cluster下的14条业务线，以及无法稳定归入业务线时的REVIEW_OR_NA。

·A cluster： A01、A02、A04，主要覆盖组织级认证／审核、ESG／可持续治理、培训辅导与体系建设。

·P cluster： P04＿PV、P04＿SUSTAINABILITY、P05，主要覆盖光伏／储能产品与项目技术服务、产品碳足迹／EPD／LCA、医疗器械认证合规。

·M cluster：M04、M05，主要覆盖汽车与轨交相关认证、测试、监管准入和系统安全。

·I cluster： I01、I04、I05、I06＿PETROCHEM、I06＿WIND、I07，主要覆盖工业检测、建筑环境、石化过程安全、风电、软件／功能安全／网络安全。

·REVIEW＿OR＿NA：公告证据不足、与14条业务线无稳定关系、或只看到采购／入围／普通建设外壳时使用。
```

## Minimal Reusable Script

Replace `WORD_FILE` with your local Word document path.

```bash
BASE_URL="https://ada.alphafina.cn/api/documents"
WORD_FILE="business_line_rules_for_review.docx"

UPLOAD_RESPONSE=$(curl -k -sS -X POST \
  -F "file=@${WORD_FILE};type=application/vnd.openxmlformats-officedocument.wordprocessingml.document" \
  "${BASE_URL}/v1/assets")

echo "${UPLOAD_RESPONSE}"

ASSET_ID=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["asset_id"])' <<< "${UPLOAD_RESPONSE}")

RUN_RESPONSE=$(curl -k -sS -X POST \
  -H "Content-Type: application/json" \
  "${BASE_URL}/v1/runs" \
  -d "{
    \"operation\": \"document.parse\",
    \"engine\": \"textin\",
    \"inputs\": {\"source_asset_id\": \"${ASSET_ID}\"},
    \"params\": {
      \"output_formats\": [\"markdown\", \"json\"],
      \"mode\": \"balanced\",
      \"language_hint\": [\"zh\", \"en\"]
    }
  }")

echo "${RUN_RESPONSE}"

RUN_ID=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])' <<< "${RUN_RESPONSE}")

for i in $(seq 1 60); do
  STATUS_RESPONSE=$(curl -k -sS "${BASE_URL}/v1/runs/${RUN_ID}")
  STATUS=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' <<< "${STATUS_RESPONSE}")
  echo "poll=${i} status=${STATUS}"
  if [ "${STATUS}" = "succeeded" ] || [ "${STATUS}" = "failed" ] || [ "${STATUS}" = "cancelled" ]; then
    echo "${STATUS_RESPONSE}"
    break
  fi
  sleep 5
done

curl -k -sS "${BASE_URL}/v1/runs/${RUN_ID}/outputs/markdown"
```

