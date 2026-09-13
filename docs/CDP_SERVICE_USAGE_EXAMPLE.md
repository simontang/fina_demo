# CDP Service Usage Example

This document shows a minimal end-to-end flow for using `cdp_service`:

1. Create a segment definition with SQL.
2. Run processing for that definition.
3. Read the saved segment data JSON snapshot.
4. Clean up test data.

## Service URL

Local development:

```bash
export CDP_BASE_URL="http://localhost:5706/api/v1"
```

Remote server through an SSH tunnel:

```bash
ssh -L 15706:127.0.0.1:5706 deploy@ada.alphafina.cn
export CDP_BASE_URL="http://127.0.0.1:15706/api/v1"
```

If nginx exposes CDP under `/api/cdp/`, use:

```bash
export CDP_BASE_URL="https://ada.alphafina.cn/api/cdp"
```

## Tenant Header

Every request should send tenant information:

```bash
export TENANT_ID="demo_tenant"
export TENANT_HEADER="X-Tenant-Id: ${TENANT_ID}"
```

If the header is missing, the service uses tenant `default`. Do not send
`tenantId` in the JSON body; the service always resolves tenant from the
request header.

## Response Format

All APIs return the same wrapper:

```json
{
  "code": 200,
  "message": "success",
  "data": {}
}
```

Errors return `code: 400` with an error message.

## 1. Create Segment Definition

`datasourceId` must exist in the master table `t_datasource_config`. The
example below uses `11`, which was available in the demo database during smoke
testing. Change it if your environment uses another datasource.

The SQL may use named parameters like `:minValue`; pass values later in the
processing request.

```bash
curl -sS -X POST "${CDP_BASE_URL}/segment-definitions" \
  -H "Content-Type: application/json" \
  -H "${TENANT_HEADER}" \
  -d '{
    "name": "High value demo customers",
    "description": "Customers selected from demo_customers by lifetime value",
    "datasourceId": 11,
    "querySql": "select customer_id, anonymous_alias, member_level, lifetime_value, serum_cohort_type from public.demo_customers where lifetime_value >= :minValue order by lifetime_value desc limit 10",
    "status": 1
  }' | jq .
```

Save the returned definition id:

```bash
export DEFINITION_ID="<id from response data.id>"
```

## 2. List And Read Definitions

List definitions for the current tenant:

```bash
curl -sS "${CDP_BASE_URL}/segment-definitions" \
  -H "${TENANT_HEADER}" | jq .
```

Read one definition:

```bash
curl -sS "${CDP_BASE_URL}/segment-definitions/${DEFINITION_ID}" \
  -H "${TENANT_HEADER}" | jq .
```

## 3. Process The Definition

Processing loads the definition by `{tenantId, id}`, validates that `status=1`,
executes the read-only SQL against the definition's datasource, then saves the
full result set as one JSON array string in `t_segment_data.data_json`.

```bash
curl -sS -X POST "${CDP_BASE_URL}/segment-definitions/${DEFINITION_ID}/process" \
  -H "Content-Type: application/json" \
  -H "${TENANT_HEADER}" \
  -d '{
    "params": {
      "minValue": 10000
    }
  }' | jq .
```

The response contains a newly created segment data snapshot:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": 123,
    "tenantId": "demo_tenant",
    "definitionId": 45,
    "runId": "seg_run_...",
    "dataJson": "[{\"customer_id\":\"C0001\",\"anonymous_alias\":\"匿名顾客001\"}]",
    "rowCount": 1,
    "createdAt": "2026-07-12T10:00:00",
    "updatedAt": "2026-07-12T10:00:00"
  }
}
```

Save the returned data id:

```bash
export SEGMENT_DATA_ID="<id from response data.id>"
```

## 4. Read Segment Data

List snapshots for one definition:

```bash
curl -sS "${CDP_BASE_URL}/segment-data?definitionId=${DEFINITION_ID}&page=1&pageSize=20" \
  -H "${TENANT_HEADER}" | jq .
```

Read one snapshot:

```bash
curl -sS "${CDP_BASE_URL}/segment-data/${SEGMENT_DATA_ID}" \
  -H "${TENANT_HEADER}" | jq .
```

Parse `dataJson` locally:

```bash
curl -sS "${CDP_BASE_URL}/segment-data/${SEGMENT_DATA_ID}" \
  -H "${TENANT_HEADER}" \
  | jq -r '.data.dataJson' \
  | jq .
```

`dataJson` is stored as text because each SQL may return a different set of
columns.

## 5. Manual Segment Data CRUD

Normally segment data is created by processing. Manual CRUD is available for
tests or imports.

Create:

```bash
curl -sS -X POST "${CDP_BASE_URL}/segment-data" \
  -H "Content-Type: application/json" \
  -H "${TENANT_HEADER}" \
  -d "{
    \"definitionId\": ${DEFINITION_ID},
    \"runId\": \"manual_run_001\",
    \"dataJson\": \"[{\\\"customer_id\\\":\\\"C_TEST\\\",\\\"score\\\":99}]\"
  }" | jq .
```

Update:

```bash
curl -sS -X PUT "${CDP_BASE_URL}/segment-data/${SEGMENT_DATA_ID}" \
  -H "Content-Type: application/json" \
  -H "${TENANT_HEADER}" \
  -d "{
    \"definitionId\": ${DEFINITION_ID},
    \"runId\": \"manual_run_001_updated\",
    \"dataJson\": \"[{\\\"customer_id\\\":\\\"C_TEST\\\",\\\"score\\\":100}]\"
  }" | jq .
```

The service validates that `dataJson` is valid JSON.

## 6. Update Or Disable Definition

Update the SQL or metadata:

```bash
curl -sS -X PUT "${CDP_BASE_URL}/segment-definitions/${DEFINITION_ID}" \
  -H "Content-Type: application/json" \
  -H "${TENANT_HEADER}" \
  -d '{
    "name": "High value demo customers v2",
    "description": "Updated threshold example",
    "datasourceId": 11,
    "querySql": "select customer_id, anonymous_alias, lifetime_value from public.demo_customers where lifetime_value >= :minValue limit 5",
    "status": 1
  }' | jq .
```

Disable processing without deleting the definition:

```bash
curl -sS -X PUT "${CDP_BASE_URL}/segment-definitions/${DEFINITION_ID}" \
  -H "Content-Type: application/json" \
  -H "${TENANT_HEADER}" \
  -d '{
    "name": "High value demo customers v2",
    "description": "Disabled example",
    "datasourceId": 11,
    "querySql": "select customer_id from public.demo_customers limit 5",
    "status": 0
  }' | jq .
```

## 7. Tenant Isolation Check

The same id cannot be read from another tenant:

```bash
curl -sS "${CDP_BASE_URL}/segment-definitions/${DEFINITION_ID}" \
  -H "X-Tenant-Id: another_tenant" | jq .
```

Expected result:

```json
{
  "code": 400,
  "message": "Segment definition not found: <id>"
}
```

## 8. Cleanup

Delete segment data first, then the definition:

```bash
curl -sS -X DELETE "${CDP_BASE_URL}/segment-data/${SEGMENT_DATA_ID}" \
  -H "${TENANT_HEADER}" | jq .

curl -sS -X DELETE "${CDP_BASE_URL}/segment-definitions/${DEFINITION_ID}" \
  -H "${TENANT_HEADER}" | jq .
```

## SQL Rules

The processing SQL must be read-only:

- Allowed: one `SELECT` statement.
- Allowed: one `WITH ... SELECT ...` statement.
- Rejected: SQL containing `;`.
- Rejected: write or DDL keywords such as `insert`, `update`, `delete`,
  `drop`, `alter`, `truncate`, `create`, `merge`, `call`, `grant`, `revoke`,
  `vacuum`, or `analyze`.

Always add a reasonable `limit` while testing.

## API Summary

Definition APIs:

- `GET /api/v1/segment-definitions`
- `GET /api/v1/segment-definitions/{id}`
- `POST /api/v1/segment-definitions`
- `PUT /api/v1/segment-definitions/{id}`
- `DELETE /api/v1/segment-definitions/{id}`
- `POST /api/v1/segment-definitions/{id}/process`

Data APIs:

- `GET /api/v1/segment-data?definitionId=&page=&pageSize=`
- `GET /api/v1/segment-data/{id}`
- `POST /api/v1/segment-data`
- `PUT /api/v1/segment-data/{id}`
- `DELETE /api/v1/segment-data/{id}`

