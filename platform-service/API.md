# platform-service File API

All examples below are real request/response pairs captured against a running
service (2026-09-14). Base path `/api/v1/files`; public prefix `/api/filesvc/`.

## Conventions

| Item | Value |
| --- | --- |
| Addressing | **uuid** (32 hex chars, unguessable). The logical path `dir/filename` is metadata — set at upload, used for query/display, never an address |
| Required header | `X-Tenant-Id` — the only place the tenant travels (never in body/path, never echoed back) |
| Optional headers | `X-Api-Key` (enforced when `FILE_SERVICE_API_KEY` is set), `X-User-Id` (fills `createdBy`) |
| Storage | Discrete + flat: `{tenantId}/{uuid}`. Storage keys are never exposed |
| Versioning | Re-upload with identical content dedupes (same uuid); different content appends a version as a **new uuid**. Each uuid pins exactly one stored version |

---

## 1. `POST /upload` — multipart upload (server generates uuid)

Request

```
POST /api/v1/files/upload
X-Tenant-Id: rev-001138
Content-Type: multipart/form-data; boundary=----X

file          = stock.csv  (text/csv, 37 bytes)
path          = ops/2026-09        (metadata)
fileName      = stock.csv          (optional; defaults to the uploaded name)
fileCategory  = raw                (optional)
usage         = import             (optional)
meta          = {"src":"wms"}      (optional, free-form JSON)
```

Response `200`

```json
{"uuid":"06c1097c09694b0d94f49f1a36e84123",
 "fullPath":"ops/2026-09/stock.csv","path":"ops/2026-09","filename":"stock.csv",
 "version":1,
 "sha256":"6310db39c08c8dec3fc69c6459643ab2d9fb0d3be0b93df9da1a1eeac4ef0b0c",
 "md5":"a65ff050a30cc45c0afaf34d2e6c153c",
 "size":37,"mime":"text/csv","fileCategory":"raw","usage":"import",
 "meta":"{\"src\": \"wms\"}","status":"active","createdBy":"api",
 "createdAt":"2026-09-14T00:11:39.124557","deduplicated":false}
```

Versioning / dedupe (measured)

| Action | Result |
| --- | --- |
| First upload of `v1` | `uuid=c9bf7383…, version=1, deduplicated=false` |
| Same bytes again | `uuid=c9bf7383…, version=1, deduplicated=true` (same row, no new object) |
| Changed bytes | `uuid=c68bb030…, version=2, deduplicated=false` — **new uuid**; the v1 uuid still downloads `v1` |

## 2. `PUT /{uuid}` — raw-stream upload (client chooses uuid)

Request

```
PUT /api/v1/files/1bc0b6aa7847429ea5c49422cb852f7a
X-Tenant-Id: rev-001138
X-File-Path: ops/stream
X-File-Name: raw.bin
X-File-Category: raw
X-File-Usage: export
Content-Type: application/octet-stream

hello-stream
```

Response `200`

```json
{"uuid":"1bc0b6aa7847429ea5c49422cb852f7a","fullPath":"ops/stream/raw.bin",
 "path":"ops/stream","filename":"raw.bin","version":1,
 "sha256":"efa2b3babba7a399f3e3dae714c57eaee1575ecebb3450a2810b3d82a6f88990",
 "md5":"cbd0f07b59c36af0c906f4a379644977","size":13,
 "mime":"application/octet-stream","fileCategory":"raw","usage":"export",
 "meta":null,"status":"active","createdBy":"api",
 "createdAt":"2026-09-14T00:11:43.354434","deduplicated":false}
```

`X-File-Path`/`X-File-Name` are required for a uuid that does not exist yet; for
an existing uuid they default to that row's metadata (re-PUT = new version).

## 3. `GET /{uuid}` — metadata (never content)

Request `GET /api/v1/files/06c1097c09694b0d94f49f1a36e84123` + `X-Tenant-Id`

Response `200` — same JSON shape as the upload receipt.

## 4. `GET /{uuid}/download` — content

Request `GET /api/v1/files/06c1097c…/download` + `X-Tenant-Id`

Response `200`

```
Content-Disposition: Attachment;Filename*=utf-8''stock.csv
X-File-Version: 1
Content-Type: text/csv

sku,qty,price
A-1,10,9.5
A-2,20,3.25
```

`?bom=true` prepends a UTF-8 BOM for CSV/text (first bytes become `efbbbf736b75`);
an existing BOM is not duplicated.

## 5. `HEAD /{uuid}` — metadata as headers, empty body

Response `200`

```
ETag: "6310db39c08c8dec3fc69c6459643ab2d9fb0d3be0b93df9da1a1eeac4ef0b0c"   (sha256)
X-File-Md5: a65ff050a30cc45c0afaf34d2e6c153c
X-File-Version: 1
X-File-Path: ops/2026-09/stock.csv
X-File-Category: raw
X-File-Usage: import
X-File-Meta: {"src": "wms"}
Content-Type: text/csv
Content-Length: 37
```

## 6. `GET /` — find files under a folder

```
GET /api/v1/files?path=ops/2026-09&q=stock&recursive=true
   &fileCategory=raw&usage=import&from=2026-09-01&to=2026-09-30&page=1&size=20
```

| Param | Meaning |
| --- | --- |
| `path` | folder to scope to (default: tenant root) |
| `q` | filename substring (case-insensitive) |
| `recursive` | `false` (default) = that folder only; `true` = include all descendants |
| `fileCategory`, `usage` | exact-match filters |
| `from`, `to` | `YYYY-MM-DD` (inclusive; `to` covers the whole day) or ISO timestamp |
| `page`, `size` | page number (1-based) / rows per page (default 20, max 1000) |

Measured responses

```jsonc
// ?path=ops            → browse one level: folders to drill into, no files yet
{"path":"ops","recursive":false,"query":null,"directories":["2026-09","stream"],
 "files":[],"page":1,"size":20,"total":0,"totalPages":0}

// ?path=ops&recursive=true&page=1&size=2
{"path":"ops","recursive":true,"query":null,"directories":[],
 "files":[{"fullPath":"ops/stream/raw.bin",…},{"fullPath":"ops/2026-09/stock.csv",…}],
 "page":1,"size":2,"total":2,"totalPages":1}

// ?q=stock&recursive=true
{"path":"","recursive":true,"query":"stock","directories":[],
 "files":[{"fullPath":"ops/2026-09/stock.csv",…}],"page":1,"size":20,"total":1,"totalPages":1}

// ?path=ops&q=raw&fileCategory=raw  (non-recursive → 0, because raw.bin sits in ops/stream)
{"path":"ops","recursive":false,"query":"raw","directories":["2026-09","stream"],
 "files":[],"page":1,"size":20,"total":0,"totalPages":0}

// ?path=ops&q=raw&fileCategory=raw&recursive=true
{"path":"ops","recursive":true,"query":"raw","directories":[],
 "files":[{"fullPath":"ops/stream/raw.bin",…}],"page":1,"size":20,"total":1,"totalPages":1}
```

Note the third/fourth rows: with `recursive=false` the scope is exactly that
folder, so a file one level down is not returned — that is the common gotcha.

## 7. `POST /presign` — get a download link

Request

```
POST /api/v1/files/presign
X-Tenant-Id: rev-001138
Content-Type: application/json

{"uuid":"06c1097c09694b0d94f49f1a36e84123","ttlSeconds":600}
```

Response `200`

```json
{"uuid":"06c1097c09694b0d94f49f1a36e84123","expiresInSeconds":600,
 "url":"http://localhost:5707/api/v1/files/06c1097c…/download","kind":"direct"}
```

`kind` is decided per request from storage reachability (`FILE_LINK_MODE=auto`):

| Storage | kind | url |
| --- | --- | --- |
| Cloud/reachable (TOS, S3) | `presigned` | storage-native signed URL (`X-Amz-Signature`), download bypasses this service |
| Internal (self-hosted MinIO) | `direct` | our own `/{uuid}/download` URL |

## 8. `DELETE /{uuid}` — soft delete (that version only)

Request `DELETE /api/v1/files/06c1097c…` + `X-Tenant-Id`

Response `200` `{"deleted":1}` — then `GET` returns `404`; the stored object and
the row are retained, and sibling versions of the same logical file are untouched.

---

## Errors

| Case | Status | Body |
| --- | --- | --- |
| Missing `X-Tenant-Id` | 400 | `{"code":"TENANT_REQUIRED","message":"X-Tenant-Id header is required"}` |
| Wrong/missing `X-Api-Key` (when enabled) | 401 | `{"code":"API_KEY_INVALID","message":"X-Api-Key header is missing or invalid"}` |
| `upload` without multipart body | 415 | `{"code":"UNSUPPORTED_MEDIA_TYPE","message":"Current request is not a multipart request"}` |
| Empty file | 400 | `{"code":"BAD_REQUEST","message":"file part is empty"}` |
| Malformed uuid | 400 | `{"code":"BAD_REQUEST","message":"uuid must be 32 hex characters"}` |
| Unknown/deleted uuid | 404 | `{"code":"NOT_FOUND","message":"no active file matches the given address"}` |
| Cross-tenant uuid | 404 | same as above (tenants are isolated; another tenant's uuid is indistinguishable from a missing one) |
| Path with `..` or `\` | 400 | `{"code":"BAD_REQUEST","message":"path must not contain '..' or backslash"}` |
| `presign` without a valid uuid | 400 | `{"code":"BAD_REQUEST","message":"uuid must be 32 hex characters"}` |
| Bad date filter | 400 | `{"code":"BAD_REQUEST","message":"invalid date: bogus (expected YYYY-MM-DD)"}` |
| `PUT` to a new uuid without `X-File-Name` | 400 | `{"code":"BAD_REQUEST","message":"X-File-Name header is required for a new uuid"}` |
