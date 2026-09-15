# platform-service Webhook API

Real request/response pairs captured against a running service (2026-09-14).
Base path `/api/v1/webhooks`; public prefix `/api/webhooks/`.

## What this is

A thin, tenant-aware facade over a self-hosted **Svix** server. Callers use
Svix-compatible `eventType`, `filterTypes`, `payload`, and `channels` while
Svix applications stay hidden: the tenant rides only in `X-Tenant-Id`; the
facade maps it to a Svix application (created on first use) and each destination
to a Svix endpoint with its own `whsec_` signing secret.

```
factory agent / script ──► platform-service /api/v1/webhooks ──► svix-server ──► customer endpoint
   X-Tenant-Id               (tenant→application, eventType→EventType)   (retries, signing)
```

EventTypes in use: `import.completed`, `gate.passed`, `decision.captured`,
`job.completed`, `run.published`. Deliveries carry Standard Webhooks headers
(`webhook-id`, `webhook-timestamp`, `webhook-signature`, HMAC-SHA256) with
at-least-once delivery and Svix's automatic retry schedule.

---

## 1. `POST /destinations` — register a delivery target

Request

```
POST /api/v1/webhooks/destinations
X-Tenant-Id: whrev-001624
Content-Type: application/json

{"url":"http://host.docker.internal:5909/catch",
 "filterTypes":["job.completed","gate.passed"],
 "channels":["vip-customers"],
 "description":"review demo"}
```

Response `200`

```json
{"endpointId":"ep_3JHRotHEn0IaYlNRo56HDJmj36i",
 "url":"http://host.docker.internal:5909/catch",
 "filterTypes":["job.completed","gate.passed"],
 "channels":["vip-customers"],
 "secret":"whsec_2OxENpeHlS8GZRep/Kh9yyt6WFs9td/t"}
```

The `secret` is what the receiver uses to verify signatures. **Store it once —
Svix returns it at creation time only.**

`filterTypes` maps directly to Svix endpoint `filterTypes`; omit or pass an
empty list to let the endpoint receive all event types. `channels` maps directly
to Svix endpoint channels. If an endpoint has channels, it receives only
messages published with at least one matching channel. Leave `channels` empty
for normal eventType-only broadcast behavior. Channel names follow Svix's API
contract: max 10 values, max 128 characters each, using only letters, digits,
`-`, `_`, `.`, `:`, `+`.

## 2. `GET /destinations` — list targets

Request `GET /api/v1/webhooks/destinations` + `X-Tenant-Id`

Response `200`

```json
[{"endpointId":"ep_3JHRotHEn0IaYlNRo56HDJmj36i",
  "url":"http://host.docker.internal:5909/catch",
  "filterTypes":["job.completed","gate.passed"],
  "channels":["vip-customers"],
  "disabled":false}]
```

## 3. `DELETE /destinations/{endpointId}` — remove a target

Request `DELETE /api/v1/webhooks/destinations/ep_3JHRotHEn0IaYlNRo56HDJmj36i`

Response `200` `{"deleted":"ep_3JHRotHEn0IaYlNRo56HDJmj36i"}`

## 4. `POST /publish` — emit an event, optionally channel-filtered

Broadcast request

```
POST /api/v1/webhooks/publish
X-Tenant-Id: whfinal-001823
Content-Type: application/json

{"eventType":"gate.passed","payload":{"gate":"scope_confirmed","decisions":["D-2026-001"]}}
```

Response `200`

```json
{"messageId":"msg_3JHTFnhm9CTgBOgRzyeRaJ9peDk","eventType":"gate.passed","channels":[]}
```

Channel-filtered request (Svix native targeting model)

```
POST /api/v1/webhooks/publish
X-Tenant-Id: whfinal-001823
Content-Type: application/json

{"eventType":"gate.passed",
 "channels":["vip-customers"],
 "payload":{"gate":"scope_confirmed","decisions":["D-2026-001"]}}
```

This is the supported way to send to a subset of destinations. Do not send
`endpointIds`: Svix does not model publish as a direct endpoint-id operation,
and this facade rejects `endpointIds` with `400` so it cannot be silently
interpreted as a broadcast.

What the receiving endpoint actually gets (measured)

```
POST /catch
webhook-id: msg_3JHTFnhm9CTgBOgRzyeRaJ9peDk
webhook-timestamp: 1789316892
webhook-signature: v1,IBSl8td+sPuCKa2qk3/nitf92oObm9RlsroTqEcuHzY=

{"gate":"scope_confirmed","decisions":["D-2026-001"]}
```

Signature verification (receiver side): HMAC-SHA256 over
`{webhook-id}.{webhook-timestamp}.{body}` with the base64 payload of the
destination's `whsec_…` secret; compare constant-time against each `v1,` entry.
`scripts/mock-receiver.py` implements this and reports `signature-valid`.

An eventType that has never been used is registered on first publish (lazy), so
a typo creates a new event type rather than failing — see the note below.

## 5. `GET /messages?limit=` — recent events for this tenant

Request `GET /api/v1/webhooks/messages?limit=5`

Response `200`

```json
[{"messageId":"msg_3JHRqFXQ8FkYA6Wgliluye8C5oQ",
  "eventType":"gate.passed","timestamp":"2026-09-13T16:16:35.838619Z"},
 {"messageId":"msg_3JHRpJ63AuQiLHP2aoEdXQmqsSh",
  "eventType":"job.completed","timestamp":"2026-09-13T16:16:28.327537Z"}]
```

## 6. `GET /messages/{messageId}/attempts` — delivery status per target

Request `GET /api/v1/webhooks/messages/msg_3JHRpJ63AuQiLHP2aoEdXQmqsSh/attempts`

Response `200` — one entry per destination the message fanned out to:

```json
// delivered successfully
[{"endpointId":"ep_3JHRotHEn0IaYlNRo56HDJmj36i",
  "url":"http://host.docker.internal:5909/catch",
  "status":"success"}]

// failed, with the scheduled retry
[{"endpointId":"ep_3JHRotHEn0IaYlNRo56HDJmj36i",
  "url":"http://host.docker.internal:5909/catch",
  "status":"fail",
  "nextAttempt":"2026-09-13T16:21:26.078772Z"}]
```

`status` is Svix's human-readable `statusText` (`success` / `pending` / `fail`);
`nextAttempt` appears only while a retry is pending.

---

## Errors

| Case | Status | Body |
| --- | --- | --- |
| Missing `X-Tenant-Id` | 400 | `{"code":"TENANT_REQUIRED","message":"X-Tenant-Id header is required"}` |
| Destination without `url` | 400 | `{"code":"BAD_REQUEST","message":"webhook backend rejected the request"}` |
| Publish with `endpointIds` | 400 | `{"code":"BAD_REQUEST","message":"endpointIds is not supported ... use channels"}` |
| Unknown `endpointId` on delete | 404 | `{"code":"NOT_FOUND","message":"webhook resource not found"}` |
| Unknown `messageId` on attempts | 404 | `{"code":"NOT_FOUND","message":"webhook resource not found"}` |
| svix-server unreachable | 502 | `{"code":"WEBHOOK_BACKEND_ERROR","message":"webhook backend returned HTTP …"}` |

The facade translates the backend's statuses into this contract — a missing
destination or message is a 404 for the caller, never a 500.

---

## Notes / open points

1. **Lazy eventType registration**: `publish` with an unregistered eventType returns
   `200` and creates that event type. Handy for adding eventTypes without a
   migration, but a typo (`job.complete` vs `job.completed`) silently succeeds
   and delivers nothing. If you prefer strictness, publishing can be validated
   against a fixed eventType list.
2. **Tenant → application mapping is cached in-process**, so the first call
   after a restart re-resolves it (a list lookup by uid, since this
   svix-server build has no get-by-uid route).
3. **No inbound webhook receiver here** — this service only sends. Callbacks
   from external systems (e.g. a "decision confirmed" ping) would be a separate
   endpoint.
4. **Portal**: `GET /portal` offers the same data in a UI (destinations,
   events, delivery status).
