# el-ai-gateway

Estée Lauder AI project API gateway for voice-file tagging. Standalone Fastify + TypeScript
service that orchestrates:

- `platform-service` (5707) files — upload + presigned download URL
- `agent` (5702) `POST /api/runs` — dispatches a background run to the voice agent (transcription + tagging)
- `agent` (5702) `/open/mcp` — task/activity tools (`task_manage_task`)

## Endpoints (all require `Authorization: Bearer <GATEWAY_API_KEYS key>`)

> 完整 API 文档（含两个 webhook 事件格式）见 [API.md](./API.md)；回调地址注册与接收端实现见 [WEBHOOK.md](./WEBHOOK.md)。

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/files?path=&fileName=` | multipart `file`; returns platform upload receipt (`uuid`, ...) |
| POST | `/api/v1/voice-tagging` | `{uuid?,title?,description?,assistantId?}`; presigns, creates a task, dispatches an agent run (background) with only the task id (`uuid` defaults to `VOICE_TAGGING_FILE_UUID`) |
| GET | `/api/v1/voice-tagging/:id` | task status + recent activities |
| POST | `/api/v1/voice-tagging/:id/feedback` | `{content,summary?}`; relays feedback to the agent via a background run (agent appends activity) |

## Run

```bash
cp .env.example .env   # fill AGENT_LOGIN_EMAIL / AGENT_LOGIN_PASSWORD / MCP_API_KEY
pnpm install
pnpm dev
```

## Probe the MCP server

```bash
pnpm mcp:probe                       # prints tools/list
MCP_PROBE_ACTION=create_task pnpm mcp:probe   # also tries task_manage_task create
```

## Test

```bash
pnpm test
pnpm typecheck
```

## Smoke (against running gateway + real upstreams)

```bash
./scripts/smoke.sh /path/to/voice.wav
```

## Notes

- The two webhook callbacks are fired by the agent platform, not this gateway.
- Task/activity state lives in the agent platform (`task_manage_task`); the gateway is stateless.
- The gateway only creates tasks and reads status. `add_activity` / `set_status` are performed by the agent,
  so feedback is relayed to the agent as a run.
- The agent trigger is **fire-and-forget**: the gateway logs in (session token, cached), POSTs a background
  run to `/api/runs` carrying only the `taskId`, and returns immediately (errors are logged). Task status is
  read from the task created via MCP.
