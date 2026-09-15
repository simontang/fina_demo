# el-ai-gateway

Estée Lauder AI project API gateway for voice-file tagging. Standalone Fastify + TypeScript
service that orchestrates:

- `platform-service` (5707) files — upload + presigned download URL
- `agent` (5702) A2A — triggers the voice agent (transcription + tagging)
- `agent` (5702) `/open/mcp` — task/activity tools (`task_manage_task`)

## Endpoints (all require `Authorization: Bearer <GATEWAY_API_KEYS key>`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/files?path=&fileName=` | multipart `file`; returns platform upload receipt (`uuid`, ...) |
| POST | `/api/v1/voice-tagging` | `{uuid?,title?,description?,assistantId?}`; presigns, creates a task, triggers A2A with only the task id (`uuid` defaults to `A2A_VOICE_TAGGING_FILE_UUID`) |
| GET | `/api/v1/voice-tagging/:id` | task status + recent activities |
| POST | `/api/v1/voice-tagging/:id/feedback` | `{content,summary?}`; relays feedback to the agent over A2A (agent appends activity) |

## Run

```bash
cp .env.example .env   # fill A2A_API_KEY / MCP_API_KEY / A2A_VOICE_TAGGING_ASSISTANT_ID
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
- The gateway only creates tasks and reads status. `add_activity` / `set_status` are performed by the agent
  (the Open MCP path has no runtime identity), so feedback is relayed to the agent over A2A.
