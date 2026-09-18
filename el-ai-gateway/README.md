# el-ai-gateway

Estée Lauder AI project API gateway for **voice-file tagging and customer/file/task management**. Standalone Fastify + TypeScript service that orchestrates:

- `platform-service` (5707) files — upload + presigned download URL
- `agent` (5702) `POST /api/runs` — dispatches a background run to the voice agent (transcription + tagging)
- `agent` (5702) `/open/mcp` — task/activity tools (`task_manage_task`)

## Endpoints (all require `Authorization: Bearer <GATEWAY_API_KEYS key>`)

> 完整 API 文档见 [API.md](./API.md)：**场景/界面导向**（客户详情 / 全部记录 / 补充客户标签 / 语音手记）+ **接口参考**（文件 / 语音打标任务 / 客户标签）+ [核心概念：两层标签](./API.md#2-核心概念两层标签客户标签-vs-任务标签)。转写与打标结果**全部通过查询接口获取**（无 webhook）。

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/files?path=&fileName=` | multipart `file`; returns platform upload receipt (`uuid`, ...) |
| GET | `/api/v1/files?baId=&customerId=` | list files for a BA + customer (both required) |
| GET | `/api/v1/files/:uuid/url` | presigned playback/download URL for an `<audio>` |
| POST | `/api/v1/voice-tagging` | `{uuid?,title?,description?,assistantId?}`; presigns, creates a task, dispatches an agent run (background) with only the task id (`uuid` defaults to `VOICE_TAGGING_FILE_UUID`) |
| GET | `/api/v1/voice-tagging?baId=&customerId=` | list a BA's tasks for a customer (fileId/taskId/status/tags) |
| GET | `/api/v1/voice-tagging/:id` | task status + tags |
| GET | `/api/v1/voice-tagging/:id/activities` | task activity timeline |
| PUT | `/api/v1/voice-tagging/:id/tags` | **overwrite** this task's tags `{tags:[tagId,...]}`; records an activity (**task-level, not customer-level**) |
| POST | `/api/v1/voice-tagging/:id/feedback` | `{content,summary?}`; relays feedback to the agent via a background run (agent appends activity) |
| GET | `/api/v1/customers/:customerId/tags` | a customer's business tags (name + tag uuid); **customer-level, query-only** (updated internally) |

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

- Transcription/tagging results are read through the query endpoints; the gateway does not deliver callbacks.
- Task/activity state lives in the agent platform (`task_manage_task`); the gateway is stateless.
- The gateway only creates tasks and reads status. `add_activity` / `set_status` are performed by the agent,
  so feedback is relayed to the agent as a run.
- The agent trigger is **fire-and-forget**: the gateway logs in (session token, cached), POSTs a background
  run to `/api/runs` carrying only the `taskId`, and returns immediately (errors are logged). Task status is
  read from the task created via MCP.
