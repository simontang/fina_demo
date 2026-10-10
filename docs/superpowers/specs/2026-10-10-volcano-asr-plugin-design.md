# Volcano ASR 插件设计（native agent Plugin）

- 日期：2026-10-10
- 状态：Approved（设计已定稿，待实现）
- 范围：`agent/src/agents/volcano_asr/`（新增）+ `agent/src/agents/index.ts`（一行注册）
- 参照：既有 native 插件 `agent/src/agents/platform_service/storage/`、`webhooks/`、`sap_b1/`

## 1. 背景与目标

把火山引擎（豆包语音）录音文件识别（ASR）HTTP API 封装为 Axiom Lattice **native Plugin**（不是沙盒插件），
暴露给平台 agent 与 Open/MCP 面。功能对齐既有沙盒版 volc-asr：

- 标准版（异步，需轮询）：submit + query，resource id `volc.seedasr.auc`（2.0）/ `volc.bigasr.auc`（1.0）
- 极速版（同步）：`recognize/flash`，resource id `volc.bigasr.auc_turbo`

**目标**：一个自包含插件，含连接（apiKey/baseUrl/resourceId）、连通测试、四个工具、单测。

**非目标（YAGNI）**：

- 不做沙盒文件字节上传（对齐沙盒版：只接受公网可访问 `audioUrl`）。
- 不做流式/实时 ASR（走 RTC，另有一套）。
- 不做 UI resource / MCP App。
- 不引入 skill / builder agent。

## 2. 决策总览

| 项 | 决策 |
|---|---|
| 插件类型 id | `volcano-asr`（即 Connection Store type 与 Open grant domain） |
| 位置 | `agent/src/agents/volcano_asr/`（应用层，与 `sap_b1` 同款） |
| 注册 | `plugin.ts` 末尾 `PluginRegistry.register(...)`；`volcano_asr/index.ts` 副作用导入；`agents/index.ts` 增一行 |
| 工具 | `submit_transcription`、`get_transcription`、`recognize`、`recognize_flash` |
| 连接字段 | `apiKey`(password,required)、`baseUrl`(string,optional)、`resourceId`(string,optional) |
| 认证/状态 | 走 HTTP header：请求 `X-Api-Key`/`X-Api-Resource-Id`/`X-Api-Request-Id`/`X-Api-Sequence`；响应 `X-Api-Status-Code`/`X-Api-Message` |
| 状态码 | `20000000` 成功；`20000001`/`20000002` 处理中 |
| Open 面 | 四个工具全部暴露；`get_transcription` 标 `readOnly` |
| 恢复力 | 参数全部走 `resolveConnection`（pre-resolved → 连接选择器 → env/默认）|

## 3. 目录结构

```
agent/src/agents/
  index.ts                      # += import "./volcano_asr"
  volcano_asr/
    index.ts                    # import "./plugin"
    client.ts                   # 连接解析 + volc HTTP + 文本抽取 + 错误映射
    executors.ts                # 纯执行器（可单测）
    plugin.ts                   # Plugin 定义 + connection.test + middleware + register
    __tests__/mockResponse.ts   # 测试用 fetch Response 工厂
    __tests__/client.test.ts
    __tests__/executors.test.ts
    __tests__/plugin.test.ts
```

## 4. 连接

`meta.type = "volcano-asr"` 同时是连接选择器类型。连接字段：

| key | type | widget | required | default | 说明 |
|---|---|---|---|---|---|
| apiKey | password | password | 是 | `process.env.VOLC_ASR_API_KEY` | 作为 `X-Api-Key` |
| baseUrl | string | input | 否 | `https://openspeech.bytedance.com` | 服务根地址，去尾 `/` |
| resourceId | string | input | 否 | `volc.seedasr.auc` | 标准版资源 id；极速版工具内部固定用 turbo |

`configSchema` 使用标准连接选择器：`connections: string[]`（widget `connectionSelect`）、`connectAll?: boolean`。
`defaultConfig = { connections: [], connectAll: false }`。

### 4.1 connection.test

对齐沙盒版 test.js：对文档中的示例 MP3 发起一次真实 submit，返回 `PluginConnectionTestResult`：

- 成功 → `{ ok: true, message: "连接正常：… statusCode=20000000，耗时 Xms", details }`
- 鉴权失败（statusCode 45000010 / Invalid X-Api-Key / HTTP 401）→ `{ ok: false, message, details }`
- 其他失败 → `{ ok: false, message, details }`

`details` 含 baseUrl、resourceId、masked apiKey、endpoint、testAudioUrl、httpStatus、statusCode、apiMessage、requestId、responseBody(截断)、elapsedMs。

## 5. HTTP 客户端（client.ts）

```ts
resolveConnection(rawConfig, exeConfig): Promise<{ apiKey; baseUrl; resourceId }>
// pre-resolved _resolvedConnections[0].config → 连接选择器(connectionType="volcano-asr", connections, connectAll)
//   → env/默认；无 apiKey 时调用方硬失败

callApi({ baseUrl, apiKey, resourceId, path, body, requestId, sequence })
// POST，返回 { httpStatus, statusCode, message, requestId, body, rawBody }

isSuccess(code) / isProcessing(code)
extractText(body) / extractUtterances(body)
errorResult(err) // 统一 JSON 错误（不抛裸异常）
maskSecret(key)
```

常量：`DEFAULT_BASE_URL`、`DEFAULT_RESOURCE_ID`、`SUBMIT_PATH`、`QUERY_PATH`、`FLASH_PATH`、`FLASH_RESOURCE_ID`、
`STATUS_SUCCESS`、`STATUS_PROCESSING`。

## 6. 工具

所有 executor 签名 `(input, exeConfig, rawConfig) => Promise<string>`，返回 JSON 字符串（与 storage 一致）。

| 工具 | 映射 | 入参要点 | 输出 |
|---|---|---|---|
| `submit_transcription` | `POST .../submit`（`X-Api-Sequence:-1`） | `audioUrl` 必填；可选 format/codec/rate/bits/channel、modelName、enableItn/Punc/Ddc/SpeakerInfo、showUtterances、sensitiveWordsFilter | `{requestId,statusCode,message,audioUrl,hint}` |
| `get_transcription` | `POST .../query` | `requestId` 必填；可选 `timeoutSeconds`(120)/`intervalSeconds`(3)/`responseFormat` | `{requestId,status,statusCode,text,attempts[,utterances,raw]}`；超时 → `status:"timeout"` |
| `recognize` | submit + query 轮询 | 同 submit + 轮询参数 | 同 `get_transcription` 成功态 |
| `recognize_flash` | `POST .../recognize/flash` | `audioUrl`+`format` 必填；可选 modelName、enableItn/Punc/Ddc/ChannelSplit、showUtterances、vadSegment、enableAutoLang/Lid、endWindowSize、sensitiveWordsFilter | `{requestId,status:"completed",mode:"flash",text[,utterances,raw]}` |

`responseFormat`：`"text"` 只回文本；`"standard"`（默认）文本+元信息；`"full"` 再加 utterances/raw。

标准版固定字段（对齐沙盒）：`enable_channel_split:false`、`vad_segment:false`（submit/recognize）；
flash 的 `vad_segment` 可传。

## 7. 错误处理

- 缺 `apiKey` → 返回 `{ok:false, code:"NO_API_KEY", message}`，不抛。
- 缺 `audioUrl`/`requestId`/`format` → `{ok:false, code:"BAD_REQUEST", message}`。
- submit/flash 非成功状态码 → `{ok:false, statusCode, httpStatus, message}`。
- query 非成功且非处理中 → 终态错误返回。
- 轮询超时 → `{ok:true, status:"timeout", hint}`（任务可能仍未完成，可再次查询）。
- 网络异常 → `errorResult`。
- `apiKey` 不写入日志/结果（test 里 mask）。

## 8. Open/MCP 暴露

`openExpose`：`submit_transcription`、`get_transcription`(readOnly)、`recognize`、`recognize_flash`。
工具名在 meta.tools 与 middleware 中保持裸名（与 storage/webhooks 一致），由框架按 domain 前缀。

## 9. 测试策略（jest + mock fetch，无网络）

- 注册冒烟：`PluginRegistry.register` 收到 `volcanoAsrPlugin`；`meta.type === "volcano-asr"`。
- 工具集合恰为四个；`openExpose` 名字 ⊆ middleware 工具名（防漂移），且 `get_transcription` readOnly。
- 连接解析优先级：pre-resolved → 选择器（`resolvePluginConnections` 收到 `"volcano-asr"`）→ env/默认。
- `submit_transcription`：正确 path/headers(`X-Api-Key`/`X-Api-Resource-Id`/`X-Api-Sequence:-1`)/body；成功回 requestId；失败回错误。
- `get_transcription`：处理中→成功；终态错误；超时；`responseFormat` 三态。
- `recognize`：submit + 轮询到成功。
- `recognize_flash`：固定 turbo resource；缺 `format` 报错；成功回文本。
- `connection.test`：成功 / 鉴权失败 / 网络异常的 message 与 details。

## 10. 开放问题 / 风险

1. **MCP 连接解析**：MCP 路径的 `_resolvedConnections` / tenant 注入需运行期验证（与 platform_service §15.4 同一风险）；失败时 Open 面工具会缺 apiKey。已有 fallback 链缓解。
2. **resourceId 语义**：标准版 2.0 用 `volc.seedasr.auc`；若账号仍是 1.0，需在连接里改 `volc.bigasr.auc`。默认取 2.0。
3. **计费**：submit/recognize 会真实创建识别任务并计费；连接测试同样会提交一次示例音频，需注意。
