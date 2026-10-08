# 插件工具内联 HTML UI（MCP Apps）设计

- 日期：2026-10-08
- 状态：Draft（设计已定稿，待实现）
- 范围：
  - `agent/package.json` / `ai_web/package.json`（升级 `@axiom-lattice/*`）
  - `agent/src/agents/platform_service/business_objects/`（新增 `businessObjectsUi.ts` + 改 `plugin.ts`）
  - `agent/src/agents/platform_service/storage/`（新增 `storageUi.ts` + 改 `plugin.ts`）
  - `agent/src/agents/semantic_metrics/`（新增 `semanticMetricsUi.ts` + 改 `plugin.ts`）
  - 各插件 `__tests__/`
- 不改动：`platform-service/`（Java 接口与响应 DTO 不变）、各插件 `executors.ts` / `client.ts` / `connection.ts` 的业务逻辑

## 1. 背景与目标

fina_demo 的插件工具返回的都是 JSON 字符串，前端聊天里只能看到一大段 JSON，不直观。目标是让部分**只读、可表格化**的工具在聊天中额外渲染一个内联 HTML 表格（MCP App），提升可读性，同时不改变工具给模型返回的文本语义。

第一批（本期）：
- `business-objects`：`list_objects`、`query_records`
- `storage`：`list`
- `semantic-metrics`：`metrics_meta_tool`（list/read 动作）

目标形态：统一、干净、可读的只读表格；不同工具按其数据结构映射列。

## 2. 关键调研结论（约束）

1. **前端已支持渲染**：`@axiom-lattice/react-sdk`（ai_web 4.7.0）会解析工具结果里的 ` ```mcp_app ` fence，通过 `/api/mcp-apps/resource` 取 HTML，在沙箱 iframe 里渲染。
2. **core 不会给全局 TS 插件自动加 fence**：core 只对**沙箱插件**和 **MCP server 工具**自动追加 fence。fina_demo 的插件都是全局 TS 插件（`PluginRegistry.register`），必须在 handler 里**手动** `appendUiFence`。
3. **`appendUiFence` 已由 core 导出**（`@axiom-lattice/core`，6.3.3 起），签名 `appendUiFence(content, ref)`。
4. **Gateway 4.8.3 缺少 Console 内联分支**：`fetchPluginUiResource` 在 4.8.3 只会读沙箱插件文件；读取全局插件 `meta.uiResources` 内联 HTML 的能力由 commit `c85835ecd`（ADR-171，晚于 4.8.4）引入，**已发布的 gateway 4.9.0 包含**。因此**必须升级 gateway ≥ 4.9.0**。
5. **上游范例**：`/Users/simon/code/agentic`（`@axiom-lattice/*` 源码）里的内置 `sandbox-plugins` 插件是唯一的"全局插件 + 内联 HTML UI"范例：
   - 声明：`packages/core/src/middlewares/sandboxPluginsMiddleware.ts:166-187`（`meta.tools[].ui` + `meta.uiResources`）
   - 追加 fence：同文件 `:93-98`
   - HTML：`packages/core/src/middlewares/sandboxPluginsUi.ts:21,32-201`（只读表格 App）
   - Gateway 解析：`packages/gateway/src/routes/mcp-apps.ts:264-294`

## 3. 决策总览

| 项 | 决策 |
|---|---|
| 渲染机制 | 工具 handler 手动 `appendUiFence` + `meta.tools[].ui` + `meta.uiResources` |
| 依赖 | 升级 agent 与 ai_web 的 `@axiom-lattice/*` 到 latest（gateway ≥ 4.9.0） |
| 代码组织 | **每个插件一个自包含 `*Ui.ts`，不做任何跨插件共享代码**（插件未来可独立成 repo） |
| 样式统一 | 靠**同一份 HTML 模板约定**（复制到各 `*Ui.ts`），非共享模块 |
| 交互程度 | 只读展示（不排序/翻页/回调） |
| 数据映射 | 每工具显式列配置 + 动态兜底（行内未声明的 key 自动补列） |
| 资源粒度 | 每个工具一个 `ui://` 资源；`metrics_meta_tool` 单资源内自动区分列表/详情 |
| 失败处理 | 结果非成功 JSON（解析失败 / `ok:false` / `Error:`）时**不**追加 fence，原样返回 |
| 模型可见性 | fence 追加在返回文本尾部（与上游一致）；前端展示时会剥离，模型上下文会看到一小段 JSON（可接受，后续可优化） |

## 4. 机制

每个受影响工具：

1. 在插件 `meta.tools[]` 对应条目加：
   ```ts
   ui: { resource: "<UI_RESOURCE>", displayMode: "inline" }
   ```
2. 在 `meta.uiResources` 注册：
   ```ts
   uiResources: { [UI_RESOURCE]: { html: APP_HTML } }
   ```
3. handler 返回时追加 fence：
   ```ts
   const REF: McpUiRef = {
     kind: "plugin",
     pluginType: "<plugin.meta.type>",
     resource: "<UI_RESOURCE>",
     displayMode: "inline",
   };
   return appendUiFence(resultText, REF) as string;
   ```

`pluginType` 必须等于 `plugin.meta.type`；gateway 用 `meta.tools[].ui.resource` 找到工具、再用 `meta.uiResources[resource]` 取内联 HTML。

## 5. 自包含 `*Ui.ts` 契约

每个 `*Ui.ts` 导出自包含内容（仅依赖框架层 `@axiom-lattice/core` / `@axiom-lattice/protocols`，不 import 其它插件）。一个文件可导出**一个或多个**资源，每个资源一组：

```ts
export const <NAME>_UI_RESOURCE = "ui://<plugin-type>/<name>";
export const <NAME>_APP_HTML = `<!doctype html> ...自包含 HTML... `;
export function with<Name>Ui(result: string): string; // 解析失败/ok:false/Error: 原样返回，否则 appendUiFence
```

`with<Name>Ui` 内部持有该资源的 `McpUiRef`（`{ kind:"plugin", pluginType: meta.type, resource, displayMode:"inline" }`），调用方无需暴露 ref。business-objects 一个文件导出 `OBJECTS_*` 与 `RECORDS_*` 两组（含 `withObjectsUi` / `withRecordsUi`）。

### 5.1 规范模板（各 `*Ui.ts` 复制，只改内嵌 `CONFIG`）

所有 App 使用同一骨架（源自上游 `sandboxPluginsUi.ts`），保证样式一致：

- **握手顺序（关键，不可调换）**：
  1. 先注册 `message` 监听（接收 `ui/notifications/tool-result`）
  2. 再 `rpc("ui/initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo, appCapabilities: { availableDisplayModes: ["inline"] } })`
  3. 成功后 `notify("ui/notifications/initialized", {})`
- **取数**：`readResult(params)` 优先 `params.structuredContent`，否则解析 `params.content[0].text` 的 JSON（fence 已被 host 剥离）。
- **高度**：`notify("ui/notifications/size-changed", { width, height })` + `ResizeObserver`。
- **安全**：严格 CSP、opaque origin；无外部资源；所有插值经 `esc()` 转义；只读，不调用 `tools/call`。
- **内嵌配置**：`const CONFIG = { ... }`（JSON 字面量），由各工具填充。

内嵌 `CONFIG` 结构：

```ts
interface TableAppConfig {
  title: string;
  subtitle?: string;
  rowsPath?: string;          // 如 "files" / "rows" / "items"；缺省自动探测
  columns?: ColumnSpec[];     // 显式列（key/label/format）
  dynamic?: boolean;          // 默认 true：行内未声明 key 自动补列
  summary?: string[];         // 顶层字段，渲染成一行摘要（如 total/page）
  emptyText?: string;
  detail?: boolean;           // 单对象 → 键值详情卡（缺省按数据形状自动判断）
}
interface ColumnSpec { key: string; label?: string; format?: Format; }
type Format = "text" | "mono" | "number" | "bytes" | "datetime" | "status" | "bool" | "json";
```

渲染规则：
- 解析出的目标是数组（或 `rowsPath` 命中数组）→ 表格；显式列按序在前，`dynamic` 时补未声明列。
- 目标是对象且 `detail` → 键值卡；否则单行表格。
- 空数组 / 取不到 → 显示 `emptyText`。
- 格式器：`bytes` 人类可读（KB/MB）、`datetime` 本地化、`status` 圆点标签、`mono` 等宽、`json` 截断、`bool` ✓/✗。

样式：表头大写小字、行底分隔、`word-break`、`max-width` 截断、`prefers-color-scheme: dark` 适配、无边框卡片。

## 6. 各插件设计

### 6.1 business-objects

新增 `business_objects/businessObjectsUi.ts`，导出 2 个资源：

- `OBJECTS_UI_RESOURCE = "ui://business-objects/objects"`
  - 数据：`list_objects` → `ObjectDefinitionResponse[]`（顶层数组）
  - 列：`objectKey`(mono)、`displayName`、`tableName`(mono)、`storeKey`(mono)、`status`、`deleteMode`、`fields`(数量)、`indexes`(数量)、`description`；`dynamic: false`
  - `title: "Business Objects"`，`emptyText: "No objects"`
- `RECORDS_UI_RESOURCE = "ui://business-objects/records"`
  - 数据：`query_records` → `QueryResponse{ objectKey, page, pageSize, total, rows[] }`
  - `rowsPath: "rows"`，`summary: ["objectKey","page","pageSize","total"]`
  - 列：`id`(mono) 显式在前；`dynamic: true`（记录字段动态）
  - `title: "Records"`

`plugin.ts` 改动：
- `meta.tools` 中 `list_objects` / `query_records` 两条加 `ui`。
- 加 `meta.uiResources`（两个 key）。
- middleware 里包结果：
  ```ts
  const out = await boObjectList(input, exeConfig, pluginConfig);
  return withObjectsUi(out);
  ```
  `boRecordQuery` 同理用 `withRecordsUi`。

### 6.2 storage

新增 `storage/storageUi.ts`，导出 1 个资源：

- `FILES_UI_RESOURCE = "ui://storage/files"`
  - 数据：`storageList` → `PathListing{ path, recursive, query, directories[], files[], page, size, total, totalPages }`
  - `rowsPath: "files"`，`summary: ["path","total","page","size","totalPages"]`
  - 列：`filename`、`fullPath`、`size`(bytes)、`mime`、`fileCategory`、`usage`、`version`、`status`、`createdAt`(datetime)、`uuid`(mono，截断)
  - `title: "Files"`，`emptyText: "No files"`

`plugin.ts` 改动：
- **补全 `meta.tools`**：当前 storage 没有 `meta.tools`，需列出全部 5 个工具（`upload`/`list`/`get_metadata`/`get_download_url`/`delete`），`ui` 只加在 `list` 上（避免前端 allowedTools 面板退化）。
- 加 `meta.uiResources`。
- middleware 里 `withFilesUi(await storageList(...))`。

### 6.3 semantic-metrics

新增 `semantic_metrics/semanticMetricsUi.ts`，导出 1 个资源：

- `META_UI_RESOURCE = "ui://semantic-metrics/meta"`
  - 数据（按 action）：
    - `list_tables` / `list_metrics` → `{ items[], total }` → 表格（`rowsPath: "items"`，`dynamic: true`，`summary: ["total"]`）
    - `read_table_meta` / `read_metric_meta` → 单对象 → 键值详情卡（`detail: true`）
  - App 自动按数据形状区分：命中 `items` 数组渲染表格，否则渲染详情卡。
  - `title: "Semantic Meta"`

`plugin.ts` 改动：
- `meta.tools` 的 `metrics_meta_tool` 加 `ui`。
- 加 `meta.uiResources`。
- `metrics_meta_tool` handler 内，仅对只读动作追加 fence：
  ```ts
  const text = JSON.stringify(...); // 现有返回值
  return isReadAction ? withMetaUi(text) : text;
  // isReadAction = list_tables | list_metrics | read_table_meta | read_metric_meta
  ```
  写动作（`create_*` / `update_*`）与异常分支不追加。

## 7. 依赖升级

- agent：`pnpm run up_lattice`（`@axiom-lattice/core|gateway|pg-stores|protocols` → latest），随后重新构建（`pnpm build`）。
- ai_web：`pnpm run up_lattice`（`@axiom-lattice/react-sdk|protocols` → latest），随后重新构建。
- 必须确认升级后 `@axiom-lattice/gateway` ≥ 4.9.0（含 `fetchPluginUiResource` 内联分支）。

## 8. 测试

各插件 `__tests__/` 新增/更新：

- `*Ui.ts`：
  - `APP_HTML` 含握手标记（`ui/initialize`、`ui/notifications/initialized`、`ui/notifications/tool-result`、`ui/notifications/size-changed`）与内嵌 `CONFIG`（title/rowsPath/columns）。
  - `with*Ui`：成功 JSON 追加 ` ```mcp_app ` 且 ref 的 `pluginType`/`resource` 正确；`Error:` / `{ok:false}` / 非 JSON 原样返回。
- `plugin.ts` 不变量：
  - 每个 `meta.uiResources` key 都能在 `meta.tools[].ui.resource` 找到（一一对应）。
  - `pluginType` 与 `meta.type` 一致。
  - storage 的 `meta.tools` 含全部 5 个工具名。
- 更新受影响的既有测试（如 registration/barrel）以包含新增的 `uiResources`。

## 9. 验证

1. 升级依赖并构建 agent / ai_web。
2. 本地起 agent，前端聊天里依次触发：`list_objects`、`query_records`、`list`、`metrics_meta_tool`（list 与 read）。
3. 确认：内联表格渲染、样式统一、空结果/错误不渲染、高度自适应、明暗主题正常。
4. 确认模型仍能拿到原始 JSON（fence 只影响前端展示）。

## 10. 风险与开放项

- **依赖升级风险**：gateway/core 跨小版本升级可能带来行为变化；升级后需跑 agent 现有测试与一次手工冒烟。
- **模型上下文污染**：fence JSON 会进入模型可见的工具结果文本（与上游沙箱插件一致）。若后续要消除，可改为 `content_and_artifact` 或框架层剥离（本期不做）。
- **列随服务端漂移**：显式列基于当前 DTO；服务端字段变更时靠 `dynamic` 兜底显示新字段。
- **`displayMode`/`permissions`/`prefersBorder`**：当前 host 解析但未完全生效（`displayMode` 默认 inline）；本期只依赖 inline。
- **metrics 列表项结构**：`items[]` 的具体字段由 Metrics Server 定义，本期用 `dynamic` 列兜底，不写死列。
