# SAP B1 插件封装修复设计

日期: 2026-08-06
范围: `agent/src/agents/sap_b1/`（生产只加载 `plugin.ts`；`tools.ts` 为死代码）

## 背景与问题

当前 `sap_api_call` 工具存在系统性缺陷，导致大量 curl 能返回的 API 用工具调用失败：

1. **默认 `$select` 注入 + 硬编码字段表不可信**（根因）：LLM 未显式传 `$select` 时，工具强制注入整份硬编码字段（`plugin.ts:814-837`）。字段表存在已确认错误的条目（`Z20_*` 六张 UDT 表字段 `DocEntry/DocNum/Period/Instance/...` 不存在于 B1 UDT 实体，UDT 只有 `Code/Name` + `U_` 前缀 UDF），任一字段错误 → SL 400。curl 不带 `$select` 反而成功。
2. **FunctionImport 按 EntitySet 处理**：`SBOBobService_GetCurrencyRate` 等函数导入在真实元数据中均为 POST + 必填参数，工具允许 GET 并注入 `$top=20` → 400。
3. **主键路径误导**：schema 鼓励 `id`（`/Orders('1173')`），描述又警告易 500，自相矛盾。
4. **URL 编码不完整**：仅 `'`→`%27`；`&`/`#`/`+` 在 filter 值中未编码 → 参数拆分/按空格解码。
5. **响应超载**：默认 `$select` 含导航集合（`DocumentLines/BPAddresses/ItemPrices/...`）×`$top=20`，裁剪只覆盖 `DocumentLines/DocumentAdditionalExpenses` → token 截断。
6. **认证模型与代理链路不匹配**：生产走 `b1s.alphafina.cn` 代理，代理剥 Cookie、用服务端账号自动登录，按 `tenantId+companyDb` 键控会话。插件无 `X-Company-DB`/`X-Tenant-Id` 支持 → 永远查默认公司 SBODemoUS；cookie 机制（含硬编码 `ROUTEID=.node0`）完全无效。
7. **死代码与测试漂移**：`tools.ts`（registerToolLattice 旧实现）与 `plugin.ts` 双份重复；31 个测试全部测 `tools.ts`，`plugin.ts` 零覆盖。
8. **裁剪不一致**：`lineFields` 宣传 `Price`，`EXPAND_FIELDS.DocumentLines` keep 表缺失。
9. **分页丢失**：`odata.nextLink` 被删除。
10. **400 错误无 hint**（SL 最常见错误反而没有指引）。

## 设计目标

让工具行为与 curl 等价可靠（curl 能通的 API 工具必通），同时保留「默认 `$select` 防 metadata 全量返回」的机制，但注入内容改为经过验证的保守白名单。

## 架构与组件

保留两工具结构不变（`sap_api_search` + `sap_api_call`），修复内部实现。新增独立纯函数模块便于测试。

### 组件 1：元数据（`API_LIST` 修正）

- 以仓库已有 `agent/src/agents/sap_b1/XSM_ZSK_metadata_interfaces.md`（真实 1099 接口）为事实源，校验全部实体名、实体类型、FunctionImport 的 POST-only 属性。
- `Z20_*` 六张 UDT 表（`Z20_COST/Z20_CPAT/Z20_OINP/Z20_PWAG/Z20_HOLD/Z20_IMIT`）字段改为 `["Code", "Name"]`；描述注明 UDF 字段为 `U_` 前缀，需先查 `$metadata` 或 search 返回的字段。
- 删除/修正可疑字段：`Items.InventoryUOM`、`PriceLists.GroupNum`、`InventoryPostings.PriceList`、`BinLocations.Sublevel1-4`（导航属性，从标量字段列表移除）。
- 原则：**任何无法从现有资料（metadata md、代码内已知用法）确认存在的字段，一律从 `fields` 移除**，确保默认注入永不携带未验证字段。LLM 需要时可在 `$select` 中显式传入。
- `API_LIST` 以修正后的 `fields` 为准（保持简单，不引入额外标记）。

### 组件 2：默认 `$select` 保守白名单（核心变更）

- 新增 `DEFAULT_SELECT_BY_CLASS`：按实体类别提供经过验证的高频字段子集：
  - Document（所有 `*Document*` 类型实体）：`DocEntry,DocNum,DocDate,CardCode,CardName,DocTotal,DocCurrency,DocumentStatus`
  - BusinessPartner：`CardCode,CardName,CardType,GroupCode,Phone1,Valid`
  - Item：`ItemCode,ItemName,ItemsGroupCode,QuantityOnStock`
  - Warehouse/库存：`WarehouseCode,WarehouseName,DocEntry,DocNum,DocDate`
  - UDT：`Code,Name`
- `applyDefaultSelect` 逻辑改为：
  - 仅 `kind === "EntitySet"` 的 GET 且 LLM 未传 `$select` 时注入 `$select=<该实体类别的白名单>`；
  - `$top=20` 注入逻辑保留（仅 EntitySet、无 id、无 `$top`）；
  - **任何情况下不再整表注入 `fields`**。
- `sap_api_search` 返回的 `readySelect` 同步改为白名单子集，`fields` 完整列表保留供 LLM 自选字段时参考。

### 组件 3：URL 构造与编码（纯函数 `buildUrl`/`encodeQueryOptions`）

- 新增 `encodeQueryOptions(queryOptions)`：按 `&` 分段，段内对引号包裹的值做安全编码：`'`→`%27`、`&`→`%26`、`#`→`%23`、`+`→`%2B`；`$filter`/`contains`/`eq` 等 OData 语法关键字不做处理。
- **幂等编码（关键）**：LLM 输入可能是原始表达式或已部分编码的混合形态，编码前识别已存在的 `%xx` 十六进制序列并跳过，禁止双重编码（`%27` 不得变成 `%2527`）。
- **工具描述规则**：明确要求 LLM 传原始 OData 表达式（裸单引号、裸 `&`），编码全部由工具负责，禁止 LLM 自行 encode。
- `id` 仅用于 PATCH/DELETE；GET 不再接受 `id`（schema 描述改为「查询请用 `$filter`」），消除 `/Orders('1173')` 500 路径。
- `entitySet` 用 `encodeURIComponent` 处理（已有）。

### 组件 4：FunctionImport 隔离

- `applyDefaultSelect` 对 `kind === "FunctionImport"` 的 GET 不注入任何参数；schema 的 `method` 描述明确「FunctionImport 一律 POST，参数放 `body.parameters`」。
- `sap_api_search` 对 FunctionImport 结果增加 `callHint`：`POST /{name}，参数体 {parameters: {...}}`。

### 组件 5：认证与公司上下文（代理链路优先）

- 连接字段（`connection.fields`）增加可选 `companyDb`、`tenantId`。
- 执行时：若配置了 `companyDb`/`tenantId`，请求头加 `X-Company-DB` / `X-Tenant-Id`。
- Cookie 逻辑：若 `cookie` 是完整 `B1SESSION=...; ROUTEID=...` 原样使用；否则拼 `B1SESSION=<值>`（去掉硬编码 `ROUTEID=.node0`）。
- 描述更新：走代理（`b1s.alphafina.cn`）时 Cookie 无效，认证由代理管理，公司切换靠 `companyDb` 连接配置。

### 组件 6：响应裁剪

- `EXPAND_FIELDS.DocumentLines` keep 表补 `Price`（与 `lineFields` 对齐）。
- 为 `ItemPrices`、`BPAddresses`、`ContactEmployees`、`ItemWarehouseInfoCollection` 增加裁剪白名单。
- 保留 `odata.nextLink`，在结果中返回 `hasMore` 提示（附「可用 $skip 取下一页」指引）。

### 组件 7：错误处理

- 400/500 时解析 SL `error.message.value` 放入 `hint`，附「字段可能不存在，用 sap_api_search 核对字段名」指引。

### 组件 8：测试与死代码清理

- 删除 `tools.ts`（死代码），`index.ts` 保持只 import `./plugin`。
- 从 `plugin.ts` 导出纯函数（`buildUrl`、`encodeQueryOptions`、`applyDefaultSelect`、`cleanODataNoise`、`trimNestedCollections`）。
- 测试改为 import `../plugin`：
  - 纯函数单测：URL 构造、编码（`'` `&` `#` `+`）、默认 select 注入（EntitySet/FunctionImport/UDT 分支）、裁剪。
  - `sap_api_call` 用 mock fetch 覆盖：401/404/400/成功、POST/PATCH/DELETE、代理头注入。
  - 保留现有 `sap_api_search` 语义测试（改为测 `plugin.ts` 内的 executor）。

## 数据流

```
LLM → sap_api_call(entitySet, method, queryOptions, body)
  → 配置解析（_resolvedConnections[0].config: baseUrl/cookie/companyDb/tenantId）
  → applyDefaultSelect（仅 EntitySet+GET：注入白名单 $select、$top=20）
  → encodeQueryOptions（安全编码 filter 值）
  → buildUrl → fetch(headers: X-Company-DB/X-Tenant-Id/Cookie 兜底)
  → cleanODataNoise + trimNestedCollections
  → 统一结果 { ok, status, data, hint }
```

## 错误处理

- 400：hint 含 SL 具体错误 + 字段核对指引。
- 401：hint「走代理则检查代理账号/公司配置；直连则检查 Cookie」。
- 404：hint「检查 entitySet 名称（用 sap_api_search 核对）与 id」。
- 网络错误：返回 url + 提示检查 BASE_URL。

## 测试策略

- 单元测试（jest，mock fetch，无网络）：覆盖组件 2/3/4/6 的全部纯函数分支。
- 现有 31 个用例迁移到 `plugin.ts` 后保持通过（`sap_api_call` 联网用例改为 mock）。
- 可选联调用例（跳过 `$top`/`$select` 默认注入，直连代理验证真实字段）手动执行。

## 不做的事（YAGNI）

- 不引入运行时 `$metadata` 拉取缓存（方案 B 暂缓）。
- 不拆分强类型专用工具。
- 不做自动纠错重试（400 自动回退字段重试）。
- 不改 `sap_api_search` 的 schema（仅改返回内容与描述）。
