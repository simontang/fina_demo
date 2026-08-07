# SAP B1 插件封装修复设计

日期: 2026-08-06
范围: `agent/src/agents/sap_b1/`（生产只加载 `plugin.ts`；`tools.ts` 为死代码）

## 背景与问题

当前 `sap_api_call` 工具存在系统性缺陷，导致大量 curl 能返回的 API 用工具调用失败：

1. **默认 `$select` 注入 + 硬编码字段表有错**（根因）：LLM 未显式传 `$select` 时，工具强制注入整份硬编码字段（`plugin.ts:814-837`），任一字段名不存在 → SL 400。curl 不带 `$select` 反而成功。
   **已通过真实 `$metadata` 联调验证（2026-08-06，`b1s.alphafina.cn/b1s/v1/$metadata`，SBODemoUS + XSM_ZSK 两公司）**，API_LIST 约 40 个实体中仅 6 处字段错误（见组件 1 实测表）。此前的推断（Z20 字段必错、`InventoryUOM/GroupNum/PriceList/Sublevel` 可疑）**经实测全部不成立**，Z20 六表字段真实存在。
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

### 组件 1：元数据（`API_LIST` 修正，基于实测）

- **事实源**：已通过真实 API 拉取两公司 `$metadata` 并固化为 fixture：`agent/src/agents/sap_b1/__fixtures__/sap_metadata.json`（含全部 EntityType 的 properties/navigation，310KB）。实施时新增「字段校验测试」：API_LIST 每个字段必须存在于对应 EntityType 的 fixture 中，防回归。
- **实测确认需修复的 6 处字段错误**：

| EntitySet | 错误字段 | 真实字段 |
|---|---|---|
| `Orders` | `RoundDif` | `RoundingDiffAmount` |
| `Drafts` | `U_YWLX, U_YWLX2` | 仅 XSM_ZSK 公司存在（SBODemoUS 无）→ 从默认字段移除 |
| `LandedCosts` | `DocNum, DocDate, CardCode, Comments` | `LandedCostNumber, PostingDate, VendorCode, Remarks` |
| `SalesPersons` | `Valid` | `Active`（另有 `Locked`） |
| `ItemGroups` | `CommissionGroup` | 不存在（该字段在 SalesPerson） |
| `BarCodes` | `ItemCode, BarCode` | `ItemNo, Barcode` |

- 其余字段（含 Z20 六表、`Items.InventoryUOM`、`PriceLists.GroupNum`、`InventoryPostings.PriceList`、`BinLocations.Sublevel1-4`）经实测**全部存在**，保留。
- 原则：字段增改一律以 fixture 校验；无法确认存在的字段不进入默认注入。

### 组件 2：默认 `$select` 保守白名单（核心变更）

- 新增 `DEFAULT_SELECT_BY_CLASS`：按实体类别提供经过验证的高频字段子集：
  - `document`（销售/采购单据类）：`DocEntry,DocNum,DocDate,CardCode,CardName,DocTotal,DocCurrency,DocumentStatus`
  - `inventory_doc`（InventoryGenEntries/InventoryGenExits）：`DocEntry,DocNum,DocDate`
  - `stock_transfer`（StockTransfers）：`DocEntry,DocNum,DocDate`
  - `business_partner`：`CardCode,CardName,CardType,GroupCode,Phone1,Valid`
  - `item`：`ItemCode,ItemName,ItemsGroupCode,QuantityOnStock`
  - `warehouse`：`WarehouseCode,WarehouseName`
  - `udt_doc`（Z20 系列自定义表，实测含单据类字段）：`DocEntry,DocNum,Period,Instance,Status,CreateDate`
- **交集规则（防再犯）**：实际注入值 = `DEFAULT_SELECT_BY_CLASS[class] ∩ 该实体修正后的 fields`（组件 1 已保证 fields 可信）。例如 `Warehouses` 注入后只剩 `WarehouseCode,WarehouseName`，`InventoryGenEntries` 只剩 `DocEntry,DocNum,DocDate`。类映射不到的任何实体 → **不注入 `$select`**（保持 `$top=20` 裸查）。
- `applyDefaultSelect` 逻辑改为：
  - 仅 `kind === "EntitySet"` 的 GET 且 LLM 未传 `$select` 时注入（按上述规则）；
  - `$top=20` 注入逻辑保留（仅 EntitySet、无 id、无 `$top`）；
  - **任何情况下不再整表注入 `fields`**。
- `sap_api_search` 返回的 `readySelect` 同步改为白名单子集，`fields` 完整列表保留供 LLM 自选字段时参考。

### 组件 3：URL 构造与编码（纯函数 `buildUrl`/`encodeQueryOptions`）

- 新增 `encodeQueryOptions(queryOptions)`：**只在引号外部的 `&` 处分段**（引号内的 `&` 是值的一部分）；段内对引号包裹的值做安全编码：`'`→`%27`、`&`→`%26`、`#`→`%23`、`+`→`%2B`；`$filter`/`contains`/`eq` 等 OData 语法关键字不做处理。分段失败（引号不配对等畸形输入）时整体编码值部分并继续。
- **幂等编码（关键）**：LLM 输入可能是原始表达式或已部分编码的混合形态，编码前识别已存在的 `%xx` 十六进制序列并跳过，禁止双重编码（`%27` 不得变成 `%2527`）。
- **工具描述规则**：明确要求 LLM 传原始 OData 表达式（裸单引号、裸 `&`），编码全部由工具负责，禁止 LLM 自行 encode。
- `id` 仅用于 PATCH/DELETE；GET 不再接受 `id`（schema 描述改为「查询请用 `$filter`」），消除 `/Orders('1173')` 500 路径。
- `entitySet` 用 `encodeURIComponent` 处理（已有）。

### 组件 4：FunctionImport 隔离

- `applyDefaultSelect` 对 `kind === "FunctionImport"` 的 GET 不注入任何参数；schema 的 `method` 描述明确「FunctionImport 一律 POST，参数放 `body.parameters`」。
- `sap_api_search` 对 FunctionImport 结果增加 `callHint`：`POST /{name}，参数体 {parameters: {...}}`。
- **参数体格式待实测确认**（仓库 metadata md 无参数明细）：实施时用 `SBOBobService_GetCurrencyRate` 联调一次，确认 body 是 `{parameters:{...}}` 还是平铺 JSON，再固化到 `callHint` 与描述。

### 组件 5：认证与公司上下文（代理链路优先）

- 连接字段（`connection.fields`）增加可选 `companyDb`、`tenantId`。
- 执行时：若配置了 `companyDb`/`tenantId`，请求头加 `X-Company-DB` / `X-Tenant-Id`。
- `connection.test` 同步使用连接配置的 `companyDb`/`tenantId` 发起测试请求。
- Cookie 逻辑：若 `cookie` 是完整 `B1SESSION=...; ROUTEID=...` 原样使用；否则拼 `B1SESSION=<值>`（去掉硬编码 `ROUTEID=.node0`）。
- 描述更新：走代理（`b1s.alphafina.cn`）时 Cookie 无效，认证由代理管理，公司切换靠 `companyDb` 连接配置。

### 组件 6：响应裁剪

- `EXPAND_FIELDS.DocumentLines` keep 表补 `Price`（与 `lineFields` 对齐）。
- 为 `ItemPrices`、`BPAddresses`、`ContactEmployees`、`ItemWarehouseInfoCollection` 增加裁剪白名单。
- 分页：不把上游 `odata.nextLink` 暴露给 LLM（内含上游完整 URL）；改为当 `value.length === $top` 时返回 `hasMore: true`，并附「结果已达 $top 上限，可用 $skip=N 取下一页」提示。

### 组件 7：错误处理

- 400/500 时解析 SL `error.message.value` 放入 `hint`，附「字段可能不存在，用 sap_api_search 核对字段名」指引。

### 组件 8：测试与死代码清理

- 删除 `tools.ts`（死代码），`index.ts` 保持只 import `./plugin`。实施前确认 jest.config / tsconfig 及其它文件无 `tools.ts` 引用。
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
- **字段校验测试（新增）**：API_LIST 每个字段对照 `__fixtures__/sap_metadata.json` 断言存在，字段表改动必须过此测试。
- 现有 31 个用例迁移到 `plugin.ts` 后保持通过（`sap_api_call` 联网用例改为 mock）。
- 可选联调用例（跳过 `$top`/`$select` 默认注入，直连代理验证真实字段）手动执行。
- `$metadata` 刷新：`curl "https://b1s.alphafina.cn/b1s/v1/$metadata" -H "X-Company-DB: <公司>"` 重新生成 fixture（含日期备注）。

## 不做的事（YAGNI）

- 不引入运行时 `$metadata` 拉取缓存（方案 B 暂缓）。
- 不拆分强类型专用工具。
- 不做自动纠错重试（400 自动回退字段重试）。
- 不改 `sap_api_search` 的 schema（仅改返回内容与描述）。
- **暂不纳入 UDF/自定义表**（`U_` 前缀字段、`Z20_*` UDT 实体已从 API_LIST 移除）：UDF 依赖真实 `$metadata` 按公司动态获取，当前场景用不到，聚焦内置实体；未来按需从 `__fixtures__/sap_metadata.json`（或实时 `$metadata`）接入。
