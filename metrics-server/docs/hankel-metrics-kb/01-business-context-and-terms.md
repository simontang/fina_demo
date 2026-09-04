---
kb_id: hankel-business-context-and-terms
tenant_id: hankel
domain: shared
status: curated
tags: [glossary, grain, identity, dimensions, joins]
---

# 业务上下文与通用口径

## 业务域边界

Hankel ACM 当前包含三组语义资产：

| 业务域 | 事实 | 用途 |
|---|---|---|
| River | Sell-in、Sell-out、Inventory | 经销商进销存和终端客户分析 |
| Caren | Project、New Order、Sales Name Mapping | Won 项目订单覆盖验证 |
| Generic Project Dashboard | Project | 项目经营分析 |

River 与 Caren 当前没有经过客户确认的跨域归因键。不能把 Project 直接 Join 到 Sell-in 或 Sell-out 后计算收入归因。未来需要补充 `sold_to/Product IDH -> Combined ID/Product master` 映射及 Sell-out 归因规则。

## 事实粒度

| 事实 | 业务粒度 | 类型 | 安全比较粒度 |
|---|---|---|---|
| Sell-in | 一条导入的经销商订货/销售行 | 流量 | 月份、Sales Type |
| Sell-out | 一条经销商向终端客户销售的分摊行 | 流量 | 月份、Sales Type |
| Inventory | 经销商 × 产品 × 月份 × 来源 × Sales Team 分摊 | 月末存量快照 | 单一月份、Sales Type |
| Project | Opportunity × Product 行 | 项目事实 | Opportunity ID、Product IDH |
| New Order | Sales Document × Item 行 | 订单事实 | Sales Document、Item |

库存是存量，不能跨月份求和。Sell-in 和 Sell-out 是流量，可以按期间求和。

## 关联边界

- Sell-in、Sell-out、Inventory 是相互独立且多对多的事实。
- 禁止在明细粒度直接连接三张 River 事实表。
- 跨事实比较时，各事实先按自己的正确口径汇总，再按月份和 Sales Type 对齐。
- 单事实“最新”取该事实的最新完整月份。
- 跨事实“最新”尚未确认使用共同最新月还是各自最新月；必须展示每个事实的数据截至月，必要时询问用户。
- Project 与 New Order 通过精确 Match Key 关联，不通过金额、描述或近似姓名关联。

## 关键术语消歧

| 用户用语 | 解释规则 |
|---|---|
| 销售额 / Sales amount | 歧义词。必须区分 Sell-in amount（NES）与 Sell-out amount（Territory Sell-out）。 |
| 客户 / Customer | 日常语境默认指 End Customer。指经销商时应明确说 Distributor、Sold-to 或经销商。 |
| 经销商 / Distributor | 向 Hankel 订货并向终端客户销售的渠道客户，统一键为 Combined ID。 |
| Sales region / 销售区域 | 在跨事实分析语境中等同 Sales Type；不要和原始 Sales Team 或 ACM territory 混用。 |
| Territory | Sell-out 和 Inventory 唯一允许用于业务汇总的金额/数量分摊口径。 |
| TA | 客户答复中等同 SPL。 |
| Report Cut-off | 每次运行由用户提供的报告截止日；不限制 New Order 的目标自然年范围。 |
| Check-period Won Y1 | Won Y1 按 Close Date 到 Report Cut-off 折算的验证基数，只用于 50% 验证。 |
| Coverage | Matched New Order Value / Check-period Won Y1，不是 New Order / Required。 |

## 主数据和身份解析

| 对象 | 统一键 | 允许的匹配方法 | 禁止做法 |
|---|---|---|---|
| Distributor | Combined ID | 编码精确匹配；名称由 lookup 补充 | 仅按经销商名称合并 |
| Product | 各事实对应的 Product IDH | 编码精确 lookup | 按产品描述自动合并 |
| End Customer | 当前暂用规范化名称 | 规范化后完全一致，并检查城市等同名碰撞 | 模糊匹配后自动合并 |
| Sales | Canonical Sales Name | 使用维护的 Sales_Name_Mapping 精确映射 | 中英文姓名直接互猜或近似匹配 |
| Calendar Month | 自然月第一天 | 有效年月解析 | 把无法解析的年月默认为当前月 |

所有业务编码按标识符处理。即使只包含数字，也不得参与算术；比较前去除 Excel 造成的无意义 `.0`，但保留前导零语义。

## 主要分析维度

客户明确关注的维度包括：

- 年月
- Sales Type
- 国标行业中类
- Distributor Combined ID 和经销商名称
- End Customer
- Product IDH、产品名称、Product Category
- `Henkel_keyMidInd`
- `data_source`、Platform
- `whether_DKA`、Grades、`whether_SPL`
- ACM Application L2

分类维度可能一条事实对应多个分类，因此分类汇总不一定能再次加总为总计。Agent 必须提醒非可加总风险。

Sales Type 是跨 River 事实比较的统一维度，通过维护的映射分别对应 Sell-in Team、Sell-out Team 和 Inventory Team。IPR 等场景可能一对多，不能假设每个 Sales Type 只对应一个来源团队。
