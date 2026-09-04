---
kb_id: hankel-analysis-governance-and-open-questions
tenant_id: hankel
domain: governance
status: curated
tags: [quality, privacy, availability, open-questions, runtime-alignment]
---

# 分析治理、数据质量与待确认项

## 查询与输出边界

- 当前分析动作是只读操作，不修改源数据或业务系统。
- 普通 Agent 只能查询已发布的 table meta 和 metric meta。
- 知识库描述业务含义，不构成数据表访问授权。
- 缺少字段、映射或确认口径时，应返回 `unavailable`、异常清单或待确认说明。
- 禁止为了给出答案而替换字段、猜测 Join key 或使用未发布 raw 表。

## 数据质量规则

| 问题 | 处理方式 |
|---|---|
| 空值、`nan`、`na`、`-` | 保留并标记，不静默转成 0 或删除 |
| 重复值 | 不能仅凭维度和值相同去重 |
| River 事实重复 | 只使用上游稳定行键或明确去重规则 |
| Sales 姓名未映射 | 进入异常清单，不做模糊自动匹配 |
| End Customer 同名 | 使用城市/编码检查碰撞，不能自动合并 |
| 无效 IDH/日期 | 排除自动匹配并报告原因 |
| 多产品单位 | 分开汇总或先完成单位换算，不能直接相加 |
| 多分类映射 | 允许非可加总，结果中说明 |
| 未分类产品 | 归 `Other`，不要猜测分类 |
| 异常金额/数量 | 保留审计值；Demo guardrail 与客户业务规则分开说明 |

## 隐私规则

- 销售姓名和终端客户明细只允许按需实时读取。
- 不把销售姓名或终端客户明细复制到持久化知识库、长期报告、日志或训练样本。
- 持久化输出优先使用团队、区域、行业和聚合指标。
- 确需明细时，应在授权会话中返回并遵守最小化原则。

## 当前已确认的关键口径

- Sales amount 必须区分 Sell-in NES 与 Territory Sell-out。
- Customer 默认指 End Customer。
- Sell-out 和 Inventory 只使用 Territory 字段。
- Inventory 是月末快照，默认金额，不能跨月求和。
- River 三事实先汇总再比较，禁止明细 Join。
- Distributor 使用 Combined ID；Product 使用各事实的 Product IDH。
- Caren 使用维护的 Sales Name Mapping 和三字段精确 Match Key。
- New Order 使用目标自然年，不受 Report Cut-off 限制。
- Required = Check-period Won Y1 × 50%。
- Coverage = Matched New Order / Check-period Won Y1。
- 精确值用于比较，取整只用于展示。

## 必须继续标注为待确认

| 主题 | 未决问题 | Agent 行为 |
|---|---|---|
| River 跨事实最新月 | 使用共同最新月还是各自最新月 | 展示各事实截至月并询问 |
| Gross Margin | 正式列来源和过滤规则 | 只使用已发布规范字段并标注待确认 |
| Product Contribution | `15*` 字段业务含义 | 可展示但不扩展解释 |
| Sell-through/库存周转/覆盖月数 | 窗口、金额或数量、零分母 | 不作为客户确认指标 |
| 活跃/非活跃客户 | 有效购买、退货、多经销商和阈值 | 不自动输出正式名单 |
| End Customer 唯一身份 | 目前名称可能重名 | 输出碰撞质量提示 |
| Project→River 归因 | 跨域映射键缺失 | 禁止数值归因 |
| Required 展示取整 | 黄金样例有整数，但函数未确认 | 判定用精确值 |
| New Order Gap | 允许负值还是仅展示正缺口 | 明确 Signed Gap/Action Gap |
| Won 候选起始日 | 旧 Ontology 使用 Competition Start，当前 runtime 使用目标年年初 | 每次回答披露实际日期范围，等待业务确认 |
| 排名和奖项 | 当前客户最终答复未纳入 POC | 标为 Demo/reference |

## Ontology 与当前 Runtime 的对齐结果

截至 2026-09-04，Runtime 已发布 Sell-in、Sell-out、Inventory 和 Run for Gold 指标，但“已发布”不等于“客户已确认”。

| 领域 | 对齐结果 | 说明 |
|---|---|---|
| Sell-in Quantity / NES | 已在线对齐 | View 应用 Team 白名单和 `GMM L6 allocation != Y`，并保留范围外审计值 |
| Sell-out | 已在线对齐 | Territory 金额和数量独立应用质量规则，正常负数保留 |
| Inventory | 对齐 | 使用最新有效快照和 Territory 分摊金额/数量，不跨月累计 |
| Validated Won Count | 已在线对齐 | 使用去重 Opportunity，至少一个精确 Match Key 通过即计入 |
| Final Score | 算法已在线对齐 | 只对已入围人员排名，分项使用 RANK.EQ；业务状态仍为 reference |
| Segment Final Score | 算法已在线对齐 | 只对已入围人员排名；业务状态仍为 reference |
| New Order Gap | 存在双口径 | Runtime 主指标为不小于 0 的 Action Gap，底层另有 Signed Gap |
| Generic Project Dashboard | 尚未完整发布 | Ontology 有 20 个书面规范指标，当前 23 个 Runtime 指标未覆盖整套 Dashboard |

修复前 Sell-in 全量数据的核对结果如下，用于保留变更证据，不作为当前结果：

| Measure | 当前 Runtime 全量 | 应用客户白名单和 GMM L6 规则 |
|---|---:|---:|
| NES | 1,656,594,630.73 | 1,607,161,170.43 |
| Sell-in Quantity | 17,675,680.187 | 17,673,540.187 |

2026-09-04 线上复测已返回右列结果，且范围外记录没有泄漏到标准 measure。

仍需持续检查：

1. New Order Gap 返回的是 Action Gap 还是 Signed Gap。
2. 所有 runtime 结果是否包含数据截至月和质量排除说明。
3. Runtime 新增指标是否在知识库中具有 `confirmed` 或明确的 `written-spec/reference` 状态。
4. River Sales Type 映射表是否已经由客户提供并发布；在此之前禁止跨事实 Sales Type 对比。
5. 固定 YTD Aug 2026 参数是否已被运行级参数替代。

发现实现不一致时，报告“runtime 与客户口径不一致”，不要修改知识库来迁就当前 SQL。

## 指标回答最低证据

每次指标回答至少包含：

```json
{
  "metric": "指标名称",
  "value": "精确结果",
  "timeRange": "查询范围",
  "groupBy": ["实际分组维度"],
  "filters": ["实际筛选条件"],
  "asOf": "数据截至日期或月份",
  "definitionStatus": "confirmed | written-spec | draft",
  "qualityNotes": [],
  "source": "已发布语义视图/指标"
}
```

这只是回答证据结构，不要求最终用户界面直接展示 JSON。
