---
kb_id: hankel-caren-project-won-validation
tenant_id: hankel
domain: caren
status: curated
tags: [project, new-order, won-validation, coverage, run-for-gold]
---

# Caren Project 与 Won→New Order 验证口径

## 当前客户确认范围

客户最终答复确认的核心是 Won→New Order 的 50% 验证，不是排名和奖项。核心指标为：

- Check-period Won Y1
- Matched New Order Value
- Required New Order Value
- New Order Gap
- Coverage
- Pass / Below 50%

Run for Gold 排名、Final Score、奖项和 Sales Edition 当前属于书面规则或 Demo reference，不应描述为客户最终确认的 POC 范围。

## 运行参数与时间

| 参数 | 规则 |
|---|---|
| Target Year | 运行时明确，例如 2026 |
| Competition Start | 当前 Demo 为 `2026-07-01` |
| Report Cut-off | 每次运行由用户手工提供，例如 `2026-08-31` |
| New Project 范围 | `Competition Start <= Creation Date <= Report Cut-off` |
| Won 验证范围 | `Status='Won'`；起始日期存在版本差异，见下方说明 |
| New Order 范围 | 目标自然年 `01-01` 至 `12-31`，不受 Report Cut-off 限制 |

`competition.yaml` 中“New Order 从 2026-07-01 起且无结束上限”是被客户最终答复覆盖的旧口径，不得使用。

Won 候选范围没有出现在客户最终答复中。旧 KPI 文件要求 `Close Year=目标年` 且 `Close Date >= Competition Start`，当前 runtime 使用目标年 `01-01` 至 Report Cut-off。该差异必须由业务确认；在确认前，回答应明确报告实际采用的日期范围。

## 身份标准化

1. Project 和 New Order 的原始销售姓名都必须通过 `Sales_Name_Mapping` 转成 Canonical Sales Name。
2. 匹配只允许维护表中的精确映射；大小写和空白可按已定义规则标准化。
3. 未映射姓名不强行匹配，进入异常清单。
4. Sold-to IDH 和 Product IDH 按标识符比较，清理 Excel `.0` 后必须完全一致。
5. 任一 Match Key 组成字段为空或非法时，不参与自动验证，进入异常清单。

## Match Key

```text
Canonical Sales Name + Sold-to IDH + Product IDH
```

三个字段必须全部完全一致。每个 Match Key 只进行一次金额覆盖验证，防止同一 New Order 金额重复用于多个机会组合。

不得使用以下方式替代 Match Key：

- 只按 Opportunity ID 匹配订单；
- 只按 Sold-to 或产品匹配；
- 按姓名模糊相似度匹配；
- 按金额相等或接近匹配；
- 使用 Project Segment 和 New Order Segment 连接。两者含义不同。

## Project 与 New Order 行规则

- Project 事实粒度为 Opportunity × Product。
- New Project Count 按 `DISTINCT Opportunity ID`，不按产品行计数。
- New Project Y1 必须先按销售 + Opportunity ID 对所有产品行执行 `SUM(y1_value)`，再向销售人员或团队粒度汇总。不得使用 `MAX(y1_value)` 代替产品行求和。
- New Order 使用 `order_value_cny`。
- `Reject = X/rejected` 的订单行排除。
- 除明确 Reject 外，不根据 Confirmed、Shipped、Delivered、Open、Blocked 等 Line Status 额外排除。

## 50% 验证公式

对于每个符合条件的 Won 行：

```text
Close Year Total Months = 13 - month(Close Date)
Recognized Months = max(0, min(Close Year Total Months,
                    Close Month 到 Report Cut-off Month 的含首尾月数))
Check-period Won Y1 = Won Y1 / Close Year Total Months * Recognized Months
```

在 Match Key 粒度聚合后：

```text
Required New Order Value = Check-period Won Y1 * 0.5
Matched New Order Value  = 目标自然年内匹配订单金额之和
Coverage                 = Matched New Order Value / Check-period Won Y1
Pass                     = Matched New Order Value >= Required New Order Value
Below 50%                = Matched New Order Value < Required New Order Value
```

`Check-period Won Y1 = 0` 时 Coverage 返回 null，不能返回 0% 或无限值。

Coverage 不封顶。超过 100% 表示订单金额高于折算 Won Y1，不代表计算错误。

## Validated Won Count

Validated Won Count 的粒度是 Opportunity，不是 Match Key：

```text
Validated Won Count = DISTINCTCOUNT(Opportunity ID)
                      where at least one exact product Match Key = Pass
```

同一 Opportunity 即使包含多个通过的产品 Match Key，也只计一次；同一 Match Key 聚合了多个 Opportunity 时，每个 Opportunity 分别计入。Match Key 数量只能作为技术诊断指标，不能替代 Validated Won Count。

## Gap 口径差异

客户 Ontology 的确认公式允许保留负值：

```text
Signed Gap = Required New Order Value - Matched New Order Value
```

当前 Metrics runtime 的主指标使用行动缺口：

```text
Action Gap = max(Required New Order Value - Matched New Order Value, 0)
```

底层 view 同时保留 `new_order_gap_raw`。Agent 必须区分：

- 回答“还差多少订单”使用 Action Gap；
- 回答“超额/不足多少”使用 Signed Gap；
- 对外固化单一口径前需客户确认。

## Golden Case

已确认验收案例：

| 字段 | 值 |
|---|---:|
| Team | Shanghai |
| Sales | 测试销售人员（已脱敏） |
| Sold-to | 463856 |
| Product IDH | 2982627 |
| Check-period Won Y1 | 12,487 |
| Required New Order（展示） | 6,243 |
| Matched New Order Value | 9,524 |
| Coverage | 76.3% |
| Result | Pass |

Pass/Below 比较必须使用未取整的精确 Required 值。`6,243` 只是展示结果，具体展示取整函数尚未确认。

## 排名规则（参考，不属于当前客户确认 POC）

```text
Rank Score = (N - Rank + 1) / N * 100
Final Score = 30% * New Y1 Rank Score + 70% * Won Y1 Rank Score
```

- `N` 是奖池实际入围人数。
- 使用降序 `RANK.EQ`。
- 并列顺序：Won Y1、Validated Won Count、New Y1、New Count、人工复核。
- Final Score 是销售人员粒度的非可加总指标，查询必须按 Canonical Sales Name 分组。
- Segment Final Score 必须按 Segment + Canonical Sales Name 分组。
- Check-period Won Y1 只用于验证，不用于 Final Score 排名；排名使用验证通过后的原始 Won Y1。
- 未达到 New Project Count 和 Validated Won Count 门槛的人员保留在资格状态视图中，但不得进入排行榜或获得金银铜位置。
