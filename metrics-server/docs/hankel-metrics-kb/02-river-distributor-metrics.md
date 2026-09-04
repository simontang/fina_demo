---
kb_id: hankel-river-distributor-metrics
tenant_id: hankel
domain: river
status: curated
tags: [sell-in, sell-out, inventory, distributor, territory]
---

# River 经销商进销存指标口径

## 强制规则

1. Sell-out 金额和数量只能使用 Territory 字段。
2. Inventory 金额和数量只能使用 Territory 分摊字段。
3. Inventory 默认指金额；用户问“库存数量”时才返回数量。
4. Sell-in、Sell-out、Inventory 必须独立汇总，不能明细 Join。
5. 相同维度和相同数值不代表重复交易，不得据此去重。
6. `data_source` 的不同取值由客户声明为不重叠来源，应全部保留。

`TA` 在客户答复中等同 `SPL`。`GM SPL` 可由以下 `whether_SPL` 值识别：`SPL_Frekote_prod`、`SPL_Sounderhoff_prod`、`SPL_PA_prod`、`SPL_Loudspeaker`。当前 POC 不对 DKA/NCA 额外排除，除非具体指标另有明确规则。

当前 River POC 选择的三个测试指标是 Sell-in Quantity、Sell-out Quantity 和 Inventory Quantity。客户同时确认“Inventory”无修饰时默认指金额，因此 Inventory Value 可以用于解释和参考，但不得描述为当前 POC 已选指标。

## Sell-in

Sell-in 表示经销商向 Hankel 订货形成的销售事实，是期间流量。

### 有效范围

Sell-in 数量和 NES 只纳入以下 Sales Team，且 `GMM L6 allocation != 'Y'`：

```text
GM TA
MRO
North
North Jiangsu
Shanghai
South Jiangsu
South1
South2
Zhejiang&Fujian
GM North
GM Hangzhou
GM Middle China
GM Nanjing
GM Suzhou
GM Shenzhen
GM Ningbo
GM Shanghai
GM Guangzhou
GM Anhui&Shandong
GMM EC
GM Beijing
GM SPL
IPR Team_IPR Product
IPR Team_Non IPR Product
IPR Team_Non IPR NES
IPR Team_IPR NES
```

### 指标

| 指标 | 公式 | 状态与解释 |
|---|---|---|
| Sell-in Quantity | `SUM(sell_in_quantity)`，应用 Team 白名单和 GMM L6 排除 | 客户确认，可用于 POC |
| NES | `SUM(nes)`，应用同一有效范围 | 业务解释为未税净销售额；过滤/冲销细节仍需生产确认 |
| Gross Margin | `SUM(gross_margin)` | 最新加工表应只有一个规范字段；业务列来源仍需上线前确认 |
| Gross Margin Rate | `Gross Margin / NES`，NES 为 0 返回 null | 派生口径，结果不可对百分比直接求和 |
| Product Contribution | `SUM(product_contribution)` | 字段可查询，但 `15*` 前缀业务含义待确认 |

Sell-in 中的负数、退货和冲销不得静默删除。若没有明确规则，应保留净额并在结果中报告。

当前语义 View 在保留每条原始行的同时，将有效范围内的值写入标准 measure，将范围外值保存在 `raw_*`、`excluded_*` 和 `scope_exclusion_reason` 审计字段。Agent 使用标准指标时不得再次把范围外记录加回。

## Sell-out

Sell-out 表示经销商向终端客户销售，是期间流量。业务分析必须使用 Sales Team/Territory 分摊后的字段：

| 指标 | 公式 | 禁止使用 |
|---|---|---|
| Sell-out Value | `SUM(territory_sell_out)` | raw `sell_out_value` |
| Sell-out Quantity | `SUM(territory_quantity)` | raw `sales_quantity` |

业务规则：

- 金额按净额汇总，负数必须保留。
- 数量与金额分别汇总，不要求符号或比例一一对应。
- 数量异常只影响数量指标；金额异常或金额不可解析只影响金额指标，两类质量规则不得交叉排除。
- 回补或冲销归入业务实际发生月。
- 同一经销商、终端客户、产品、月份出现两次数量相同的交易，两条都计入。
- `nan`、`na`、`-` 不自动删除；保留并作为空值/质量类别处理。
- 未分类产品归 `Other`，但不得把未知值归入某个已知产品类别。

系统的 sell-out 异常值阈值是 Demo 数据质量保护，不是客户业务公式。被排除值应保留在审计字段中，不能在知识库中描述为客户定义。

## Inventory

Inventory 是月末存量快照。默认查询最新可用快照，不跨月份累加。

| 指标 | 公式 | 说明 |
|---|---|---|
| Inventory Value | `SUM(territory_inventory_value)` | 客户默认库存指标 |
| Inventory Quantity | `SUM(territory_inventory_quantity)` | 用户明确询问数量时使用 |

如果库存数据按 Sales Team 展开，必须使用 Territory 分摊值，不能在展开行上重复累计 raw inventory。查询历史趋势时按月分别展示；查询当前库存时只取全局最新有效快照。

## 跨事实分析

正确流程：

1. 分别过滤 Sell-in、Sell-out 和 Inventory 的有效记录。
2. 分别按年月和 Sales Type 汇总。
3. 检查产品单位是否可比较。
4. 再对齐结果，不连接原始行。
5. 在结果中展示三个事实各自的数据截至月。

客户已确认 Sales Type 是跨事实共同维度，但尚未提供可执行的 Sell-in Team、Sell-out Team、Inventory Team 映射表。在该映射进入受治理表之前，Agent 只能分别按各事实的 `sales_team` 查询，不得把来源团队名称直接当作统一 Sales Type，也不得生成跨事实 Sales Type 对比。

以下指标目前只能作为候选，不得无条件回答为客户确认指标：

| 指标 | 候选公式 | 未确认内容 |
|---|---|---|
| Sell-through Rate | Sell-out / Sell-in | 分子分母金额或数量、滞后月份、零分母 |
| Sell-in/Sell-out Gap | Sell-in − Sell-out | 单位、时间和 Sales Type 映射 |
| Inventory Coverage Months | Ending Inventory / Avg Monthly Sell-out | 平均窗口、零销量处理 |
| Inventory Turnover | Sell-out / Average Inventory | 数量或金额、平均库存算法 |
| Active Distributor | 去重有活动经销商 | 使用 Sell-in 还是 Sell-out |
| Active/New/Inactive End Customer | 按客户活动规则去重 | 活跃窗口、NCA 和关系粒度仍需确认 |

## 非活跃客户

该流程当前为 draft。可保留的客户答复包括：

- 分析粒度倾向于 Distributor × End Customer × 截止月份。
- 截止时间由用户提供。
- 近 24 个月完全无购买记录的客户不进入候选范围。
- 近 24 个月只有一次购买时，候选购买频率为 24 个月。
- `Bian_endCust` 非有效记录不自动排除。
- 再次购买后可恢复活跃。

有效购买、退货月份、多经销商合并方式及精确活跃阈值没有完整客户确认，因此 Agent 不应自行判定非活跃名单。
