---
kb_id: hankel-generic-project-dashboard
tenant_id: hankel
domain: generic-project-dashboard
status: written-spec
tags: [project, won, lost, active, y1, y2]
---

# Generic Project Dashboard v2.0 指标口径

本页来自书面业务规范。它与 Caren 最终答复中的 Won→New Order POC 范围不同：这些指标可作为 Project Dashboard 规范使用，但不能据此声称客户已把排名/奖项纳入当前 POC。

## 通用参数

- `Selected Team`：查询指定销售团队。
- `CY`：Report Cut-off 所在自然年。
- `YTD`：CY 年初至 Report Cut-off。
- 项目数量统一按 `DISTINCT Opportunity ID`。
- Won/Lost 使用 `Close Date`。
- New Added 使用 `Creation Date`，不能误用 Close Date。
- Active Pipeline 使用 `Status='In Process'`，不施加 YTD Month 截断。
- 除法分母为 0 时返回 null。

## 当前字段可支持的指标

| 指标 | 公式/筛选 |
|---|---|
| 本年活跃项目数 | `COUNT_DISTINCT(Opportunity ID)` where `Status='In Process'` |
| 活跃项目 Y1 | `SUM(Y1)` where `Status='In Process'` |
| 本年新增项目数 | `COUNT_DISTINCT(Opportunity ID)` where Creation Date in CY YTD |
| 新增项目 Y1 | `SUM(Y1)` where Creation Date in CY YTD |
| 本年 Won 项目数 | `COUNT_DISTINCT(Opportunity ID)` where Status=Won and Close Date in CY YTD |
| 本年 Won Y1 | `SUM(Y1)` where Status=Won and Close Date in CY YTD |
| 本年 Lost 项目数 | `COUNT_DISTINCT(Opportunity ID)` where Status=Lost and Close Date in CY YTD |
| 本年 Lost Y1 | `SUM(Y1)` where Status=Lost and Close Date in CY YTD |
| 项目数量赢单率 | `Won Count / (Won Count + Lost Count)` |
| 项目价值赢单率 | `Won Y1 / (Won Y1 + Lost Y1)` |
| 平均 Won Y1 | `Won Y1 / Won Count` |

项目价值赢单率是 Dashboard headline Win Rate；不要用项目数量赢单率替代。

## 依赖字段存在时才可计算

| 指标 | 公式 | 必需字段 |
|---|---|---|
| 活跃项目折算 Y1 | `SUM(Discounted Y1)` where In Process | Discounted Y1 |
| 活跃项目 Y2 | `SUM(Y2)` where In Process | Y2 |
| 新增项目 Y2 | `SUM(Y2)` where Creation Date in CY YTD | Y2 |
| Won Y2 | `SUM(Y2)` where Won and Close Date in CY YTD | Y2 |
| Lost Y2 | `SUM(Y2)` where Lost and Close Date in CY YTD | Y2 |
| 平均 Won Y2 | `Won Y2 / Won Count` | Y2 |
| 上年结转项目价值 | `LY full-year Won Y2 - LY full-year Won Y1`，无 YTD 过滤 | Y2 |
| 项目总增长影响 | `Carry-over + CY YTD Won Y1 + CY Active Discounted Y1` | Y2、Discounted Y1 |
| 项目平均周期（月） | Opportunity 去重后 `AVG(Days Duration) / 30` | Days Duration |

如果缺少 Y2、Discounted Y1、Discounted Y2、Project Source、Reason for Status、Duration、Category 或 Opportunity Description，只将依赖该字段的指标标记为 `unavailable`。不得用 Y1、Creation Date 或其他近似字段补算。

## 项目行重复与聚合

Project 表是 Opportunity × Product 行，一个 Opportunity 可以出现多行：

- 项目数必须先按 Opportunity ID 去重。
- Opportunity 级属性若在多行重复，必须先归并到 Opportunity 粒度再参与项目级平均或计数。
- 金额指标是否为行金额或 Opportunity 总额必须依据字段定义；不能同时在每行重复总额后再 SUM。
- Project Average Duration 必须先按 Opportunity ID 去重，再求天数平均，最后除以 30。
