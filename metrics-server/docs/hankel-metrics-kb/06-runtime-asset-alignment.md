---
kb_id: hankel-runtime-asset-alignment
tenant_id: hankel
domain: governance
status: curated
tags: [runtime, views, metrics, alignment, lineage]
---

# Knowledge、View 与 Metric 对齐表

## 状态定义

| business_status | 含义 | Agent 行为 |
|---|---|---|
| `customer_confirmed` | 客户最终答复确认 | 可作为正式口径回答 |
| `customer_confirmed_semantic_foundation` | 支撑确认口径的标准化资产 | 可用于确认指标，但必须遵守资产限制 |
| `written_spec_reference` | 来自书面规则，未纳入当前客户确认 POC | 必须标注 reference |
| `pending_business_confirmation` | 仍有字段、时间或展示口径待确认 | 必须披露未决项 |
| `demo_quality` | Demo 数据质量或 QA 资产 | 不得描述为业务 KPI |
| `technical_diagnostic` | 技术诊断指标 | 不得替代业务指标 |
| `demo_fixed_parameters` | 当前 Demo 固定参数 | 必须披露实际日期，不得描述为动态参数 |

## River Runtime

| View / Metric | 对应知识 | 状态 | 对齐说明 |
|---|---|---|---|
| `hankel_view_distr_sell_in` | Sell-in 有效范围、负数保留 | foundation | 标准 measure 已应用 Team 白名单和 GMM L6 排除；raw/excluded 字段用于审计 |
| `hankel_sell_in_nes` | Sell-in NES | confirmed | 汇总有效范围内 `nes` |
| `hankel_sell_in_quantity` | Sell-in Quantity | confirmed | 汇总有效范围内数量；属于当前 River POC |
| `hankel_sell_in_gross_margin` | Gross Margin | pending | 字段来源和生产过滤仍待确认 |
| `hankel_sell_in_gross_margin_rate` | Gross Margin / NES | pending | 使用加权比率，不平均行百分比；依赖 Gross Margin 确认 |
| `hankel_sell_in_contribution` | Product Contribution | pending | `15*` 字段业务含义待确认 |
| `hankel_view_distr_sell_out` | Territory Sell-out | foundation | 金额和数量独立质量门槛；正常负数保留 |
| `hankel_sell_out_value` | Territory Sell-out Amount | confirmed | 数量异常不会排除有效金额 |
| `hankel_sell_out_quantity` | Territory Sell-out Quantity | confirmed | 金额异常不会排除有效数量；属于当前 River POC |
| `hankel_sell_out_excluded_value` | Demo quality | quality | 仅记录金额规则排除值 |
| `hankel_sell_out_quality_issue_count` | Demo quality | quality | 记录触发任一质量规则的行数 |
| `hankel_view_distr_inventory_monthly` | Territory Inventory | foundation | 月末快照，不跨月累计 |
| `hankel_view_distr_inventory_current` | Latest Inventory | foundation | 只取最新有效日期序列对应快照 |
| `hankel_inventory_quantity` | Inventory Quantity | confirmed | 当前 River POC 指标 |
| `hankel_inventory_value` | Inventory Value | reference | “Inventory”默认含义是金额，但未入选当前 POC 三指标 |

River 尚无正式 Sales Type 映射资产。三个事实的 `sales_team` 只能在各自事实内使用，不能跨事实直接比较。`combined_id` 已发布为经销商维度；Sell-out 同时发布终端客户、行业、平台、SPL 和质量维度。

## Caren Runtime

| View / Metric | 粒度 | 状态 | 对齐说明 |
|---|---|---|---|
| `hankel_view_run_for_gold_parameters` | 一组参数 | fixed | 当前固定 YTD Aug 2026 |
| `hankel_view_sales_name_mapping` | mapping row | foundation | 仅维护表精确映射 |
| `hankel_view_project_opportunity_line` | Opportunity × Product | foundation | 标识符标准化、日期标记和质量字段 |
| `hankel_view_new_project_opportunity` | Sales × Opportunity | reference | 先 `SUM(y1_value)` 再向上汇总 |
| `hankel_view_new_order_line` | Order × Item | foundation | 目标自然年；明确 Reject 才排除 |
| `hankel_view_won_validation_match_key` | Sales × Sold-to × Product | foundation | 50% 验证、Coverage、Action/Signed Gap |
| `hankel_view_validated_won_opportunity` | Sales × Opportunity | reference | 至少一个精确 Match Key Pass，Opportunity 只计一次 |
| `hankel_required_new_order_value` | Sales summary | confirmed | 精确 Check-period Won Y1 × 50% |
| `hankel_matched_new_order_value` | Sales summary | confirmed | 目标自然年精确 Match Key 订单金额 |
| `hankel_order_coverage_rate` | Sales summary | confirmed | Matched / Check-period Won Y1，不封顶 |
| `hankel_validation_won_y1` | Sales summary | pending | 公式确认；Won 候选起始日待确认 |
| `hankel_new_order_gap` | Sales summary | pending | 兼容名称，明确表示非负 Action Gap |
| `hankel_new_order_signed_gap` | Match Key | pending | Required - Matched，负值表示超额覆盖 |
| `hankel_validated_won_count` | Sales summary | reference | 去重 Opportunity，而不是 Match Key |
| `hankel_match_key_count` | Match Key | diagnostic | 仅用于匹配质量和分布分析 |
| `hankel_view_run_for_gold_qualification_status` | Pool × Sales | reference | 所有候选和门槛状态 |
| `hankel_view_run_for_gold_leaderboard` | Pool × qualified Sales | reference | 只有入围人员参与排名和奖位 |
| `hankel_view_run_for_gold_segment_qualification_status` | Segment × Sales | reference | 所有细分候选和门槛状态 |
| `hankel_view_run_for_gold_segment_leaderboard` | Segment × qualified Sales | reference | 只有入围人员参与细分排名 |
| `hankel_final_score` | Pool × Sales | reference | required group by Canonical Sales Name |
| `hankel_segment_final_score` | Segment × Sales | reference | required group by Segment + Canonical Sales Name |
| `hankel_view_run_for_gold_report_reconciliation` | QA check | quality | Golden report 仅用于验证，不作为事实源 |

## 仍未闭环

1. River Sales Type 映射表尚未提供，跨 Sell-in、Sell-out、Inventory 的统一区域分析不可用。
2. Won 候选起始日仍有 `Competition Start` 与目标年年初两种版本；当前 View 使用目标年年初。
3. Action Gap 与 Signed Gap 均已显式发布，但对外默认口径仍需客户选择。
4. Run for Gold 参数集中到单行 View，但尚未实现真正的 run-scoped 参数化。
5. Generic Project Dashboard 书面规范指标尚未完整发布；不属于当前客户确认 POC。
6. 知识文档必须实际导入 Agent Knowledge Base 后，Runtime Agent 才能使用这里的治理说明；仅发布 Metrics Meta 不等于完成知识库接入。

## 线上对齐证据

2026-09-04 在 tenant `hankel`、datasource `15` 完成 View、table meta 和
metric meta 更新后，验证结果如下：

| 检查 | 结果 |
|---|---|
| Sell-in 范围 | PASS；NES `1,607,161,170.4347694`，Quantity `17,673,540.187` |
| Sell-out 独立质量门槛 | PASS；数量异常不会排除有效金额，amount/quantity leak 均为 0 |
| Validated Won 粒度 | PASS；去重后的 Passed Opportunity 为 `782` |
| Overall 排行资格 | PASS；排行榜 `7` 行，未入围行 `0` |
| Segment 排行资格 | PASS；排行榜 `7` 行，未入围行 `0` |
| 业务标识符标准化 | PASS；Project/New Order 中残留 `.0` 标识符均为 `0` 行 |
| Inventory 快照 | PASS；当前视图仅有 `2026-07-01` 一个快照日期 |
| Published Meta | 18 个唯一 Table、24 个唯一 Metric；全部有 `business_status` |
| SQL-free Metric Meta | PASS；24 个 metric detail 中 SQL 字段数为 `0` |

这些数值用于验证当前加载数据和语义实现，不应被复制成固定业务答案。Agent
仍应通过 Metrics Runtime 查询实时结果，并返回实际时间范围、筛选条件和质量说明。
