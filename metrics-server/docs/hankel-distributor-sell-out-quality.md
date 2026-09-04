# Hankel Sell-out 数据质量处理

## 结论

`hankel_distr_sell_out` 的数据库导入行数和金额与原始工作簿一致。排名异常来自两个业务数据问题：原始文件包含极端数量/金额记录；原始金额在销售团队分摊行中重复保存，不能直接跨行汇总。

## 正确粒度

原始表的一行表示一个销售团队分摊行。同一客户、终端客户、产品和月份的基础交易可以拆成多行：

- `sales_quantity`、`sell_out_value` 是拆分前的基础交易值，会在多行重复。
- `territory_sales_quantity`、`territory_sell_out` 是每个销售团队的分摊值，可以在当前粒度累加。

因此 runtime 指标必须基于分摊字段，而不能直接汇总原始字段。

## 规范化视图

`public.hankel_view_distr_sell_out` 保留三组字段：

- 原始字段：`raw_sales_quantity`、`raw_sell_out_value`
- 分摊字段：`allocated_sales_quantity`、`allocated_sell_out_value`
- 默认指标字段：`sell_out_quantity`、`sell_out_value`

默认指标字段按指标分别应用质量规则：数量异常只影响 `sell_out_quantity`，金额异常或金额不可解析只影响 `sell_out_value`。被排除值仍保存在 `excluded_sell_out_quantity` 和 `excluded_sell_out_value` 中。

## 当前质量规则

当前 demo 使用两条显式 guardrail：

- `abs(raw_sales_quantity) >= 10,000,000`
- `abs(raw_sell_out_value) >= 1,000,000,000`

科学计数法（例如 `4.82E-2`）属于合法 numeric 输入。同时，`territory_sell_out` 为空或不能转换为 numeric 时会排除。规则命中结果记录在：

- `is_quantity_outlier`
- `is_amount_outlier`
- `has_invalid_allocation_value`
- `is_quantity_quality_excluded`
- `is_value_quality_excluded`
- `is_quality_excluded`
- `quality_issues`

这些阈值用于当前 demo 数据，生产使用前需要由业务方确认或迁移到可配置的数据质量策略。

当前数据验证命中 7 个分摊行，对应 3 笔基础交易：

| 客户 | 月份 | 原始数量 | 原始金额 | 分摊行数 |
| --- | --- | ---: | ---: | ---: |
| 沈阳赛福化工材料有限公司 | 2025-05 | 2,147,483,647 | 343,929,169,743.46 | 3 |
| 沈阳赛福化工材料有限公司 | 2026-02 | 20,000,203 | 3,203,122,511.36 | 3 |
| 江苏梅珀尔润滑油有限公司 | 2025-05 | 20,010,065 | 710,838,216.06 | 1 |

这些记录仍保留在原始表和视图的 raw/allocated 字段中。金额和数量分别决定是否从对应业务指标排除。例如江苏梅珀尔记录只触发数量阈值，其 Territory 金额仍计入 Sell-out Value。

## Runtime Metrics

- `hankel_sell_out_value`：有效分摊出货金额
- `hankel_sell_out_quantity`：有效分摊出货数量
- `hankel_sell_out_excluded_value`：质量规则排除金额
- `hankel_sell_out_quality_issue_count`：异常分摊行数

前两个是业务分析默认指标，后两个用于质量监控和人工复核。

## 部署顺序

1. 在 datasource 15 执行 `scripts/hankel/distributor-sell-out-view.sql`。
2. 执行 `scripts/hankel/publish-distributor-sell-out-meta.sh` 发布 table/metric meta。
3. 通过 `GET /api/v1/datasources/15/meta` 检查发布结果。
4. 通过 `POST /api/v1/metrics/query` 验证排名和 QA 指标。
