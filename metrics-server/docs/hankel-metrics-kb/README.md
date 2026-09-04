---
kb_id: hankel-metrics-analysis
tenant_id: hankel
title: Hankel ACM 指标分析知识库
language: zh-CN
source_version: gtm-ontology-1.0.0
source_updated: 2026-09-03
reviewed_at: 2026-09-04
kb_version: 1.1.0
---

# Hankel ACM 指标分析知识库

本目录从客户提供的 `gtm-ontology` 资料包提炼业务口径，供 Data Agent、Metrics Agent 和人工分析人员解释指标、选择维度、检查数据质量及说明不可用原因。

源 Ontology 使用 `Henkel/ACM` 名称，系统 tenant 和数据库对象使用 `hankel` / `hankel_`。两者在本知识库中指同一客户语境，不应被识别成两个客户。

## 文档路由

| 问题类型 | 应加载文档 |
|---|---|
| “销售额”“客户”“区域”“最新”等词义消歧 | `01-business-context-and-terms.md` |
| Sell-in、Sell-out、库存、经销商、终端客户分析 | `02-river-distributor-metrics.md` |
| Project、New Order、50% 验证、Coverage、Gap | `03-caren-project-won-validation.md` |
| Generic Project Dashboard 的 Won/Lost/Active/Y1/Y2 指标 | `04-generic-project-dashboard.md` |
| 空值、重复、异常、隐私、未确认口径和 runtime 差异 | `05-analysis-governance-and-open-questions.md` |
| View、Metric、业务状态和知识条目的逐项映射 | `06-runtime-asset-alignment.md` |

`manifest.json` 提供机器可读的文档清单、状态枚举和 Runtime API 入口。导入知识库时应保持同一 tenant collection，并保留文档文件名作为 source metadata。

## 证据优先级

口径冲突时按以下顺序处理：

1. 客户最终答复与 `evidence/decisions/*.json`。
2. `meta.status=confirmed` 的对象、指标和流程。
3. Generic Project Dashboard v2.0 书面规范。
4. `draft`、`inferred`、`candidate` 和 `reference_metric`，只能作为候选解释，不能当作已确认公式。

不得用运行中的 SQL、报表结果或字段名反向覆盖客户口径。若实现和客户口径不一致，应报告差异。

主要来源文件：

- `evidence/decisions/river-decisions.json`
- `evidence/decisions/caren-decisions.json`
- `evidence/responses/acm-poc-response-20260903-1825.json`
- `evidence/responses/acm-poc-response-20260903-1838.json`
- `catalog/metric-contracts.json`
- `catalog/tables-and-relationships.json`
- `context/glossary.yaml`
- `governance/agent-policy.yaml`

## Agent 使用规则

- 回答指标问题时必须说明指标、时间范围、筛选条件、分组粒度和数据截至日期。
- 只使用已发布的 Metrics meta 维度；知识库负责解释含义，不授予数据库访问权限。
- 指标所需字段缺失时返回 `unavailable`，不得用相似字段估算。
- `draft` 指标必须明确标注“待业务确认”或“书面规范指标”，不得描述为客户已确认。
- River 三张事实表不得做明细行 Join；先独立汇总，再按共同维度比较。
- Caren 匹配必须使用维护的销售姓名映射和精确 Match Key，不得模糊匹配。
- 结果计算使用精确值；取整只发生在展示层。
- 不在持久化报告、知识库或日志中保存销售人员姓名及终端客户明细。

## 推荐回答结构

```text
结论
指标口径
时间与维度
数据质量/例外
数据截至时间与来源
```

若用户使用歧义词，先完成口径消歧，再查询；不要同时返回多个口径后让用户自行猜测。
