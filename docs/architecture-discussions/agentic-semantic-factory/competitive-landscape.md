# 同类 AI 产品参照系

- 状态：draft v0.1（2026-09-05 搜索整理；Databricks/Snowflake 部分基于既有产品知识，搜索超时未复核最新动态）
- 目的：为 Agentic Semantic Factory 各层寻找市场参照——验证方向、汲取设计、明确差异

## 1. 分层对照表

| 工厂层 | 代表产品 | 值得借鉴 | 我们的差异点 |
|---|---|---|---|
| L5 运行时问答（AI 分析师） | ThoughtSpot Spotter（"verifiable, no-hallucination"、锚定治理数据）、Zenlytic Zoë（agent-first、对话即主界面）、Databricks Genie（只在 trusted assets 上回答）、Snowflake Cortex Analyst（semantic model YAML + verified queries） | ① 全行业收敛到"答案必须锚定治理资产，不裸 SQL"——与 P4/P8 同构；② Cortex Analyst 的 **verified queries**（人工确认过的问答样例）≈ 我们 golden report 对账样例的产品化先例；③ semantic model YAML 可作为我们 metric meta schema 的参照格式 | 回答证据里带 `business_status`（draft/confirmed）与 asOf——市面产品普遍没有"口径状态"概念 |
| 语义层/指标层 | dbt Semantic Layer（MetricFlow）、Cube（AI API + 行级安全）、Wren Engine | 机器可读指标契约、权限与维度约束的 API 化 | `based_on_decisions` 双向追溯：指标可回答"这个口径谁在什么时候确认的" |
| L3 转换 | dbt（事实标准）、SQLMesh（虚拟数据环境） | SQLMesh 的 virtual environment 是另一条蓝绿路线，佐证我们 in-place replace 的取舍需要记录边界 | — |
| L1 数据质量诊断 | Monte Carlo（转向 "Data + AI Observability"、公布 agent fleet 路线）、Anomalo（ML 深度异常检测）、Bigeye、Elementary（dbt-native 观察性） | ① 2026 明确趋势"AI agent 自己写并执行质量检查"——Profiler agent 方向被市场验证；② 起步阶段用 Elementary 这类 dbt-native 工具避免自建观测 | 诊断产物不是告警而是**决策积压**——质量发现闭环到口径确认与知识层，观察性产品普遍止步于告警 |
| CDP/行动面（Process Agent 平台） | Hightouch AI Decisioning（$80M C 轮，"Agentic Marketing Platform"）、Hightouch Agents、Salesforce Agentforce（Data Cloud 内自主建 segments、校验 consent） | ① 验证了 Data Agent 平台 + Process Agent 平台的双平台分工（agentic-cdp spec 的架构）；② composable（仓原生）vs suite 之争——我们天然站仓原生一侧 | consent/budget/approval gate 的显式审计设计 |
| 开源引擎备选 | Vanna AI（MIT，RAG/agentic retrieval text-to-SQL 库）、Wren AI（GenBI 平台，RAG + 语义建模引擎） | Vanna 可作为特定嵌入场景的 text-to-SQL 备选；Wren 的语义引擎思路可研究 | 我们主路线是 metric query（不裸生成 SQL），只在 T2/T3 边缘场景需要 |
| 元数据/知识 | Atlan、Alation（均转向 Data + AI 定位） | glossary/lineage 的产品化形态 | KB ↔ decisions 双向绑定、business_status 状态机 |

## 2. 三个结论

1. **方向被验证**：四个子方向都有成熟玩家与资本验证（Spotter/Zenlytic/Genie/Cortex 的 agentic analytics、Hightouch 的 agentic marketing、Monte Carlo 的 agent fleet）。我们不是在无人区，而是在一条被确认的路上做纵深。
2. **行业共同收敛点应直接遵循**：答案锚定治理资产、语义契约机器可读、dbt 作转换标准、观测 agent 化。这些不用发明，照做即可。
3. **真正的差异在"口径知识生产循环"**：市面产品要么是查询侧 AI（Spotter/Genie/Cortex），要么是观测侧 AI（Monte Carlo/Anomalo），**没有产品做"诊断 → 决策捕获 → 知识编译 → SQL 回流"的确认闭环**。business_status、decisions/*.json、确认 app 这套东西是可对外讲的差异化故事——也是传统数仓"业务+IT 反复"痛点的直接解药。

## 3. 可拿来主义清单

| 借什么 | 从哪 | 用在哪 |
|---|---|---|
| semantic model YAML 的字段设计 | Cortex Analyst / dbt Semantic Layer | metric meta schema v0.2 评审时对照 |
| verified queries（确认样例库） | Cortex Analyst | golden report 对账样例 + 回答回归测试 |
| dbt-native 质量报告 | Elementary | M3 起步的观测，不自建 |
| 竞品对照叙事 | Spotter 的"verifiable"主张 | 方案书：可验证性 = 治理资产 + 口径状态 + 对账基线 |
| 双平台分工话术 | Hightouch/Agentforce 的 agentic marketing | agentic-cdp 场景的对外叙事 |

## 4. 风险提示

平台巨头（Databricks/Snowflake/ThoughtSpot）在通用问答侧会持续碾压——我们的定位不应是与它们拼通用 NL 查询，而是：**私有部署贴合客户数据边界、口径知识生产循环、业务自助边界治理**这三个它们都不碰的纵深。

## 5. "平台都会做这个"的推演与应对

### 5.1 诚实的商品化预测

| 能力 | 平台会做的概率 | 时间感 | 我们的应对 |
|---|---|---|---|
| T1 运行时问答 | 必然 | 已在做 | 不投入差异化；运行面做薄、可插拔 |
| T2 指标起草辅助（AI 建议/生成 metric） | 高 | 1–2 年 | 差异不在生成，在"生成物挂 decisions 与状态机" |
| T3 dbt 模型生成 | 高（dbt Copilot 类已在做） | 已在做 | 同上；test 成对 + 对账基线是我们独有 |
| 确认 app / 决策捕获 | 中 | 更晚 | 见 5.2，结构性难点 |
| 口径知识生产闭环 | 低–中 | 未知 | 核心差异位，见 5.2 |

### 5.2 平台结构性做不了的三件事

1. **跨组织的确认闭环**：确认人（客户业务方）是平台边界**之外**的角色。BI 平台服务的是拥有数据的企业内部团队；而"把口径问题发给客户专家 → 捕获决策 → 状态流转"是一个跨组织工件流。平台没有这个产品面，也不该有——它是服务型业务的形态。
2. **产品化服务与冷启动**：landing、本体抽取、初始建模、对账基线是咨询性质的活。平台卖消费（compute/consumption），不会做交付；工厂是"把方法论编码成软件"的产品化服务——商业模式不同，不是功能差距。
3. **多客户隔离与行业模板**：一个工厂服务 N 个客户（P10 隐私、租户 KB 隔离、行业模板复利）是乙方形态；平台是甲方自用形态。

### 5.3 定位策略：站在语义层的上游

把商品化从威胁变成渠道：**工厂产出 AI-ready 语义资产，喂给任何下游消费平台——包括 Genie/Cortex/Spotter**。平台问答越强，对高质量语义资产的胃口越大；我们不做它们对面，做它们的上游。实现上要求 metric meta 可导出为平台格式（Cortex semantic model YAML、dbt Semantic Layer 等），这是一条明确的工程要求。

### 5.4 真正要积累的护城河

平台能抄功能，抄不走**复利资产**：decisions 语料（每个客户确认过的口径及其理由）、行业模板、对账基线与回归测试集、项目状态机的实操经验。这些随每个项目增长且客户绑定——竞争者即使复制软件，也要从零积累资产。窗口在"垂直私有部署 + 交付服务"（B1 生态、快消经销商等），不在通用产品。

### 5.5 近期更该盯的竞对

不是 BI 平台，而是：① 数据目录公司的 AI 化（Secoda/CastorDoc/Select Star——知识+血缘方向最近，但仍无跨组织决策捕获）；② 大 SI 的服务产品化（同样看到"数仓交付太慢"的痛点）。

## 来源

- [ThoughtSpot Spotter](https://www.thoughtspot.com/product/agents/spotter)、[Spotter 2026 Review](https://valiotti.com/thoughtspot-spotter-review-2026/)
- [ThoughtSpot vs Zenlytic（Basedash）](https://www.basedash.com/vs/thoughtspot-vs-zenlytic)、[Zenlytic 官方对比](https://zenlytic.com/zenlytic-vs-thoughtspot)
- [Vanna AI（GitHub）](https://github.com/vanna-ai/vanna)、[Vanna AI 官网](https://vanna.ai/)
- [Wren AI vs Vanna 对比](https://getwren.ai/post/wren-ai-vs-vanna-the-enterprise-guide-to-choosing-a-text-to-sql-solution)
- [Hightouch vs Salesforce CDP](https://hightouch.com/compare-cdps/hightouch-vs-salesforce-cdp)、[Hightouch $80M Series C](https://cmscritic.com/u-cant-touch-this-cdp-hightouch-nails-80m-series-c-to-power-its-ai-decisioning)
- [Salesforce Agentforce](https://www.salesforce.com/agentforce/what-is-agentic-ai/)
- [Data Observability 工具盘点（Atlan）](https://atlan.com/know/data-observability-tools/)、[Monte Carlo vs Anomalo](https://www.anomalo.com/blog/monte-carlo-vs-anomalo/)、[Monte Carlo 替代品（含 Elementary）](https://dataworkers.io/resources/monte-carlo-alternatives/)
