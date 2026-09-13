# 开放问题：建议方案

- 状态：draft v0.1（README §9 的展开，逐题给出建议、理由与第一步）
- 关联：[README（方法论与目标架构）](./README.md)、[m1-bootstrap-design.md](./m1-bootstrap-design.md)

## Q1 dbt-core 还是自研加工层？

**建议：直接用 dbt-core（dbt-postgres）起步，不自研。**

理由：

1. 差异化不在 SQL runner。这套体系的护城河是"知识获取循环 + 状态机 + 确认门"，不是又一个建模框架。自研 schema.yml 等价物、test 框架、血缘解析、docs，是数月工作量换零差异。
2. 对 LLM agent 特别有利：dbt 的模式（model/test/schema.yml/Jinja）在训练语料里极多，Modeler agent 的产出质量天然高于面对自研 DSL；而框架的"model 必须配 test"约束恰好落实 P1/P8 原则。
3. 白送的能力：DAG 血缘（manifest.json/catalog.json，正好供 Publisher 和追溯消费）、原子替换、tests、docs、`run_results.json`（agent 读失败结果迭代的标准接口）。
4. 迁移成本低：`run-for-gold-views.sql` 的 `CREATE OR REPLACE VIEW` 直接映射为 `materialized='view'` 的 models；landing 原表注册为 dbt `sources`，dbt 不碰落地区。

边界与备选：metrics-server 仍拥有 runtime meta 发布权（dbt 不替代 publish 脚本，但 publish 改为消费 dbt artifacts）；若未来要多租户动态建 schema，用 Python `dbtRunner` 编程驱动即可，不必换框架。SQLMesh 的虚拟环境值得留意，但生态与 LLM 熟悉度不如 dbt，列为主要备选而非首选。

**第一步**：M3 迁移时以 `dbt init` 起新项目，models 命名沿用 `hankel_view_*` 语义（staging 层用 `stg_` 前缀），CI 里 `dbt build` 全绿即 gate。

## Q2 确认 app 的形态？

**建议：双轨同源，M2 只做对外静态页。**

- **对外（客户/业务确认人）**：单文件静态 HTML，决策积压内嵌，逐条选择 + 批注，底部"导出 decisions.json"。零服务端、零安装、可离线、可过企业防火墙——客户侧使用门槛是硬约束，这一条压倒一切。JSON 从既有渠道回来（邮件/文档服务/人工放置），由 `project import-decisions` 动词校验导入（schema 校验 + decision_id 查重 + 状态流转）。
- **对内（项目内 gate）**：同一个生成器，页面直接 POST 到 agent 网关（`agent/` Fastify 已有）：`POST /api/v1/projects/{id}/decisions`。省去文件搬运，幂等键就是 decision_id。
- 两轨唯一的差别是传输；schema 完全一致，状态永远以 decisions/*.json 为准。

明确不做：M2 不做带登录/会话/权限的 Web 应用——那是几周 Plumbing，价值密度极低。确认 app 的价值在**积压质量**（证据渲染是否足以让人拍板），不在工程外壳。

**第一步**：M1 的 CLI 先落 `import-decisions` 动词（纯文件模式），M2 的生成器以 hankel 11 条待确认项为素材出第一版。

## Q3 decisions 与 KB/模型的追溯粒度？

**建议：分三级，L1/L2 从第一天强制，L3 列级自动血缘只做 CI 警告、渐进收紧。**

| 级别 | 内容 | 强制性 | 成本 |
|---|---|---|---|
| L1 | decision.scope.applies_to 指向资产集合（metric/view 名） | **强制**，写进 schema 必填 | 零（纯元数据） |
| L2 | 反向索引：资产 → 决策。metric meta 增 `based_on_decisions` 字段，Publisher 从 L1 自动注入 | **强制**，自动派生 | 极低 |
| L3 | 列级自动血缘（decision → model 列 → metric） | **不强制**；CI 警告渐进 | 高，且人工维护必腐 |

理由：追溯的目的是**可查询、可影响分析**（决策状态变了，波及哪些资产要重审），L1/L2 已足够支撑；L3 的 decision→model 映射本质是人工断言，强制自动化会造成没人维护的假血缘。dbt 的 manifest/catalog 免费提供 model 间列级血缘，两者拼接是增量工作，留到体系跑顺之后。

配套规则：CI 检查两类断链——model/test 变更但 scope 内无关联决策 → 警告；confirmed 决策的 scope 内资产变更 → 标记重审。

**第一步**：decisions schema 的 `scope.applies_to` 定为必填（M1 设计已含），Publisher 脚本加 `based_on_decisions` 注入（M3 顺手做）。

## Q4 run-scoped 参数怎么实现？

**建议：区分两类参数。业务参数进配置表（参数是数据），工程参数用 dbt vars（参数是代码）。materialization job 暂不做。**

- **业务参数**（赛期起止、cut-off、目标年、验证阈值）定义的是"哪一次比较"，生命周期跟着 run 走而不是跟着代码走。dbt vars 改一次参数要重新编译部署，且无法同时共存 2026/2027 两个赛季、无法复现历史 run——寿命不匹配。形态：

  ```sql
  create table hankel_run_parameters (
    run_id text primary key,
    competition_start date, competition_end date,
    report_cutoff date, target_year int,
    validation_threshold numeric,
    is_active boolean, created_at timestamptz, created_by text
  );
  ```

  迁移成本极小：把 `hankel_view_run_for_gold_parameters` 从常量 SELECT 改为读 `is_active` 行，同名 view 作为兼容垫片，下游 models 一行不改。Runtime 侧把 run_id（或沿用"active run"语义）发布为维度。
- **工程参数**（目标 schema、QA 容差、环境开关）：`dbt_project.yml` vars / `--vars`，随部署变化。
- **materialization job**：等 views 性能真成为问题、需要物化 marts 时再上，M3 不做。

治理配套：**新建 run 本身是一次决策**（它定义了比较范围），run 的创建链接一条 decision 记录——KB06 里 `demo_fixed_parameters` 状态到那时才能退役。

**第一步**：并入 M3 迁移（顺路做，避免二次返工）。

## Q5 多租户 KB 隔离与 manifest 规范？

**建议：collection 与 tenant 强绑定 + manifest 机器校验 + 禁止跨租户引用；绝不复制 metrics v1 的弱租户回退。**

1. **绑定规则**：一个 tenant 一个 KB collection，collection id = tenant_id；每篇文档 frontmatter 必带 `tenant_id` + `kb_id`（hankel KB 已是此形态，升为规范）。导入时校验 frontmatter 与目标 collection 一致，不一致拒绝。
2. **manifest 规范**（把 hankel 的 `manifest.json` 提为模板）：`kb_id`、`tenant_id`、`kb_version`、`source_version`、`reviewed_at`、`docs[]`（file/doc_type/status/tags）、`runtime`（meta/query 入口）、`constraints.cross_tenant_citation: forbidden`。写一个 linter：frontmatter↔manifest 一致、状态枚举合法、README 路由表覆盖所有文档。
3. **隔离红线**：Analysis agent 严格按 `X-Tenant-Id` 解析 collection，**无跨租户回退**。特别提示：run-for-gold 文档自己承认 metrics v1 的 tenantId 是弱上下文（无租户 grants 时回退同 datasource 的 active grants）——KB 侧必须比它严格，并把这个差异显式写进 agent-policy。
4. **目录收敛**：项目 KB 从 `metrics-server/docs/hankel-metrics-kb` 迁到 `knowledge/projects/<project_id>/kb/`（架构文档 §6 约定），metrics-server/docs 只留模板。模板命名空间任何租户可读不可写。

**第一步**：linter + manifest 模板（半天工作量），迁移目录放在 M1 落 `knowledge/` 骨架时一起做。

## Q6 模板回流实例的审核机制？

**建议：两条硬性晋升标准 + KB owner 一票审批 + git PR 流。初期宁滥勿缺。**

- **晋升标准**（同时满足）：① 在 ≥2 个独立项目 confirmed，或 1 个项目 + 书面规范；② 项目无关化（无客户专名/专有字段）；③ 带 `origin: [decision_id...]` 溯源；④ 通过隐私清洗（P10：无人名/终端客户明细）。
- **把关人**：设一个 **KB owner** 角色（demo 阶段就是一个人 + 一张 checklist），不必成立委员会。Curator agent 负责**起草**模板 diff 和出处链，人只做批准/打回。
- **机制**：`knowledge/templates/` 走 git PR（和代码同流），模板带版本号，只增版本不改历史；某项目若与模板冲突，冲突本身成为该项目的一条 decision，并给模板打"待修订"标记——模板永远不会被静默改写。
- **初期策略**：晋升门槛执行从宽，宁可模板目录略乱，也不要知识困死在单项目里。复利的瓶颈是"没人沉淀"，不是"沉淀得不够精"。

**第一步**：在 README §6 目录约定下加 `templates/REVIEW.md`（checklist + 两条标准），第一个候选就是 hankel 的质量规则表。

## Q7 Generic Project Dashboard 20 指标作为压力测试？

**建议：定为 M5——M1–M4 完成后的第一次全流水线正式验收，并给"工厂"自身定义量化指标。**

为什么它是正确的首个负载：20 个指标有书面规范（status=written_spec_reference，无需客户口径循环，不卡 gate）；当前 runtime 指标集未覆盖它（真实积压）；横跨 Won/Lost/Active/Y1/Y2 多域，能逼出 staging/intermediate/marts 分层。

用工厂自身的指标来量化验收（这也是"第二个项目应是第一个的零头"从口号变成数字的方式）：

| 工厂指标 | 目标 |
|---|---|
| 非 gate 的人工介入次数 | 0 |
| gate 数量与总耗时 | 记录并逐年下降 |
| 缺陷捕获点 | test/对账捕获为主，人检与线上为辅 |
| 工件自动生成率 | 探查/建模/发布/对账工件 ≥90% 由 agent 产出 |
| 总耗时 | 对照 Hankel 手工基线（约 2 天）显著压缩 |

一个可预期的考验：Dashboard 的 Won YTD 口径与 Run for Gold 的 Competition 窗口不得混用（KB06 已标注）——这恰好检验 Q4 的 run-scoped 参数能否让两套口径在同一模型上共存。**这个冲突是特性不是 bug，压力测试就要测它。**

**第一步**：M1–M4 收尾后，把 Dashboard 20 指标的书面规范整理成决策积压 + 探查提案（Curator/Explorer 的输入工件），按流水线走完并记录上表数据。
