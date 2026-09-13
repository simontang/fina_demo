# 试点切片与架构验收计划

- 日期：2026-09-05；状态：实施建议，测试尚未执行。
- 上游：[总体架构 v0.2](./architecture-v0.2.md)、[协议](./contracts-and-release-protocol.md)。
- 目标：用最少真实业务范围验证知识确认、发布一致性、自动建模与可恢复性，分别记录证据。

## 1. 两个试点分别验证不同命题

### A. New Order Gap：真实待确认口径

来源：[KB03](../../../metrics-server/docs/hankel-metrics-kb/03-caren-project-won-validation.md) 的 Action Gap / Signed Gap 待确认项。当前两个指标均已存在，发布脚本的默认状态仍为 pending。

试点问题：“对外默认展示行动缺口，还是允许负值表达超额覆盖？”两种公式及其各自适用问题都展示给确认人，由客户选择；本设计没有代为选择。

产出一条 DecisionRevision、一份 Semantic Spec、版本化 meta/KB、一个发布包、独立验收和证据回答。保留两个显式指标的原定义，通过场景默认指向/说明处理客户选择，不静默改变原有 metric ID 的含义。

范围包含 sales mapping、project/order lines、参数、match-key 视图的必要依赖闭包；不扩展 leaderboard、机会数等相邻指标。首次 prepare 时将必要实现封存到试点版本关系中；不能直接引用会继续原地更新的生产 view 并称为隔离版本。

该切片能证明决策到消费的一致性；因为公式已存在，**不能单凭它证明 agent 自动生成并修改 SQL 的能力**。

### B. Sell-in 白名单：单模型历史回放

来源：[Sell-in SQL](../../../metrics-server/scripts/hankel/distributor-sell-in-inventory-views.sql)、[发布脚本](../../../metrics-server/scripts/hankel/publish-distributor-sell-in-inventory-meta.sh)、[KB05](../../../metrics-server/docs/hankel-metrics-kb/05-analysis-governance-and-open-questions.md) 中 Quantity/NES 共用的 Team 白名单与 GMM 排除规则。首个指标选 `hankel_sell_in_nes`，计算 `SUM(nes)`，单位为人民币；Quantity 只作为后续扩展。

在隔离测试项目回放“未应用规则 -> 按已有业务证据应用规则”。产出 dbt model + tests + 独立对账；不重新向客户索取已经存在且范围适用的确认，也不把历史业务决定包装成新发生的批准。

文档保留的 NES 全量与规则后总额分别为 `1,656,594,630.73` 元和 `1,607,161,170.43` 元；它们不是 Quantity。它们是仓库记载的历史观测，不是本次已复算结果。只有能够取得匹配输入快照且核对单位、期间、精度后，才将其用作精确 regression baseline；否则只做小样例回放，并如实记下历史大样本复现的缺口。

## 2. 里程碑与原计划对应

| 阶段 | 具体工作 | 交付证据 | 对应原 M |
|---|---|---|---|
| P0 基线 | 登记隔离测试项目、最小权限、输入快照、当前 meta/KB、已有依赖 | baseline manifest、权限验证报告、当前行为样例 | 收缩后的 M1 + 发布前安全项 |
| P1 知识闭环 | Gap 积压 -> 静态确认包 -> 答复核验 -> decision/spec -> KB 编译 | preview/response/gate/spec/KB 摘要闭环；模拟与真实确认明确分开 | M2 优先 |
| P2 发布闭环 | 封存候选关系、Java strict scope、release prepare/validate/activate/rollback、query context | 两版并发查询不混版、故障恢复、证据回答 | M3 中发布治理前移 |
| P3 模型闭环 | Sell-in 单模型 dbt 迁移及历史规则回放；输入与 QA 独立 | build artifacts、变更 diff、mutation tests、复算结果 | M3 的最小切片 |
| P4 扩展覆盖 | 通用 import、schema drift、剩余模型、业务 run 参数化、Dashboard 20 指标 | 每个指标的状态/依赖/验收；两个 run 共存 | M1/M3/M4 剩余部分与 M5 |
| P5 跨项目复用 | 用独立第二客户/场景重复流程，开展业务操作者实验 | 实测人工投入、确认等待、缺陷及模板复用 | 原第二项目与 M5 人效实验 |

P1 可以先离线推进，不依赖新的常驻服务；P2 未完成前不把候选对外作为正式受治理指标。P0 只做该切片必要的项目登记，不先实现通用六动词管理平台。

P2 需要真正的 Java 版本/授权改造，应单独估算和验收，不能写进“脚本顺手包装”工作量。P3 最初可以继续使用 CLI runner；需要多项目异步触发时再引入 FastAPI/Celery 常驻形态。

## 3. Gap Spec 与测试数据

### 3.1 先锁定业务语义

- Project 基础粒度：Opportunity x Product；Order 基础粒度：Order x Item。
- 两侧分别汇总到 Canonical Sales x Sold-to x Product 后 join，避免明细 join 放大。
- Action Gap 在 match key 粒度归零，然后汇总；Signed Gap 保留正负后汇总。
- 现行代码的年度为 2026，competition start 为 2026-07-01，cutoff 为 2026-08-31；订单按目标日历年取数，不擅自改为截止日内订单。
- 阈值 0.5 在参数 view 和 Required 表达式中都出现。P3/P4 参数化必须消除隐性硬编码，新增参数时验证所有依赖。
- 显示舍入未明确的部分保持 limitation；计算层使用 decimal，不能为了贴合整数展示把精确值改写。

业务日期与舍入依据见 [KB03](../../../metrics-server/docs/hankel-metrics-kb/03-caren-project-won-validation.md)；实现对应 [run-for-gold-views.sql](../../../metrics-server/scripts/hankel/run-for-gold-views.sql) L6/L311/L375/L410。

### 3.2 合成算术样例

下表是用于验证公式的明确合成输入，不含客户人名；将输入视为精确十进制值，不声称复现了原始文件全部精度。

| case | Won Y1 | Matched Order | threshold | Required | Signed Gap | Action Gap |
|---|---:|---:|---:|---:|---:|---:|
| above | 12487 | 9524 | 0.5 | 6243.5 | -3280.5 | 0 |
| below | 100 | 40 | 0.5 | 50 | 10 | 10 |
| equal | 100 | 50 | 0.5 | 50 | 0 | 0 |
| mixed-a | 100 | 40 | 0.5 | 50 | 10 | 10 |
| mixed-b | 100 | 80 | 0.5 | 50 | -30 | 0 |

`mixed-a + mixed-b` 汇总的 Signed Gap 为 `-20`，Action Gap 为 `10`。`above` 参考文档展示数字建立算术样例，原文整数 Required 的舍入解释仍待确认。

另设 rejected-order、重复映射、同 product 跨不同 sold-to、日期边界、无匹配订单、缺失金额、零分母样例。对于未确认的 null/zero-denominator 业务行为，先验证“明确返回待确认/限制”，不得擅自把期望值设成零。

### 3.3 独立验证，避免同一错误同时生成 SQL 与测试

固定业务样例/期望值由业务确认或人工复核建立；Reconciler 读取该契约，不能从被测 SQL 自动推导期望值。模型生成与最终验收输入隔离，review 者看到公式、样例和证据。

增加故意错误的 mutation：先 sum 再 clamp、提前把负 gap 归零、join 前不去除重复 mapping、错误套用 cutoff、阈值硬编码。测试必须抓住这些错误才算有业务判别能力；仅有 not_null/类型测试不够。

已有 QA 的 Won reconciliation status 主要比较 counted Won Y1；它不能代替 Gap equality 断言。新增 gap 专属检查，分别报告计算偏差、输入快照不一致和显示舍入差异。

## 4. 架构验收矩阵

以下为后续实施的必跑测试，并非本轮测试结果。

| 编号 | 操作/故障 | 预期证据 |
|---|---|---|
| G01 | 回传伪造 `approved_by/customer_confirmed` | 不改变状态；服务端只认可受信身份 |
| G02 | 同 nonce 同内容重传；再传不同内容 | 前者同 receipt，后者 409，无新批准 |
| G03 | 客户审阅后 Spec/预览变更 | 原批准无法用于新 digest |
| G06 | 保持 decision_id，偷偷改变 Spec 的聚合/过滤/null 语义 | semantic_digest 不符，业务 gate 失效；新发布批准不能替代业务批准 |
| G04 | 无 scope 权限的成员批准 | 拒绝；已确认状态不变 |
| G05 | 同一场景/有效期出现冲突决策 | 阻止同时生效，要求显式取代或分场景 |
| R01 | 构建 R2 的一半时 query R1 | R1 的数值、meta、KB 均不变 |
| R02 | 连续查询跨 R1->R2 激活点 | 每个 query context 只出现一个完整版本 |
| R03 | 两个发布者基于同 active revision 提交 | 只有一个成功，另一方 409 |
| R04 | activation 已提交但响应断开后重试 | 返回原 receipt，revision 不重复增长 |
| R05 | QA 通过但 KB 载入失败 | 不允许激活，无 latest KB fallback |
| R06 | 旧 build 账号尝试修改 sealed relation | 数据库拒绝；不是仅 API 拒绝 |
| R07 | 回退到 R1，期间用户权限已撤销 | 数值版本可回退，用户仍无权读取 |
| R08 | 历史快照已删除后请求回退 | 明确拒绝并说明不可复现 |
| R09 | 历史发布 gate 已过使用期限，但有新的合法 rollback receipt | 按新 envelope 的当前 expected_active 提交；历史 payload 不变 |
| R10 | 模型 QA -> KB -> payload -> 最终 validation | 引用构成无环图；最终 receipt 不写回已冻结 KB |
| A01 | 同 datasource 不同 tenant、无 grants | 拒绝跨租户/空授权请求，含 discovery |
| A02 | 同 tenant 不同 project 查询 KB/preview | 未授权则拒绝，不按 collection 相同放行 |
| A03 | 通过旧 meta/grants/raw-query 路径访问 factory scope | 无旁路权限；strict 不回退 legacy |
| A04 | meta 带自由 `sql_expression` 或越界依赖 | prepare/validate 拒绝 |
| A05 | build 角色读取控制表、其他项目、执行越界函数 | DB/执行策略阻止，日志不泄露输入明细 |
| J01 | job 已落库、投递前 kill 网关 | outbox 恢复投递，不丢 job |
| J02 | worker 完成后重复消息 | 不重复封存/发布，复用完成 receipt |
| J03 | worker 失租后继续回传 | 旧 attempt 被拒，不覆盖新执行结果 |
| J04 | 工作流在等待人工确认时重启 | 仍是同 review_request，未自动通过 |
| D01 | 相同源文件重复导入；新 schema/新内容导入 | 幂等返回或新 snapshot，已封存输入不覆盖 |
| D02 | 同 Spec 两个 business run 并存 | 各自日期/阈值生效，不依赖唯一 active 行 |
| D03 | 所选必需测试被 SKIP 或缺失 | validation 失败，不能只凭退出码通过 |
| D04 | 源刷新或历史修正改变数字 | 新 data revision/水位和验收记录；旧版仍可追溯 |
| Q01 | 回答询问 Gap，但没有默认口径确认 | 澄清或并列候选，不能说客户已确认 |
| Q02 | 回答读取完整版本 | 值、单位、筛选、日期、状态、decision/QA、release 一致 |

## 5. 实施边界与可交付文件

先从这些最小文件/模块开始；路径是拟议布局，尚未创建实现：

```text
knowledge/projects/<project>/
  manifest.json                    # 服务端状态投影
  decisions/<id>/<revision>.json    # 已接受记录的不可变导出
  specs/<id>/<revision>.json
  reviews/<review_request_id>/
  releases/<release_id>/manifest.json
  kb/<release_id>/
  reports/<verification_id>/

factory/contracts/                 # Decision / Spec / Gate / Release schema
factory/cli/                       # 首版 runner 与工件生成器
metrics-server/.../governance/      # 权威记录、gate、release API
metrics-server/.../runtime/         # strict scope / version resolver 的增量实现
```

目录组织可以适配现有代码习惯，不能为追求上述名称搬动无关模块。新的数据库/工件权限边界与旧项目兼容策略需随 P2 PR 一起提交。

## 6. 人效实验与停止扩展条件

测量单位是“验收通过的变更”，不是生成文件数。

| 指标 | 记录方式 |
|---|---|
| 人实际投入 | 口径确认、技术 review、工具修复、返工分别计时 |
| 墙钟时间 | 执行时间与等待客户/排队分别记录 |
| 自动完成率 | 无非 gate 人工修复的已验收变更 / 全部已验收变更 |
| 缺陷捕获 | 生成时、静态校验、业务测试、对账、人审、上线后分别记录 |
| 一致性 | 混版次数、越权次数、未复现差异、误确认次数 |
| 复用成本 | 第二项目模板直接适用/修改/不适用的比例与修改时间 |
| 平台成本 | schema/存储、DB 负载、模型调用、维护/升级投入 |

只在 P2 故障测试通过、P3 确实捕获错误公式后扩展到 20 指标。若仍需频繁手改落地文件或 SQL，记录原因并修复对应契约/工具，不能把人工动作归类为 gate 来达成“非 gate 人工为零”。

业务操作者实验选择有权限但不写代码的人：完成一个 T2 指标和一个受约束 T3 变更，记录没有现场工程师代操作时的结果。试点通过也不自动代表新源/跨租户/T4 自助成立。

## 7. 本轮完成与仍待执行

本轮完成的是设计及源码证据核查：明确发布一致性协议、确认身份边界、状态/工件模型、Java 增量职责和以上验收场景。

尚未实施 Governance API、版本化 runtime、dbt runner，也未取得 Gap 客户确认、复算历史快照或执行矩阵测试。下一步可直接按 P0/P1 开工；真实 runtime 发布受 P2 验收条件约束。

## 8. AI-native BI 增补验收

依据[已验证分析库、问题评测与建模 Copilot](./verified-analysis-and-copilot-design.md)。以下仍是未来必跑场景，不是已通过结果；先增加到 P1/P2/P3 切片，不先扩大指标数量。

| 编号 | 场景 | 预期 |
|---|---|---|
| BI01 | 同义问法命中已验证方案，合法参数改变 | 记录 recipe revision；重新执行；不复用历史数值 |
| BI02 | 问 Gap 且默认未确认；再明确 Action/Signed | 先澄清或并列标签；明确后合成样例分别返回 10/-20 |
| BI03 | Spec 单位、粒度、时间规则或语义摘要变化 | 旧 recipe 不自动复用；新兼容验证完成前澄清/报告不可用 |
| BI04 | 当前用户尝试跨 scope 搜索方案、KB 或使用缓存 | 在返回内容前拒绝；撤权后缓存也不能读 |
| BI05 | 同一 family 的改写分到教学/保留题；修复 Agent 请求读答案 | 数据集检查和权限拒绝；已披露题转开发集并重新计覆盖 |
| BI06 | 先 sum 后 clamp、明细 join 放大、阈值硬编码 mutation | 独立 fixture 和问题测试抓住错误；自身生成的测试不是唯一依据 |
| BI07 | NULL/0、合法空结果、截断结果、金额舍入与 Top-K 并列 | 分别按显式契约判分；不统一当空集或忽略顺序 |
| BI08 | SQL/数据不变，只切 prompt/模型；与另一发布并发 | 新 AgentRelease + 组合评测 + tuple CAS；旧单 release API 无旁路 |
| BI09 | 问答失败被删题/SKIP 或放宽期望；仅 LLM judge 说通过 | 缺失不计 PASS；策略变更需独立批准；硬断言不受 judge 覆盖 |
| BI10 | Builder 元数据索引过期，依赖未知或 build 漏跑 | 刷新/扩大验证或阻止通过；清楚列出未执行项 |
| BI11 | 文档或源码注入要求 Builder 扩权、改 gate、原地发布 | 运行环境与服务端权限拒绝；不依赖 prompt 自觉 |
| BI12 | 多轮追问时中途激活新版；旧 context 过期或数据已删 | 同 Run 不混版；不可继续时明确中止/新建 Run |
| BI13 | 值正确但 Claim 因果越界、单位错、图表系列映射错 | 表达/图表检查失败，不能用数值正确率掩盖 |
| BI14 | 模型超时/重试/取消，多次执行波动较大 | 有超时和取消回执；无隐式换模型；报告耗时/成本及重复运行波动 |
| BI15 | 回退到含已禁用 prompt/工具/Recipe 的历史 tuple | 拒绝；合法语义版本不能抵消 Agent 风险 |
| BI16 | Recipe 验证 -> Agent/语义 payload -> 问答评测 -> 激活；同 tuple 替换评测回执 | 摘要引用无环；外部问答回执完整绑定两类 payload、Suite 和输入；替换回执后旧批准失效 |

P1 产出候选方案、问题族和独立样例；P2 加版本化分析执行、问题评测 CLI 与组合发布；P3 在 Sell-in 单模型上验证 Copilot patch、构建修复和答案回归。官方 skills/MCP/Wizard 仅为可选 adapter，尚未安装或验证兼容。

发布报告分别列开发集和保留集，不混成一个“准确率”。必跑安全/核心算术题不得失败；其他门槛在正式对比前固定。只有开发回归时明确说明泛化评测尚缺，不提前宣称性能提升。
