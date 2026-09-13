# 清洗放哪：ETL 逻辑 vs 全 SQL

- 状态：draft v0.1
- 回答的问题：需要独立的 ETL 清洗逻辑吗，还是所有清洗都能用 SQL 解决？
- 关联：[README](./README.md)、[m1-bootstrap-design.md](./m1-bootstrap-design.md)、[dbt 动态部署](./dbt-dynamic-deploy.md)

## 0. 结论

不需要独立的"清洗层"。问题不是"要不要 ETL"，而是**清洗规则写在哪一层**：

- **EL（Extract/Load）必须是代码**（Python/bash）——解析 Excel/PDF、抽取、落地。但这一层要**薄且忠实**：使命是把数据原样落进 landing 并留痕，**明确不做清洗**。
- **T（Transform）尽量全是 SQL**（dbt staging/intermediate）——一切规则型清洗都在这里，版本化、带测试、有血缘。

Hankel 已经实证了这个划分：`run-for-gold-views.sql` 里的全部清洗（`.0` 标识符规范化、姓名 mapping join、trim/nullif、reject 过滤、KB05 空值规则）都是 SQL；非 SQL 的部分恰好都是 EL 和诊断（Excel 解析脚本、workbook_summary）。这不是巧合，是正确的分工被无意遵守了。

**不变式（升格为 README P11）**：数据一旦 landing，一切"数据→数据"的变换只有 SQL 一条通道。各组件与数据的关系：

| 组件 | 读 | 写 |
|---|---|---|
| loader | 外部源 | 仅 landing |
| dbt SQL | landing / 下层模型 | 仅 staging / intermediate / marts（变换链唯一生产者） |
| Profiler | 任意（只读诊断） | 报告与决策积压（非数据） |
| ML/打分 | 经提取或 marts | 仅新的 landing 表（作为新源，不得直写变换链） |
| Sandbox Python（比对/统计/探索） | 经沙箱通道：数据交接或只读角色 | 仅工件（报告/图表/统计结果）；需持久化时走落地通道 |
| Reconciler / tests | 只读 | 无 |
| Runtime agent | 仅经 meta + metric query | 无 |

**统一逃生门**：SQL 表达不了的逻辑（模糊匹配候选生成、迭代算法、外部模型推理），一律**先产出数据落地为新源，再由 SQL 消费**——永不插队进变换链。这条不变式的三个直接收益：安全上 builder 侧的写路径收敛为 dbt 角色一条；agent 能力上 Modeler 只需 SQL 技能；治理上 dbt DAG 成为真相的完整地图。

## 1. 判定规则：一个任务进哪层

| 任务特征 | 去哪 | 例子 |
|---|---|---|
| 把外部世界变成表 | **Python loader**（写 landing，忠实+留痕） | Excel 多 sheet 解析、PDF 抽取、异构库抽取 |
| 判断数据有什么问题 | **Python profiler**（只读，产物是报告/决策积压） | 空值分布、异常金额候选、映射缺口 |
| 每行/每组的确定性规则 | **SQL（dbt staging）** | 类型规范、去重、白名单过滤、match key、单位换算 |
| 口径变化 | **decision → SQL 实现回流** | 白名单规则变更、gap 口径二选一 |

一句话检验法：**这条清洗规则能不能写成一列 SELECT 表达式或一个 schema.yml 测试？** 能 → SQL；不能，且它的工作是"变出表" → loader；不能，且它的工作是"评价数据" → profiler。**唯一合法的"Python 写数据"是 loader 写 landing**——其他一切修改数据的代码都是 smell。

## 2. 为什么清洗要尽量 SQL：这是治理要求，不是风格偏好

1. **决策对齐**：每条清洗规则都对应一条已确认决策（KB05 的空值/重复/单位规则）。dbt staging 模型是声明的规则本身，可以逐条追溯到 `based_on_decisions`；Python 过程代码描述的是步骤，业务口径对不上步骤。
2. **血缘可走通**："这个数为什么是这个值"沿 dbt DAG 就能走完；进过 Python 黑盒的血缘断了，对账失败时无从解释——而 KB 的铁律恰恰是差异必须可解释。
3. **可测试**：SQL 清洗直接配 dbt tests（`.0` 残留=0、白名单外=0，06 的线上验证就是现成的测试集）；Python 清洗要另建测试体系，实际没人建。
4. **幂等可重放**：view/model 重算即重放；带副作用的脚本重放靠信仰。
5. **agent 可生成可评审**：LLM 写 SQL 的可靠性远高于写过程式清洗代码；SQL 变更走 PR 一目了然，Python diff 需要逐行脑内执行。

## 3. Python 的合法地盘（完整清单）

| 组件 | 职责 | 与数据的关系 |
|---|---|---|
| loader（M1 `import` 的内核） | 解析、schema 推断、落地、回执 | 只写 landing，忠实原样 |
| Profiler | 统计画像、异常候选、映射缺口 | 只读；产物是质量报告 + 决策积压 |
| document_service | PDF/文档抽取 | 产物落地为表 |
| ML/打分（如 CDP propensity） | 模型推理 | **输出落地为数据表**，作为 landing 的一种源，后续归 SQL |
| Sandbox 分析（比对/统计/探索） | 受控只读计算 | 产物为工件；需持久化则走落地通道 |
| 工具链 | dispatcher/factory-runner、KB 编译器、确认 app 生成器、publish 脚本 | 不碰业务数据 |

ML 那一行是通用模式：**代码的输出落地为数据，SQL 接管后续**——模型分数和 Excel 文件在架构里地位相同。

## 4. 反模式（第一条有 repo 实据）

1. **清洗逻辑藏在源文件里**：`tmp/hankel_dist_review_import/formula_dependencies.csv` 证明源 Excel 里带公式依赖——业务口径活在客户工作簿的公式里，既不可测也不可追溯。把这类逻辑"拔出来"放进 governed staging 模型，正是这个工厂存在的理由之一。
2. **Python 清洗黑盒**：结果对不上无法解释，规则变更无处留痕，评审等于重写。
3. **Landing 阶段"顺手修数据"**：loader 静默转类型、丢空行、合并重复——直接违反 KB05（保留并标记，不静默清零）。landing 必须忠实，脏数据是诊断的对象，不是 loader 的敌人。

## 5. 沙箱分析通道（Python 读数据的合法形态）

SQL 覆盖"规则型"分析，但**比对、显著性检验、分布统计、探索性计算**这类任务用 Python 更自然。沙箱让这件事与 P11 兼容：Python 可以读数据、做计算，但它的**输出永远是工件或新源，永远不是变换链里的数据**。

### 5.1 两种接入模式

| 模式 | 机制 | 适用 |
|---|---|---|
| **数据交接**（推荐默认） | harness 执行 SQL 导出（parquet/dataframe）→ 递给沙箱 → 沙箱纯计算 → 返回工件 | agent 生成的临时分析；沙箱**零凭据、零网络**，泄露面最小 |
| **只读角色** | 沙箱容器持 `*_qa` 只读角色（schema 白名单），自行发 SQL | Profiler 等需要多轮探查的常驻服务 |

### 5.2 沙箱约束（缺一不可）

1. 无网络出口（或仅 allowlist）——杜绝数据外带；
2. 只读权限或零凭据——写路径物理不存在；
3. 资源上限（CPU/内存/时长/结果集大小）——防失控查询；
4. 非 root + tmpfs 工作区 + pinned 镜像——可复现；
5. 代码即工件：沙箱脚本版本化、可重跑、进 PR 评审（agent 生成也不例外）；
6. 运行留痕：谁、何时、什么代码、读了哪些 schema、耗时与行数。

### 5.3 输出的两条合法出路

- **报告通道**：统计结果、图表、比对结论 → 工件注册表，供确认 app/洞察报告引用；
- **落地通道**：计算结果值得持久化（打分、模糊匹配候选）→ 经 manifest 声明写入**新的 landing 表**，从 SQL 门重新进入，获得血缘与治理。

除此之外没有第三条路：沙箱直写 staging/marts 不存在，也不允许存在。

### 5.4 与安全设计的关系

沙箱是 §2"P10 隐私边界"和 security-notes §5 的执行点：agent 生成的代码**敢跑**，靠的正是容器隔离 + 只读/零凭据 + 资源上限——即使代码被 prompt injection 污染，爆燃半径 = 一次失败的计算。隐私侧默认只见聚合与抽样，明细仅在授权会话内按最小化原则进入沙箱。

## 6. 落地动作

1. M1 的 `import` 生成回执时，同步生成 staging 模型骨架——**清洗规则从第一天就住在 SQL 里**，而不是先脚本后迁移。
2. Profiler 明确定位为只读诊断器，任何"建议修复"都以决策积压形式出现，经确认后变成 staging 模型的一条 SQL 规则 + 一个 test。
3. Code review 红线：PR 里出现"landing 之后的 Python 数据写操作"即打回，要求改成 dbt 模型或证明其属于 loader。
