# dbt 动态部署设计

- 状态：draft v0.1
- 回答的问题：多项目、agent 驱动场景下，dbt 的动态部署怎么落地
- 关联：[README](./README.md)、[m1-bootstrap-design.md](./m1-bootstrap-design.md)、[security-notes.md](./security-notes.md)、[open-questions-recommendations.md](./open-questions-recommendations.md) Q1/Q4

## 0. 拆题

"动态 deploy" 是三个问题的合称，解法各不相同：

| 子问题 | 答案 |
|---|---|
| 项目怎么来 | `project init` 动态生成 dbt 工程脚手架；文件即注册表，dbt 无需"动态注册模型" |
| 构建怎么跑 | 一个 dispatcher 服务所有项目：按 manifest 动态渲染 profiles，`--project-dir` 隔离执行 |
| 变更怎么上线 | **两段式**：`dbt build`（技术部署）≠ runtime 可见（经 Publisher diff + 发布 gate） |

## 1. 项目脚手架：onboard 时动态生成

`project init`（M1）在 `knowledge/projects/<project_id>/dbt/` 下生成：

```
dbt/
├── dbt_project.yml          # name=<project_id>, profile=<project_id>
├── macros/generate_schema_name.sql   # 工厂统一宏
├── models/
│   ├── staging/             # import 时自动生成的初始 stg 模型
│   ├── intermediate/
│   └── marts/
└── packages.yml             # 默认为空，供应链面（security-notes §5）
```

一个重要协同：**M1 的 `import` 动词产出导入回执（manifest.json：sheet→表→列映射），据此可自动生成初始 staging 模型 + schema.yml 基础测试**（每列 not_null/类型断言）。也就是说 landing 落地完成时，dbt 工程的 staging 层已经存在——Explorer/Modeler 从 intermediate 层开始工作。

**profiles.yml 永不入库**。它由 dispatcher 在执行时从 manifest（datasource 条目）+ secrets 动态渲染：

```yaml
hankel:
  target: "{{ env_var('DBT_TARGET', 'ci') }}"
  outputs:
    ci:
      type: postgres
      host: "{{ env_var('PGHOST') }}"
      user: hankel_dbt_ci          # security-notes §3 角色矩阵
      schema: "hankel_ci_{{ env_var('PR_NUMBER', 'local') }}"   # PR 影子 schema
    prod:
      type: postgres
      host: "{{ env_var('PGHOST') }}"
      user: hankel_dbt
      schema: hankel_prod
```

schema 命名由工厂统一宏控制（前缀 = 项目，后缀 = 环境隔离）：

```sql
{% macro generate_schema_name(custom_schema_name, node) -%}
  {{ target.schema }}{% if custom_schema_name %}_{{ custom_schema_name | trim }}{% endif %}
{%- endmacro %}
```

## 2. Dispatcher：一个 runner 跑所有项目

不为一项目部署一套服务。一个薄 dispatcher（M1 CLI 的 `dbt` 动词组，后续可暴露为网关端点）：

```
factory dbt build   --project hankel --target ci  [--select state:modified+ --state <prod_artifacts>]
factory dbt deploy  --project hankel              # build --target prod + 收集 artifacts
factory dbt publish --project hankel              # 见 §4，与 deploy 分离
```

执行机制：

1. 按 project_id 读 manifest → 解析 datasource 连接信息（secrets 注入环境变量，不落盘）。
2. 渲染 profiles.yml 到临时目录（tmpfs，执行后删除）。
3. 在 pinned-digest 的 dbt 容器/venv 里执行：`dbt build --project-dir ... --profiles-dir ... --target ...`。
4. 收集 `manifest.json`、`catalog.json`、`run_results.json` 存入项目工件注册表——**run_results 就是 Modeler agent 的迭代输入**（读失败结果→改 model→重跑的标准接口）。
5. 日志脱敏（连接串、密码）后归档。

Python 侧可直接用 `dbtRunner()` 编程调用（Q1 的留后路），dispatcher 本身先 bash/python 实现即可。

**增量 CI**：PR 构建用 `dbt build --select state:modified+ --defer --state <main 的 manifest>`——只构建改动模型及其下游，其余 defer 到 main 已有产物。这让 agent 的高频小改动在 CI 上保持分钟级。

## 3. 环境与 promotion 流

```
agent 分支写 model+test
  → PR：CI 跑 build --target ci（写入 hankel_ci_pr<N> 影子 schema，全量可预览、可查）
  → 评审人查影子 schema 验证 → merge main
  → CI 跑 deploy：dbt build --target prod（in-place replace）
  → deploy 后自动跑对账（循环 C gate）→ Publisher 出 diff → 发布 gate（人）
```

**关于蓝绿/schema 切换的明确取舍**：经典做法是构建到影子 schema 再整体 rename 切换，但它与本项目架构冲突——runtime meta 和 EXACT grants 都是 schema 限定的稳定对象引用（`public.hankel_view_*`），schema 换名会使已发布 meta 全部失效。因此**选择 in-place replace**：

- 当前设计全部是普通 view，`CREATE OR REPLACE` 单对象原子、瞬时完成；
- DAG 内先后替换存在短暂不一致窗口，分析型负载可接受，构建窗口避开查询高峰，build 后自动跑对账兜底；
- 未来若引入物化表且窗口不可接受，再做"双对象 + 指针 view"模式（表分层，最外层 view 永远稳定命名），而不是 schema 切换。

## 4. 两段式上线：dbt build ≠ runtime 可见

这是动态部署与整个工厂架构最重要的衔接点：

1. `dbt build` 成功只意味着 PG 里出现了新 view——**runtime 仍然不可见**（grants 未发、meta 未发布）。
2. Publisher 读 `catalog.json`/`manifest.json` 与已发布 meta 做 diff：新增/变更/消失的表和列 → 生成 meta 发布载荷 + EXACT grants 变更。
3. diff 经**发布预览 gate**（确认 app 模式复用）→ `meta/tables`、`meta/metrics` 发布 → runtime 可见。

安全收益：即使 Modeler agent 被劫持生成了恶意 view 并混过 CI，它也**到不了 runtime**——发布 gate 是最后一道独立于人不可过的门。dbt 侧的技术部署可以放开自动化，因为业务上线权牢牢在 Publisher + 人这里。

## 5. 触发面

| 触发 | 通道 | 凭据 |
|---|---|---|
| PR（常规） | git webhook → CI | CI secrets（ci 角色） |
| PM agent 重跑（决策变更后重建） | dispatcher 端点/CLI | ci 角色；agent 永不持有 prod 凭据 |
| 源数据刷新后的定时重建 | cron → dispatcher `deploy` | prod 角色（只此一条通道） |
| `dbt source freshness` 失效告警 | CI 检查 | 只读 |

## 6. 最小实现路径

1. **M1 内**：`project init` 生成脚手架 + `import` 自动产出 staging 层；dispatcher 先实现为 CLI 动词（手动触发、专用角色）。
2. **M3 内**：接 git webhook → CI PR 构建（影子 schema 预览）→ merge 后 `deploy` + 自动对账；`state:modified+ --defer` 增量构建。
3. **之后**：`publish` diff 自动化 + 发布预览 gate 接确认 app；PM agent 经 dispatcher 触发重建。

## 7. 与安全设计的衔接

- 每项目/每环境独立 PG 角色（security-notes §3 矩阵），dispatcher 是唯一持有凭据解析权的组件；
- profiles 运行时渲染、tmpfs、日志脱敏；
- agent 的部署权 = 提 PR；cron 的 prod 重建是唯一免 PR 通道，但其变更源仍是已过评审的 main；
- 供应链：dbt 镜像 pinned digest、packages 锁版本。

## 8. 宿主选型：Airflow、Java 异步任务、还是 Python 异步任务？

**结论：Airflow 现在不上；Java 内嵌明确不选；最终宿主是小型 Python 异步服务（dispatcher 进化版）。**

| 方案 | 判断 | 理由 |
|---|---|---|
| Airflow | 现在不上 | `dbt build` 是单命令，模型间 DAG 由 dbt 自己编排；Airflow 的核心价值（回填、跨 DAG 依赖、sensor）与主负载（PR 触发 + gate 驱动）不匹配；引入它等于**引入第二个编排大脑**——gate 活在工厂状态机里而不是 DAG 里，治理会出现两套真相；运维成本（scheduler/webserver/worker/元数据库）与当前负载不成比例 |
| Java（metrics-server 内嵌） | 明确不选 | dbt 是 Python CLI，Java 只能 ProcessBuilder 外壳调用，venv/版本/日志都是坑；更根本的是违反 builder/runtime 分离——metrics-server 是 runtime 与控制面的所有者（security-notes §2），绝不能同时持有 builder 侧执行权 |
| Python 异步任务 | **最终形态** | 与 dbt 同栈；复用 document_service 已验证的 Celery + Redis 运维经验 |

关键认知：**这个工厂里"编排者"已经存在**——PM agent + 项目状态机负责跨环节编排（gate 驱动），dbt 负责模型间 DAG。宿主组件只需要提供三样东西：触发入口、异步执行、状态与 artifacts 回传。这正是 dispatcher 的职责范围，不需要也不应该再引入一个通用编排平台。

### 8.1 宿主演化三阶段

1. **M1–M3**：dispatcher = CLI（人/CI 执行），零常驻组件。
2. **需要常驻时**（PM agent 触发重建 + cron 定时重建）：新增 `factory-runner` 小服务——FastAPI 受理与查询（`POST /dbt/build {project, target}` 返回 job_id；`GET /dbt/jobs/{id}`）+ Celery worker 执行（队列按项目分）。工件（run_results/catalog）写入项目注册表，完成后 webhook 通知网关。凭据解析权仍只在 runner。
3. **重估判据**（满足任一再考虑调度平台）：多项目定时 SLA、源新鲜度传感器、复杂回填、跨项目依赖图。届时优先 **Dagster**（dbt 资产原生集成，模型即资产，与工厂的工件化理念同构）> 保留 cron + dispatcher > Airflow（仅当团队已有 Airflow 运维能力或需其生态）。

### 8.2 常驻形态的工程要点

- **并发防护**：同一 project+target 用 PG advisory lock 互斥（dbt build 本身幂等——view replace，但并发替换会产生无谓竞争）。
- **重试策略**：仅对基础设施错误（连接、超时）自动重试；模型/测试错误不重试，原样回传给 agent 迭代。
- **调用方**：agent 网关（Node/lattice）与 metrics-server 都只作为 HTTP 调用方，永远不直接执行 dbt。
- **PM agent 的落点**：`agent/` 已是 @axiom-lattice 工作流引擎（workflow runs 已持久化 tenant/workflow/状态），PM agent 的状态机直接长在 lattice workflow 上，经 HTTP 触发 factory-runner——编排、执行、治理三层各归其位，无需新平台。
