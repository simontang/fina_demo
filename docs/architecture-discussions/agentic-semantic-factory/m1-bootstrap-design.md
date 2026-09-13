# M1 具体设计：项目 Manifest 与 Bootstrap 工具

- 状态：draft v0.1（供评审，可开工）
- 上游文档：[README（方法论与目标架构）](./README.md) §7 最小入手路径
- 事实依据：`tmp/hankel_tenant_setup`（空目录——tenant setup 全程手工、零留痕）；`tmp/hankel_new_raw_import/`、`tmp/hankel_report_import/`（manifest + SQL + 远程执行脚本，landing 的事实原型）；`metrics-server/docs/api-inventory.md`、`agent-tools-datasource-metrics-guide.md`（可包装的 API 面）

## 0. 为什么先做 M1

Hankel 项目的启动证据：`tmp/hankel_tenant_setup/` 是空目录。租户、datasource 15、grants、KB 骨架全部通过手工 curl 完成，过程只存在于操作者的 shell history 里。与此同时，数据导入环节反而留下了结构化工件（manifest.json 导入回执）。M1 的目标就是把这半套"无意中做对的"推广到整套启动流程。

## 1. 项目 Manifest Schema（v0.1 草案）

每个项目一份，是 PM agent 与所有环节 agent 的唯一事实源。建议路径：`knowledge/projects/<project_id>/manifest.json`（与 README §6 目录约定一致）。

```jsonc
{
  "schema_version": "0.1",
  "project_id": "hankel",
  "display_name": "Henkel ACM Run for Gold",
  "created_at": "2026-09-03",
  "participants": [
    {"role": "owner", "name": "..."},
    {"role": "builder", "name": "..."},          // 可走 grants + governed query
    {"role": "client_confirmer", "name": "..."}  // 口径决策的确认人
  ],
  "datasources": [
    {
      "datasource_id": 15,
      "instance_type": "postgres",
      "schema": "public",
      "purpose": "landing_and_marts",
      "note": "凭据只存于 datasource store，manifest 不落密码"
    }
  ],
  "sources": [
    {
      "source_id": "src_new_order",
      "kind": "excel",
      "files": ["Data of New Order.xlsx"],
      "landing_tables": ["hankel_new_order_lines"],
      "import_receipt": "imports/2026-09-03-new-order/manifest.json"
    }
  ],
  "kb": {
    "kb_id": "hankel-metrics-analysis",
    "collection": "hankel",
    "templates_applied": ["answer-contract", "evidence-priority"]
  },
  "lifecycle": {
    "stage": "verified",
    "gates": [
      {
        "gate": "scope_confirmed",
        "passed_at": "2026-09-04",
        "decision_ids": ["D-2026-001"],
        "artifact": "decisions/*.json"
      }
    ]
  },
  "artifacts": {
    "dbt_project": "dbt/hankel/",
    "quality_reports": "reports/",
    "reconciliation_baseline": "reports/baseline/"
  }
}
```

设计规则：

1. **凭据零落盘**：manifest 只引用 `datasource_id`，密码只在 datasource store（`POST /datasources` 落库前 AES 加密的既有机制）。
2. **gate 记录指向 decisions**，不复制内容——口径的单一事实源是 decisions/*.json。
3. **manifest 由工具读写**，人只读；避免手改后与实际资源漂移。
4. schema_version 从 0.1 开始，允许演化；PM agent 拒绝无 schema_version 的 manifest。

## 2. Bootstrap 工具：六个动词

形态：一个 CLI（先用 bash/python 脚本即可，不必先做服务），每个动词幂等——**先查现状，再补差异**，重跑不产生重复资源。

| 动词 | 做什么 | 包装的现有能力 | 幂等策略 |
|---|---|---|---|
| `project init` | 访谈答案（YAML/问答）→ 生成 manifest + 项目目录骨架（decisions/、reports/、dbt/、kb/） | 无（新建） | manifest 存在则校验并补缺失目录 |
| `datasource register` | 注册 datasource + 连通测试 + 激活 | `POST /api/v1/datasources`、`/test`、`/enable`（见 api-inventory） | 按 name 查重，存在则 diff 配置 |
| `import` | Excel/CSV → 建表 + 灌数 + 导入回执 | **提升 `tmp/hankel_*_import` 脚本**：sheet 探查（workbook_summary）、列映射（Excel 表头→snake_case + 类型）、SQL 生成、manifest 回执 | 回执中 row_count + source 文件指纹比对，重复导入显式拒绝或显式覆盖 |
| `grants init` | 建立初始 table-grants 基线（landing 表只授 builder 探查） | `table-grants` API | 按 datasource 现有 grants diff |
| `kb init` | 从模板实例化 KB 骨架：manifest.json、glossary、agent-policy、质量规则（引模板） | hankel-metrics-kb 结构做模板 | 目录存在则不覆盖，只补缺 |
| `status` | 展示生命周期阶段、已过 gate、各资源实际状态与 manifest 的漂移 | 各 API 的只读查询 | 只读 |

`project init` 的访谈输入刻意最少：project_id、display_name、参与者角色、数据源清单（文件/连接）、业务目标一句话、已知术语。其余留给探查环节回来补——避免"启动时逼人回答之后才会知道的问题"。

**明确不做**（M1 范围控制）：不做 Web UI、不做多租户批量子项目、不改 metrics-server 任何代码、不实现 dbt（M3）。

## 3. Decisions Schema（v0.1 草案，M2 消费、manifest 引用）

```jsonc
{
  "decision_id": "D-2026-001",
  "project_id": "hankel",
  "topic": "new_order_gap_display",
  "question": "New Order Gap 对外默认展示 Action Gap（非负）还是 Signed Gap（可为负）？",
  "options": [
    {"key": "action_gap", "implication": "对外主指标不小于0；Signed Gap 保留为底层指标"},
    {"key": "signed_gap", "implication": "负值表示超额覆盖，直接可见"}
  ],
  "chosen": "action_gap",
  "status": "customer_confirmed",   // draft | customer_confirmed | superseded
  "scope": {"applies_to": ["metric:hankel_new_order_gap"]},
  "evidence": [
    {"kind": "client_response", "ref": "evidence/responses/acm-poc-response-20260903-1838.json"}
  ],
  "decided_by": "client_confirmer",
  "decided_at": "2026-09-04",
  "rationale": "..."
}
```

这条就是 KB05 待确认表第 9 行的实例化——M2 的第一批积压素材（11 条待确认项）可以逐条套这个 schema。

## 4. 确认 App 的最小形态（M2 预研结论先记在这里）

- **面向客户**：纯静态 HTML（决策积压内嵌），逐条选择 + 批注，底部"导出 decisions.json"。零服务端依赖，链接可直接发给客户。
- **面向内部 gate**：同一生成器，输出可 POST 回 gateway 的模式，减少手工搬运。
- 生成器输入 = 决策积压 JSON（Profiler/Curator 产出），输出 = 静态页。app 本身无状态，状态全在 decisions 文件里。

## 5. 验收标准（对应 README §7 M1 行）

1. 用 `project init` onboard 一个空测试项目（建议直接用 agentic_cdp 数据当第二个"客户"），产出 manifest + 完整目录骨架。
2. `datasource register` + `import` 全程无手工 curl/psql；导入回执含 source 文件、sheet、行数、列映射。
3. 全部动词重跑一次，无重复资源、无报错（幂等）。
4. `status` 能如实报告 manifest 与实际资源的漂移（手工删一个表后能显示出来）。
5. 全程无任何凭据出现在 manifest 或日志中。

## 6. 已知风险与取舍

- **tmp 脚本直接提升 vs 重写**：`hankel_*_import` 脚本里硬编码了 hankel 表名/路径，提升时把"项目无关的机制"（Excel→SQL→回执）与"项目相关配置"（表名、列映射）分离，配置外置到 manifest/sources。
- **远程执行方式**：现脚本用 `docker inspect` 从容器环境变量挖 PG 密码，属于 demo 权宜。M1 保留该模式但收敛到一个函数，标注 TODO 换成正式部署通道，不在 M1 展开。**安全约束见 [security-notes.md](./security-notes.md)：dbt/导入工具一律不得复用 `SPRING_DATASOURCE_*` 凭据，须建项目专用 PG 角色。**
- **manifest 漂移**：凡 metrics-server 侧可编程查询的资源（datasource、grants、meta），`status` 以服务端为准；纯文件资源（decisions、KB）以文件为准。
