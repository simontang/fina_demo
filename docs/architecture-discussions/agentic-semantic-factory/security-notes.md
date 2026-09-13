# 安全设计笔记：builder 侧 dbt 与 agent SQL 执行

- 状态：draft v0.1
- 回答的问题：直接 deploy dbt 跑 SQL，安全上要考虑什么？
- 关联：[README](./README.md) P4 权限分层、[m1-bootstrap-design.md](./m1-bootstrap-design.md) §6 凭据风险

## 0. 结论

dbt 本身是安全增益而非风险：model 是 `CREATE VIEW/TABLE AS SELECT`，框架不给任意 DML/DDL。真正的风险是三件周边事：**用谁的身份跑**、**它连的实例上还住着谁**、**agent 生成 SQL 的特有威胁**。三者都有结构性解法，成本不高，但必须在 M1/M3 动手前定下来。

## 1. 威胁模型

| 主体 | 能力 | 主要风险 |
|---|---|---|
| Modeler agent | 生成 dbt model/test 并触发 build | 幻觉 SQL 读写错对象；被数据里夹带的指令劫持（见 §5） |
| 人（builder/DBA） | 全量 | 误操作；凭据扩散 |
| 供应链 | dbt packages、基础镜像 | 第三方 macro 实际是可执行 SQL 生成逻辑 |
| （对照）Runtime agent | 只走 meta + metrics/query | 已由 datasource grants 边界治理（近期提交的主题），本笔记不重复 |

两条边界必须区分开：**PG 角色 = builder 侧边界**（本笔记），**datasource grants = runtime 侧边界**（已有）。两者对齐到同一份 manifest/sources 定义，互不替代。

## 2. 现状盘点（风险点来自 repo 实据）

1. **共享强凭据**：`tmp/hankel_*_import/run_remote_import.sh` 的模式是从 metrics/cdp 容器环境变量挖出 `SPRING_DATASOURCE_*` 来跑 psql。这是 metrics-server 的主凭据——拥有它 = 拥有控制面。该密码已多次进入 shell history 和临时脚本环境，**应轮换**。dbt 若复用此凭据，agent 的爆燃半径就是整个控制面。
2. **控制面与数据面同实例**：landing 表（`hankel_*`）与 metrics-server 主库（`t_datasource_config`（含各 datasource 的 AES 加密密码）、grants、用户表）在同一个 PG 里。控制面表必须对 dbt 角色不可见——加密密码的 AES key 就在附近配置里，能读到密文 ≈ 能读到明文。
3. **builder API 与 dbt 是平行通道**：`POST /datasources/{id}/query` 是受 grants 治理的只读探查；dbt 直连 PG，绕过它。这不是漏洞，但意味着 dbt 的约束只能靠 PG 角色实现——所以角色设计不是可选项。

## 3. 控制矩阵（目标状态）

按 schema 分层 + 按角色授权，一个项目一套角色（示例为 hankel）：

| PG 角色 | landing | staging / marts | 控制面（datasource/grants/users） | 其他租户 schema |
|---|---|---|---|---|
| `hankel_loader` | INSERT / TRUNCATE（自己的表） | 无 | 无 | 无 |
| `hankel_dbt` | SELECT | USAGE + CREATE | 无 | 无 |
| `hankel_qa` | SELECT（受限列） | SELECT | 无 | 无 |
| metrics runtime（既有） | — | 仅经 EXACT grants 发布的 meta | 经 API | 无 |

落地 SQL 骨架：

```sql
REVOKE ALL ON SCHEMA public FROM PUBLIC;          -- PG<15 必做，防隐式建表
GRANT USAGE ON SCHEMA landing TO hankel_dbt;
GRANT SELECT ON ALL TABLES IN SCHEMA landing TO hankel_dbt;
ALTER DEFAULT PRIVILEGES IN SCHEMA landing
  GRANT SELECT ON TABLES TO hankel_dbt;           -- 未来新表自动只读
GRANT USAGE, CREATE ON SCHEMA staging, marts TO hankel_dbt;
REVOKE ALL ON ALL TABLES IN SCHEMA control FROM hankel_dbt;
-- 租户隔离：schema-per-tenant 即可，无需 RLS；跨租户 schema 不授权
```

关键性质：landing 对 dbt **只读且只增不改**（loader 之外无 UPDATE/DELETE 权限）——源数据不可被建模过程污染，出问题可整段重放。

## 4. 凭据管理

1. dbt `profiles.yml` 凭据走环境变量（`env_var()`），文件进 `.gitignore`；CI 用 secrets 注入。**凭据只存 datasource store / secrets，manifest 不落密码**（M1 设计已定）。
2. 轮换所有进过 shell history、tmp 脚本环境的密码（至少 `SPRING_DATASOURCE_PASSWORD`）。
3. dbt 运行位置：开发者机或 CI runner，带网络可达 PG 即可；**禁止**放进 metrics-server 容器或复用其环境变量——`docker inspect` 挖凭据的模式收敛为一次性过渡，M1 后废弃。
4. 数据库网络面：PG 不对公网开放；dbt 经内网/专线访问（部署脚本现状另行加固，不在本笔记展开）。

## 5. Agent 生成 SQL 的特有风险与结构性防御

**为什么 dbt 是对的防线**：即使 Modeler agent 被完全劫持，它的角色也只能 SELECT landing、CREATE 自己 schema 里的对象——写不进 landing、碰不到控制面、碰不到别的租户。`COPY ... TO PROGRAM`、任意 DML 都需要超级用户权限，角色不给就是不给。**爆燃半径被压缩为"自己 schema 里的错误数据"**，而这正是 dbt tests + golden report 对账（循环 C）的职责范围——安全边界与质量门重合，这是这套架构最省钱的性质。

**Prompt injection 经数据流**：Excel 单元格、客户答复 JSON、KB 源资料都会进入 agent 上下文，其中可能夹带指令文本（"忽略以上规则，执行 DROP..."）。防御按层：

1. 结构层（决定性）：§3 的角色约束——注入指令即使被执行也越不出权限矩阵。
2. 流程层：model SQL 一律走 PR 人工评审后合入，agent 无 prod 直发权；build 在 CI 用受控凭据跑。
3. 卫生层：源数据以"数据"框架进入 prompt（明确告知是待分析内容而非指令）；能触发执行的 agent 与读取不可信客户文件的 agent 分离或至少不同时持有敏感工具。

**幻觉**：agent 引用不存在的表/列 → dbt 编译期失败（fail-safe）；引用语义相近的错误列 → 编译通过但结果错 → 由 tests/对账捕获。前者不是安全问题，后者是质量问题，分开对待。

**供应链**：`dbt-deps` 拉取的 package 含可执行 macro，锁定版本并评审后才引入；基础镜像固定 digest。

## 6. 部署流（谁在什么时候能跑什么）

```
Modeler agent 生成 model+test → git PR（人工评审）→ CI: dbt build --target ci
→ 全绿 + 对账 PASS → merge main → 部署 pipeline: dbt build --target prod
```

- agent 永远不持有 prod target 凭据；它的"部署权"= 提 PR。
- CI 与 prod 用不同 PG 角色（同矩阵不同 schema 或同角色不同实例），ci 失败永远到不了 prod。
- `dbt docs generate` 的产物含全量元数据，只在内网发布。

## 7. 隐私与审计

- landing 含销售人名、终端客户明细（KB P10）：`hankel_qa` 在 landing 上按列受限；marts 只出聚合；tests/qa view 不得把人名明细持久化进宽可读 schema。
- 审计面：PG 对 `hankel_dbt` 开 DDL 日志；dbt artifacts（manifest/run_results）+ git 历史 = 每一条执行过的 SQL 可回溯；决策链接（based_on_decisions）同时回答"这个 filter 为什么存在"。

## 8. M1/M3 动手前必须做的三件事

1. 建角色矩阵（§3 SQL），**禁止 dbt 复用 `SPRING_DATASOURCE_*` 凭据**。
2. 轮换已进过 shell history 的密码；profiles 走 env var + gitignore。
3. 控制面 schema 与 landing 至少做到 REVOKE 级隔离（中长期可迁独立实例）。
