# BO 对象删除模式（soft/hard）设计

- 日期：2026-09-22
- 状态：Draft（设计已定稿，待实现）
- 范围：`platform-service/src/main/java/com/fina/platform/bo/`（DDL、删除逻辑、错误映射）+ `agent/src/agents/platform_service/business_objects/`（工具透传、builder 技能/提示词）+ 两侧测试
- 不改动：BO store/grant 管理、`delete_object`（对象定义）语义、网关 `el-ai-gateway`

## 1. 背景与目标

现状问题：BO 记录删除是**软删除**（`deleted = 1`），但对象定义里的 unique 索引是**无条件**唯一索引（如 `customer_tag` 的 `(customer_no, tag_key)`）。软删行仍占用唯一键，导致：

- 删除某标签后**无法再加回同名 key**（`create_record` 报 `500 duplicate key`）。
- 方案"先删后插"、请求重试都会踩此坑。
- 唯一冲突暴露为裸 `500 INTERNAL_ERROR`，不是可处理的 4xx。

本设计引入**按对象选择的删除模式** `soft` / `hard`，默认 `hard`；并统一修复软删 + 唯一索引的冲突。

目标：

- 建对象时可选 `deleteMode`，**不填即 `hard`（物理删除）**。
- 软删对象也能"删后再加同名 key"：unique 索引一律改为**部分唯一索引** `WHERE deleted = 0`。
- 现有 5 个对象全部重建（数据为空），统一采用新语义。

## 2. 决策总览

| 项 | 决策 |
|---|---|
| 删除模式 | 按对象配置，`soft` / `hard` |
| 默认值 | `hard`（不填即物理删除） |
| 存储位置 | 对象定义 schema JSON（`bo_object_definitions`），不加列、免迁移 |
| 建表 DDL | 不变，仍建 `deleted INT NOT NULL DEFAULT 0`（两模式统一，读取侧不分支） |
| unique 索引 | 一律 `CREATE UNIQUE INDEX ... WHERE "deleted" = 0` |
| `update_object` 改模式 | 拒绝（`400`），要改就重建对象 |
| 唯一冲突 | 映射为 `409 CONFLICT` |
| 存量对象 | 全部删除并重建（数据为空） |
| 建表交互 | builder agent 主动询问"物理删除还是软删除？（默认物理删除）" |

## 3. 数据模型

- `deleteMode` 加入对象定义的 schema JSON，取值 `"hard" | "soft"`，解析时缺省 `"hard"`（对旧定义向后兼容为 hard）。
- `createTableSql` 不变：仍含 `"deleted" INT NOT NULL DEFAULT 0`。hard 模式下 `deleted` 恒为 0，查询侧 `deleted = 0` 过滤恒真。
- `createIndexSql`：所有 `unique: true` 的索引追加 `WHERE "deleted" = 0`；非唯一索引不变。

## 4. API / 工具

- `POST /api/v1/bo/objects`：新增可选 `deleteMode`（`soft | hard`，默认 `hard`），随 schema 持久化。
- `get_object` / `list_objects`：返回 `deleteMode`。
- `PUT /api/v1/bo/objects/{objectKey}`（`update_object`）：若请求的 `deleteMode` 与现有不同，返回 `400`（提示需重建对象）。
- MCP `create_object` 透传 `deleteMode`；`business-objects-modeling` 技能与 builder 提示词要求 agent 在建表前询问用户"物理删除还是软删除？（默认物理删除）"，用户未明确时用 `hard`。

## 5. 删除行为

`deleteRecord` / `deleteRecords` 读取 `definition.deleteMode`：

- `hard` → `DELETE FROM <table> WHERE id = ?`（事务批量时为循环删除；删除不存在的 id 计 0，天然幂等）。
- `soft` → 现状：`UPDATE <table> SET deleted = 1, updated_at = now() WHERE id = ? AND deleted = 0`。

读取/查询侧不变（`buildQuery` 始终 `deleted = 0` 起手）。`delete_records` 保持原子、幂等语义。

## 6. 错误处理

- platform-service 捕获唯一约束冲突（`DuplicateKeyException` / SQL state `23505`），映射为 `409 CONFLICT`，携带可读信息。
- 其余删除/创建错误维持现有映射。

## 7. 重建与迁移

- 现有 5 个对象表均为空。删除各对象物理表与对象定义（元数据硬删），再用 `create_object` 重新创建，按需选 `soft` / `hard`。
- 新表直接使用部分唯一索引，无需单独的索引迁移脚本。
- 存量软删行随表删除一并消失。

## 8. 测试

platform-service：

- hard 对象 `delete_record` 物理删除（查询行消失）。
- soft 对象 `delete_record` 置 `deleted = 1`（查询不可见）。
- **soft 对象：软删后以同名唯一键重新 create 成功**（验证部分唯一索引）。
- 唯一冲突返回 `409`（不再 500）。
- 未指定 `deleteMode` 时默认 `hard`；`get_object`/`list_objects` 回显 `deleteMode`。
- `update_object` 试图修改 `deleteMode` 返回 `400`。

agent 插件：

- `create_object` 执行器透传 `deleteMode`。
- `business-objects-modeling` 技能/提示词包含"询问用户删除模式、默认物理删除"文案。
- `get_object`/`list_objects` 输出含 `deleteMode`。

## 9. 不做（YAGNI）

- 建对象后修改 `deleteMode`。
- 按单次调用指定删除模式。
- `delete_object` 时 `DROP TABLE` 物理表（仍保持对象定义软删）。
- 外键 / 级联删除。
