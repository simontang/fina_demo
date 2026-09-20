import type { PluginSkillDefinition } from "@axiom-lattice/protocols";

const CONTENT = `---
name: business-objects-modeling
description: Builder policy for modeling Business Object stores, object definitions, fields and indexes; covers naming, store selection, field types, indexes, v1 evolution limits, grants, confirmations, and the standard build/verify workflow.
---

# Business Objects Modeling

You are modeling business data as Business Objects. A store is one PostgreSQL database plus schema; an object definition maps an objectKey to a physical table; records are rows. This policy is mandatory for every build action.

## 1. 概念模型
- **Store**：一个 store 对应一个 PostgreSQL 库 + schema（默认 public）。创建 store 时平台自动写入一条 granteeKey=tenant、can_read/write/manage=true 的默认 grant。
- **Store grant**：granteeKey + can_read / can_write / can_manage，决定该连接键能读/写/管理哪些 store。
- **Object 定义**：objectKey → 物理表，含 fields 与 indexes；由 platform-service 同步生成 DDL。
- **Record**：object 下的一行数据，按 id 读写；删除为软删。

## 2. 命名规范
- storeKey / objectKey / field.key / index.name 一律匹配 ^[a-z][a-z0-9_]{0,62}$。
- 使用业务单数英文名词：customer、order、invoice。
- 避免 SQL 保留字；不要用大写、连字符、前导数字。

## 3. store 选择与创建
- 先 list_stores，再用 test_store 确认连通。
- 无合适 store 才 create_store：仅接受 jdbc:postgresql: URL；默认 schema public。
- 同一业务域复用同一 store，不要为每个 object 新建库。
- 创建后 list_store_grants 确认默认 tenant grant 存在。

## 4. 字段类型映射
- string：短文本，配 maxLength（常见 ≤255）。
- text：长文本，无长度上限。
- integer / long：整数；大计数用 long。
- decimal：金额/比率，必须给 precision 与 scale。
- boolean：真假。
- date / datetime：日期 / 时间戳。
- json：结构不固定的扩展字段；能用列表达就不要用 json。
- required：业务上必填的字段设为 true。
- 每个 field 建议写 description，说明业务含义。

## 5. 索引设计
- indexes[].fields 顺序即索引列顺序，把等值过滤列放前面。
- unique: true 用于业务唯一键。
- 高频过滤/排序列建索引；低基数或写多读少的列不要建。
- index.name 可省略，由服务端派生。

## 6. v1 演进约束
- update_object 只支持增量加列，**不支持删列或改类型**。
- 需要删列/改类型时：新建 object，或加新列后由数据侧迁移，不原地改。
- 加列用 update_object，fields 传完整定义。

## 7. 权限与 grant
- 默认 tenant grant 覆盖读写管理；跨 grantee 授权用 grant_store。
- can_manage 允许改定义；can_write 允许写记录；can_read 只读。
- 收窄权限用 grant_store 覆盖，不要新建 store。

## 8. 安全与确认
- delete_object / delete_record 必须先取得用户明确确认，再传 confirm: true。
- 删除为软删，但不得在未确认时执行。
- 不要把生产数据当测试数据；测试记录用完删除。

## 9. 标准工作流
1. 澄清业务实体与关键字段；不确定时用 ask_user_to_clarify。
2. list_stores / test_store 选定 store。
3. list_objects / get_object 检查是否已存在。
4. create_object（或 update_object 加列）。
5. get_object 复核定义。
6. 验证：create_record 造样本 → query_records / get_record 核对 → delete_record 清理。
7. 用 task 记录进度与最终定义。

## 10. 验收清单
- [ ] 命名全部符合 ^[a-z][a-z0-9_]{0,62}$。
- [ ] 字段类型/长度/精度合理，required 正确。
- [ ] 索引覆盖主要查询，无过度索引。
- [ ] get_object 结果与预期定义一致。
- [ ] 测试记录已清理。
- [ ] task 描述记录了最终 object 定义。`;

export const BUSINESS_OBJECTS_MODELING_SKILL: PluginSkillDefinition = {
  version: "1.0.0",
  content: CONTENT,
};
