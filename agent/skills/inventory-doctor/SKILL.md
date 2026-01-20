---
name: inventory-doctor
description: 针对 WMS 拣货缺货场景的库存异常诊断 SOP，涵盖触发、信息收集、决策分支（自动修复/物理复核）与终态汇报。
---

## 场景

- 商品 A 系统显示 5 件，拣货员现场为 0（缺货）。
- 智能体以 Think-Act-Observe 循环完成诊断与处置。

## 核心流程（TAO）

### 1) Trigger（感知与锁定）
- 事件：拣货员上报 Pick Shortage。
- 动作：锁定相关 SKU + 库位，冻结并发变更。

### 2) Data Retrieval & Triage（信息收集与分诊）
- 查路向：拉取未完结的上架/移库/波次任务，判断是否在途或延迟。
- 查历史：检索近 24 小时库位操作日志，关注未确认的移库、撤销、盘点。
- 形成初步假设：数据延迟 / 中间态卡死 / 实物错位。

### 3) Reasoning & Execution（决策执行）
- 分支 A 自动修复：接口中间态→触发 retry_sync 或状态刷新。
- 分支 B 需物理验证：生成盘点任务，派发就近理货员检查可能错放库位。
- 所有动作记录审计日志，保持可追溯。

### 4) Reporting（终态汇报）
- 输出诊断结论、采取的动作、残留风险与建议（培训/流程改进）。

## 推荐工具契约（可 Mock）

> 工具可返回随机/固定示例数据即可，无需真实后端。

- `get_wms_movement_tasks(skuId, locationId)` → { tasks: [...], pending: boolean }
- `get_location_logs(locationId, lookbackHours)` → [{ ts, action, operator, status }]
- `retry_sync(taskId)` → { fixed: boolean, message }
- `dispatch_cycle_count({ skuId, locations, priority })` → { taskId, assignee, eta }
- `notify_picker(message)` → { delivered: boolean }
- `write_case_report(markdown)` → { saved: true, path }

## 交付物模板

```markdown
### 诊断概览
- 异常：Pick Shortage / SKU: xxx / 库位: xxx
- 初步结论：数据中间态 / 物理错放 / 待验证

### 关键发现
- 路向检查：...
- 日志发现：...
- 其他迹象：...

### 处置动作
- 自动修复：已调用 retry_sync(taskId=...) / 结果 ...
- 物理验证：已派发盘点任务 {taskId, 库位 B/C, ETA}
- 通知：已告知拣货员重试/等待/改捡

### 后续建议
- 培训/流程：...
- 监控/预警：...
```
