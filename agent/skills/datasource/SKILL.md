---
name: datasource
description: DataSource 与科目映射约定。包含账户类型与 GroupMask 的对应关系，供分析 P&L（收入、成本、费用、净利润）时正确筛选科目。
metadata:
  category: datasource
---

# DataSource Skill — Account Mapping

When querying GL/P&L data from the datasource, use the following **Account Mapping** to filter by account type. The underlying schema uses `GroupMask` to classify accounts.

---

## Account Mapping

| Account Type | GroupMask | Filter (SQL / API) |
|--------------|-----------|--------------------|
| **Revenue**  | 4         | `GroupMask = 4`    |
| **COGS**     | 5         | `GroupMask = 5`    |
| **Expenses** | 6         | `GroupMask = 6`    |
| **Net Income** | 4, 5, 6 | `GroupMask IN (4, 5, 6)` |

### Usage

- **Single type**: e.g. revenue only → `WHERE GroupMask = 4`
- **Net Income** (Revenue − COGS − Expenses): use `WHERE GroupMask IN (4, 5, 6)` when aggregating all P&L lines; apply sign conventions (revenue positive, COGS/expenses negative) per your reporting rules.
- In metrics API **filters**, use dimension/field corresponding to `GroupMask` with operator `IN` and values `[4, 5, 6]` for net income scope, or `EQ` with value `4`/`5`/`6` for a single type.

---

## Reference

- **Revenue**: GroupMask = 4  
- **COGS**: GroupMask = 5  
- **Expenses**: GroupMask = 6  
- **Net Income**: GroupMask IN (4, 5, 6)
