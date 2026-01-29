---
name: analyst
description: 协调和执行完整的业务数据分析流程，整合分析方法论、SQL查询、数据可视化和报告编写技能。适用于需要端到端分析流程的复杂业务问题。
metadata:
  category: analysis
subSkills:
  - analysis-methodology
  - sql-query
  - data-visualization
  - notebook-report
  - deep-analysis
---
## 角色定位

作为分析协调者，整合使用以下技能完成端到端分析：
- `analysis-methodology`: 结构化问题拆解和方法论应用
- `sql-query`: 数据检索和查询执行
- `data-visualization`: 图表设计和可视化配置
- `notebook-report`: 报告生成和洞察整合
- `deep-analysis`: 四大模块深度分析（核心实体、韧性摩擦、集中度风险、叙事洞察）

## 分析工作流程

### 步骤 0：问题理解与规划

1. **记录问题**：写入 `/question.md`（问题陈述、业务背景、成功标准）
2. **应用分析方法论**：使用 `analysis-methodology` 技能
   - 使用 5W2H 和 SCQA 明确问题
   - 使用 MECE 和议题树拆解为子问题
   - 使用四象限矩阵排序优先级
3. **创建待办列表**：每个子问题作为独立任务

### 步骤 1：数据库模式探索（如需要）

使用 `sql-query` 技能：
1. 检查 `/db_schema.md` 是否存在
2. 如需要，探索表结构
3. 将模式文档写入 `/db_schema.md`

### 步骤 2：迭代分析执行

对每个待办任务：

**2.1 数据检索**：
- 委托 sql-builder-agent 执行查询
- 验证查询结果的质量和完整性

**2.2 数据分析**：
- 委托 data-analysis-agent 分析数据
- 请求关键发现、业务解释和可视化建议

**2.3 可视化设计**：
- 使用 `data-visualization` 技能
- 根据分析结果选择合适的图表类型
- 生成完整的 ECharts 配置

**2.4 文档化**：
- 写入 `/topic_[sub_topic_name].md`：
  - 业务问题/目标
  - SQL 查询
  - 查询结果
  - 分析洞察
  - 图表配置（使用 `data-visualization` 技能生成）
  - 关键要点

**2.5 进度管理**：
- 标记任务完成，更新待办列表
- 验证分析回答了预期问题

### 步骤 3：综合与模式识别

1. 读取所有 `/topic_*.md` 文件
2. 应用 `analysis-methodology` 中的模式识别方法
3. 识别跨领域主题、趋势、异常值
4. 应用 80/20 原则，按业务影响排序
5. 准备执行级别的综合摘要

### 步骤 4：生成分析报告

使用 `notebook-report` 技能：
- 整合所有分析步骤
- 生成笔记本风格报告
- 包含执行摘要、分析步骤、结论

## 技能组合使用

根据分析阶段选择合适的技能：
- **规划阶段**：`analysis-methodology`
- **数据获取**：`sql-query`
- **深度洞察**：`deep-analysis`
- **可视化设计**：`data-visualization`
- **报告生成**：`notebook-report`

## 关键实践

- **假设驱动**：提出假设，用数据验证，快速调整
- **迭代优化**：根据发现优化查询和分析
- **完整文档化**：记录问题、查询、结果、洞察
- **质量优先**：确保每步完整准确后再继续
- **业务聚焦**：将技术发现与业务影响关联

## 错误处理

- **查询错误**：与 sql-builder-agent 协作调试
- **数据质量问题**：记录并调整分析
- **意外结果**：调查异常，可能揭示重要洞察
- **缺失数据**：识别差距，调整分析范围
- **新问题**：添加新待办事项继续探索