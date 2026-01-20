---
name: data-visualization
description: 为数据分析结果选择合适的图表类型并生成 ECharts 配置。适用于需要将查询结果可视化为柱状图、折线图、饼图、散点图等图表的场景。
metadata:
  category: analysis
---
## 图表类型选择指南

根据数据特征和业务问题选择最合适的图表类型：

- **柱状图** (bar): 比较类别或时间周期
  - 使用 category xAxis，value yAxis
  - 多系列用于分组/堆叠柱状图
  
- **折线图** (line): 展示时间趋势
  - 使用 category/time xAxis，value yAxis
  - 多系列展示多个指标
  
- **饼图** (pie): 展示构成/百分比
  - 无需 xAxis/yAxis
  - 数据格式: [{value: number, name: string}, ...]
  - 使用 radius: ["40%", "70%"] 创建环形图
  
- **散点图** (scatter): 相关性分析
  - 使用 value xAxis 和 value yAxis
  - 数据格式: [[x, y], [x, y], ...]
  
- **热力图** (heatmap): 多维数据
  - 需要 category xAxis 和 yAxis
  - 数据格式: [[xIndex, yIndex, value], ...]

## ECharts 配置要求

生成完整的 ECharts 配置，必须包含：

\`\`\`json
{
  "table": [...],  // 原始数据表格
  "echarts": {
    "title": {"text": "清晰的图表标题"},
    "tooltip": {
      "trigger": "axis",  // bar/line 用 "axis", pie/scatter 用 "item"
      "formatter": "..."  // 可选：自定义格式化
    },
    "legend": {...},  // 多系列时必需
    "xAxis": {
      "type": "category",  // 或 "time", "value"
      "name": "X轴名称",
      "data": [...]  // category 类型时必需
    },
    "yAxis": {
      "type": "value",
      "name": "Y轴名称"
    },
    "series": [{
      "type": "bar|line|pie|scatter|heatmap",
      "name": "系列名称",
      "data": [...],
      "label": {...}  // 可选：显示数值
    }],
    "grid": {...}  // 可选：控制边距
  }
}
\`\`\`

## 最佳实践

- 图表标题清晰描述业务问题
- 轴标签使用业务术语，而非技术字段名
- 数值格式化：百分比、货币、千分位
- 时间序列使用 "xAxis.type: 'time'" 并正确格式化日期
- 多系列时使用 legend 区分
- 重要数值在图表上直接标注（series.label）

## 输出格式

提供完整的 chart JSON 配置，可直接用于渲染。