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

\`\`\`chart
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

## 常用图表用例

> **说明**：以下用例中的数据均为 Mock 示例数据，实际使用时需替换为真实业务数据。

### 1. 月度销售趋势（折线图）
```chart
{
  "table": [["月份", "销售额"], ["1月", 120000], ["2月", 135000], ["3月", 142000]],
  "echarts": {
    "title": {"text": "2024年Q1月度销售额趋势"},
    "tooltip": {"trigger": "axis"},
    "legend": {"data": ["销售额"]},
    "xAxis": {"type": "category", "data": ["1月", "2月", "3月"], "name": "月份"},
    "yAxis": {"type": "value", "name": "销售额(元)", "axisLabel": {"formatter": "{value}"}},
    "series": [{"name": "销售额", "type": "line", "data": [120000, 135000, 142000], "smooth": true}]
  }
}
```

### 2. 产品销量排名（柱状图）
```chart
{
  "table": [["产品", "销量"], ["产品A", 320], ["产品B", 280], ["产品C", 250], ["产品D", 210], ["产品E", 180]],
  "echarts": {
    "title": {"text": "产品销量TOP5排名"},
    "tooltip": {"trigger": "axis"},
    "xAxis": {"type": "category", "data": ["产品A", "产品B", "产品C", "产品D", "产品E"], "name": "产品"},
    "yAxis": {"type": "value", "name": "销量"},
    "series": [{"type": "bar", "data": [320, 280, 250, 210, 180], "label": {"show": true, "position": "top"}}]
  }
}
```

### 3. 客户渠道占比（环形饼图）
```chart
{
  "table": [["渠道", "客户数"], ["线上", 450], ["线下", 320], ["渠道商", 180], ["转介绍", 120]],
  "echarts": {
    "title": {"text": "客户来源渠道分布"},
    "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
    "legend": {"orient": "vertical", "right": "10%", "top": "center"},
    "series": [{
      "type": "pie",
      "radius": ["40%", "70%"],
      "center": ["40%", "50%"],
      "data": [
        {"value": 450, "name": "线上"},
        {"value": 320, "name": "线下"},
        {"value": 180, "name": "渠道商"},
        {"value": 120, "name": "转介绍"}
      ],
      "label": {"formatter": "{b}\n{d}%"}
    }]
  }
}
```

### 4. 销售额与利润相关性（散点图）
```chart
{
  "table": [["销售额", "利润"], [120, 35], [150, 42], [180, 48], [200, 55], [220, 60], [250, 68], [280, 72], [310, 80]],
  "echarts": {
    "title": {"text": "销售额与利润相关性分析"},
    "tooltip": {"trigger": "item", "formatter": "销售额: {c0}万\n利润: {c1}万"},
    "xAxis": {"type": "value", "name": "销售额(万)"},
    "yAxis": {"type": "value", "name": "利润(万)"},
    "series": [{"type": "scatter", "symbolSize": 15, "data": [[120, 35], [150, 42], [180, 48], [200, 55], [220, 60], [250, 68], [280, 72], [310, 80]]}]
  }
}
```

### 5. 转化漏斗分析（漏斗图）
```chart
{
  "table": [["阶段", "人数"], ["访问", 5000], ["注册", 2000], ["激活", 1500], ["付费", 600], ["复购", 240]],
  "echarts": {
    "title": {"text": "用户转化漏斗"},
    "tooltip": {"trigger": "item", "formatter": "{b}: {c}人 ({d}%)"},
    "series": [{
      "type": "funnel",
      "left": "10%",
      "width": "80%",
      "sort": "ascending",
      "gap": 4,
      "label": {"show": true, "position": "inside", "formatter": "{b}\n{c}人"},
      "data": [
        {"value": 5000, "name": "访问"},
        {"value": 2000, "name": "注册"},
        {"value": 1500, "name": "激活"},
        {"value": 600, "name": "付费"},
        {"value": 240, "name": "复购"}
      ]
    }]
  }
}
```

### 6. KPI完成率仪表盘
```chart
{
  "table": [["指标", "当前值", "目标值"], ["销售目标", 850, 1000]],
  "echarts": {
    "title": {"text": "月度销售目标完成率"},
    "series": [{
      "type": "gauge",
      "center": ["50%", "60%"],
      "radius": "90%",
      "axisLine": {"lineStyle": [{"color": [[0.3, "#91c7ae"], [0.7, "#63869e"], [1, "#c23531"]]}]},
      "pointer": {"itemStyle": {"color": "auto"}},
      "detail": {"formatter": "{value}%", "fontSize": 24, "offsetCenter": [0, "70%"]},
      "data": [{"value": 85, "name": "完成率"}]
    }]
  }
}
```

### 7. 地区销售热力图
```chart
{
  "table": [["地区", "销售", "利润"], ["华东", 1200, 360], ["华南", 980, 294], ["华北", 850, 255], ["西部", 620, 186], ["东北", 380, 114]],
  "echarts": {
    "title": {"text": "各区域销售业绩分布"},
    "tooltip": {"position": "top"},
    "visualMap": {"min": 300, "max": 1200, "calculable": true, "orient": "horizontal", "left": "center", "bottom": "5%"},
    "xAxis": {"type": "category", "data": ["华东", "华南", "华北", "西部", "东北"], "splitArea": {"show": true}},
    "yAxis": {"type": "category", "data": ["销售"], "splitArea": {"show": true}},
    "series": [{
      "type": "heatmap",
      "data": [[0, 0, 1200], [1, 0, 980], [2, 0, 850], [3, 0, 620], [4, 0, 380]],
      "label": {"show": true, "formatter": "{c}万"}
    }]
  }
}
```

### 8. 多指标对比（雷达图）
```chart
{
  "table": [["指标", "A产品", "B产品"], ["销量", 85, 70], ["利润", 75, 90], ["评分", 80, 85], ["复购率", 60, 75], ["市场份额", 65, 55]],
  "echarts": {
    "title": {"text": "A/B产品多维度对比"},
    "tooltip": {},
    "legend": {"data": ["A产品", "B产品"], "bottom": 0},
    "radar": {
      "indicator": [
        {"name": "销量", "max": 100},
        {"name": "利润", "max": 100},
        {"name": "评分", "max": 100},
        {"name": "复购率", "max": 100},
        {"name": "市场份额", "max": 100}
      ]
    },
    "series": [{
      "type": "radar",
      "data": [
        {"value": [85, 75, 80, 60, 65], "name": "A产品"},
        {"value": [70, 90, 85, 75, 55], "name": "B产品"}
      ]
    }]
  }
}
```