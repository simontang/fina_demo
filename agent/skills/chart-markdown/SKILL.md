---
name: chart-markdown
description: Selects an appropriate chart type (bar, line, pie, scatter, heatmap, funnel, gauge, radar) from data analysis or query results and generates a chart markdown block containing JSON for external chart libraries. Use when the user asks to visualize query results, analysis data, or to create bar charts, line charts, pie charts, scatter plots, or other chart types from tabular or aggregated data.
---

# Chart Markdown Block

## Chart type selection

Choose by data shape and question:

- **Bar** (bar): Compare categories or time periods. Use category xAxis, value yAxis. Multiple series for grouped or stacked bars.
- **Line** (line): Show trends over time. Use category or time xAxis, value yAxis. Multiple series for multiple metrics.
- **Pie** (pie): Show composition or share. No axes. Data: `[{value: number, name: string}, ...]`. Use `radius: ["40%", "70%"]` for donut.
- **Scatter** (scatter): Correlation or distribution. Use value xAxis and value yAxis. Data: `[[x, y], [x, y], ...]`.
- **Heatmap** (heatmap): Two dimensions + value. Category xAxis and yAxis. Data: `[[xIndex, yIndex, value], ...]`.
- **Funnel** (funnel): Sequential stages or conversion. Data: `[{value, name}, ...]`. Use `sort: "ascending"` or `"descending"`.
- **Gauge** (gauge): Single KPI or progress. One series with `data: [{value, name}]`.
- **Radar** (radar): Multi-dimensional comparison. Define `radar.indicator` and series with `value` arrays.

## Output format

Emit a single fenced code block with language **chart**. Body is one JSON object:

- **table**: Array of rows (original or summary data) for reference.
- **echarts**: ECharts option object (title, tooltip, legend, xAxis, yAxis, series, etc.).

```chart
{
  "table": [["Category", "Value"], ["A", 10], ["B", 20]],
  "echarts": { ... }
}
```

## Schema requirements

- **title**: `{"text": "Clear chart title"}`.
- **tooltip**: Use `trigger: "axis"` for bar/line; `trigger: "item"` for pie/scatter/funnel.
- **xAxis / yAxis**: Omit for pie/funnel/gauge. Use `type: "category"` or `"time"` or `"value"`; provide `data` when type is category.
- **series**: One or more items. Each has `type`, `name`, `data`. Pie data: `[{value, name}]`. Scatter data: `[[x,y], ...]`. Bar/line: array of values or category-mapped values.
- **legend**: Include when there are multiple series.

## Best practices

- Use clear, business-facing titles and axis labels (not raw field names).
- Format numbers (percent, currency, thousands) in tooltip and axis labels.
- For time series, use `xAxis.type: "time"` and consistent date format.
- Show important values on the chart with `series.label` when useful.
- For industry- or data-type-specific guidance, see [best-practices/README.md](best-practices/README.md) (data patterns, formatting, and one file per industry: sales-retail, finance, marketing-growth, operations-supply-chain, hr-people).

## Examples

### Bar chart (category comparison)

```chart
{
  "table": [["Product", "Sales"], ["Product A", 320], ["Product B", 280], ["Product C", 250]],
  "echarts": {
    "title": {"text": "Top products by sales"},
    "tooltip": {"trigger": "axis"},
    "xAxis": {"type": "category", "data": ["Product A", "Product B", "Product C"], "name": "Product"},
    "yAxis": {"type": "value", "name": "Sales"},
    "series": [{"type": "bar", "data": [320, 280, 250], "label": {"show": true, "position": "top"}}]
  }
}
```

### Pie chart (composition)

```chart
{
  "table": [["Channel", "Count"], ["Online", 450], ["Offline", 320], ["Partner", 180]],
  "echarts": {
    "title": {"text": "Customer acquisition by channel"},
    "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
    "legend": {"orient": "vertical", "right": "10%", "top": "center"},
    "series": [{
      "type": "pie",
      "radius": ["40%", "70%"],
      "data": [
        {"value": 450, "name": "Online"},
        {"value": 320, "name": "Offline"},
        {"value": 180, "name": "Partner"}
      ],
      "label": {"formatter": "{b}\n{d}%"}
    }]
  }
}
```

For more chart types (line, scatter, funnel, gauge, radar), see [examples.md](examples.md).
