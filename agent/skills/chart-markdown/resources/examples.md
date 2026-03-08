# Chart markdown examples

Additional examples for line, scatter, funnel, and gauge. Replace sample data with real query or analysis results.

## Line chart (time trend)

```chart
{
  "table": [["Month", "Revenue"], ["Jan", 120000], ["Feb", 135000], ["Mar", 142000]],
  "echarts": {
    "title": {"text": "Quarterly revenue trend"},
    "tooltip": {"trigger": "axis"},
    "legend": {"data": ["Revenue"]},
    "xAxis": {"type": "category", "data": ["Jan", "Feb", "Mar"], "name": "Month"},
    "yAxis": {"type": "value", "name": "Revenue"},
    "series": [{"name": "Revenue", "type": "line", "data": [120000, 135000, 142000], "smooth": true}]
  }
}
```

## Scatter chart (correlation)

```chart
{
  "table": [["Sales", "Profit"], [120, 35], [150, 42], [200, 55], [250, 68]],
  "echarts": {
    "title": {"text": "Sales vs profit"},
    "tooltip": {"trigger": "item", "formatter": "Sales: {c0}<br/>Profit: {c1}"},
    "xAxis": {"type": "value", "name": "Sales"},
    "yAxis": {"type": "value", "name": "Profit"},
    "series": [{"type": "scatter", "symbolSize": 12, "data": [[120, 35], [150, 42], [200, 55], [250, 68]]}]
  }
}
```

## Funnel chart (conversion stages)

```chart
{
  "table": [["Stage", "Count"], ["Visit", 5000], ["Register", 2000], ["Activate", 1500], ["Pay", 600]],
  "echarts": {
    "title": {"text": "Conversion funnel"},
    "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
    "series": [{
      "type": "funnel",
      "left": "10%",
      "width": "80%",
      "sort": "descending",
      "gap": 4,
      "label": {"show": true, "position": "inside", "formatter": "{b}\n{c}"},
      "data": [
        {"value": 5000, "name": "Visit"},
        {"value": 2000, "name": "Register"},
        {"value": 1500, "name": "Activate"},
        {"value": 600, "name": "Pay"}
      ]
    }]
  }
}
```

## Gauge (single KPI)

```chart
{
  "table": [["Metric", "Value", "Target"], ["Sales", 850, 1000]],
  "echarts": {
    "title": {"text": "Sales target completion"},
    "series": [{
      "type": "gauge",
      "center": ["50%", "60%"],
      "radius": "90%",
      "axisLine": {"lineStyle": [{"color": [[0.3, "#91c7ae"], [0.7, "#63869e"], [1, "#c23531"]]}]},
      "pointer": {"itemStyle": {"color": "auto"}},
      "detail": {"formatter": "{value}%", "fontSize": 24, "offsetCenter": [0, "70%"]},
      "data": [{"value": 85, "name": "Completion"}]
    }]
  }
}
```
