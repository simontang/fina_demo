# Chart choice by data pattern

Use when the analysis question is clear but industry is generic. Pick chart by pattern:

| Pattern | Typical question | Preferred chart | Notes |
|--------|-------------------|-----------------|--------|
| **Trend over time** | How did X change over time? | Line | Use `xAxis.type: "time"` for continuous dates; category for discrete periods (month, quarter). |
| **Category comparison** | Which category is largest? Rank? | Bar | Horizontal bar if many categories or long labels. |
| **Part-to-whole** | What is the share of each part? | Pie / donut | Limit to 5–7 segments; combine small slices into "Other" if needed. |
| **Correlation / distribution** | How do two measures relate? | Scatter | Optional: size or color for a third dimension. |
| **Conversion / funnel** | How many pass each stage? | Funnel | Sort by stage order; show count and % in tooltip. |
| **Single KPI vs target** | Are we on track? | Gauge | One value; clear min/max or target bands. |
| **Two dimensions + value** | How does value vary by two categories? | Heatmap | Use for matrices (e.g. region × product, hour × day). |
| **Multi-dimensional profile** | How do entities compare on several metrics? | Radar | Same scale per dimension; 2–3 series max. |
