---
name: dashboard-generation
description: Generate comprehensive data visualization dashboards with multiple charts, executive summaries, and key insights. Use when creating multi-chart dashboards, executive overview reports, or business intelligence visualizations that require combining trend analysis, comparisons, and KPI monitoring in a single view.
metadata:
  category: global
---

## When to Use This Skill

Activate this skill when the user needs:
- **Multi-chart dashboards**: Combining 4-8 related charts in a single view
- **Executive summaries**: High-level KPI overview with visual context
- **Business intelligence**: Comprehensive data views for decision making
- **Trend + comparison + distribution**: Multiple analysis types in one report

**Do NOT use for:**
- Single chart generation (use chart-markdown skill instead)
- Detailed written reports (use report-writer skill instead)
- Simple data tables without visualization

---

## Dashboard Types

### Executive Dashboard
**Purpose**: High-level business overview for leadership
- 4-6 KPI cards with trends
- 2-3 trend charts for main metrics
- 1-2 comparison charts
- Key findings summary

### Operational Dashboard
**Purpose**: Day-to-day monitoring for operations teams
- Real-time or near-real-time metrics
- Status indicators and progress tracking
- Detailed breakdowns by operational dimensions
- Alert thresholds and anomalies

### Analytical Dashboard
**Purpose**: Deep-dive analysis for data analysts
- Multiple trend charts with different granularities
- Correlation and distribution charts
- Segment analysis
- Drill-down capabilities

---

## Dashboard Structure

### Required Sections

1. **Executive Summary**
   - KPI overview table (4-6 key metrics)
   - Critical findings (3-5 bullet points)
   - Status indicators (✅ 🟢 🟡 🔴)

2. **Chart Sections** (2-4 sections)
   - Each section contains 1-3 related charts
   - Group by theme (e.g., "Performance Overview", "Regional Analysis")
   - Charts must include brief insights

3. **Key Findings Summary**
   - Major wins
   - Areas of concern
   - Recommended actions

4. **Data Sources**
   - Reference to input files
   - Last updated timestamp

---

## Chart Generation

**Important**: For detailed chart generation guidelines, syntax, and best practices, reference the `chart-markdown` skill.

### Quick Reference

| Purpose | Chart Type | Skill Reference |
|---------|-----------|----------------|
| Time trends | Line chart | `chart-markdown` - Line Charts |
| Category comparison | Bar chart | `chart-markdown` - Bar Charts |
| Composition | Pie/Donut | `chart-markdown` - Composition Charts |
| Distribution | Histogram | `chart-markdown` - Distribution Charts |
| Correlation | Scatter plot | `chart-markdown` - Correlation Charts |
| Progress tracking | Progress bar | `chart-markdown` - Progress Indicators |

### Chart Guidelines
- Use `chart-markdown` skill for all chart syntax and formatting rules
- Maximum 8 charts per dashboard
- Group related charts into 2-4 sections
- Prioritize important metrics at the top

---

## Output Format

### File Path
`/artifacts/dashboard-{topic}.md`

### Structure Template

```markdown
# [Dashboard Title]

**Dashboard Type:** [Executive/Operational/Analytical]
**Generated Date:** [Date]
**Data Period:** [Start] to [End]
**Data Source:** `/tmp/data-{topic}.md`

---

## Executive Summary

### Key Metrics Overview

| Metric | Current | Change | Status | Trend |
|--------|---------|--------|--------|-------|
| [Metric 1] | [Value] | [Change] | [Status] | [Trend] |
| [Metric 2] | [Value] | [Change] | [Status] | [Trend] |

### Critical Findings

1. **[Finding 1]**: [Description with data]
2. **[Finding 2]**: [Description with data]
3. **[Finding 3]**: [Description with data]

---

## Section 1: [Theme Name]

### Chart 1.1: [Chart Title]

```chart
{
  "table": [["Category", "Value"], ["A", 10], ["B", 20]],
  "echarts": {
    "title": {"text": "Chart Title"},
    "tooltip": {"trigger": "axis"},
    "xAxis": {"type": "category", "data": ["A", "B"]},
    "yAxis": {"type": "value"},
    "series": [{"type": "bar", "data": [10, 20]}]
  }
}
```

**Key Insights:**
- [Insight 1]
- [Insight 2]

---

## Key Findings Summary

### Major Wins
- [Win 1]
- [Win 2]

### Areas of Concern
- [Concern 1]
- [Concern 2]

### Recommended Actions
1. **[Priority]**: [Action]
2. **[Priority]**: [Action]

---

## Data Sources

- **Primary:** `/tmp/data-{topic}.md`
- **Analysis:** `/tmp/insight-{topic}.md`
- **Last Updated:** [Timestamp]
```

---

## Best Practices

### Chart Guidelines
- Reference `chart-markdown` skill for all chart generation
- **Maximum 8 charts** per dashboard
- **Group related charts** into 2-4 sections
- **Prioritize important metrics** at the top
- Use consistent colors across related charts

### Data Formatting
- Right-align numbers in tables
- Use thousands separators for large numbers (12,500.00)
- Include units in headers (Revenue ($), Orders (#))
- Use emojis for quick visual cues (📈 📉 ✅ ⚠️)

### Executive Summary
- Lead with KPI table (easy to scan)
- Limit to 3-5 critical findings
- Include both positive and negative insights
- Make findings actionable

---

## Examples

### Example 1: Sales Performance Dashboard

```markdown
# Sales Performance Dashboard

**Dashboard Type:** Executive
**Generated Date:** 2024-01-15
**Data Period:** 2024-07-01 to 2024-12-31
**Data Source:** `/tmp/data-sales-performance.md`

---

## Executive Summary

### Key Metrics Overview

| Metric | Current | Change | Status | Trend |
|--------|---------|--------|--------|-------|
| **Revenue** | $600K | +5% | ✅ On Track | 📈 Up |
| **Orders** | 1,250 | +4.3% | ✅ On Track | 📈 Up |
| **Conversion** | 3.2% | -0.5% | ⚠️ Attention | 📉 Down |
| **AOV** | $480 | +1.5% | ✅ On Track | ➡️ Stable |

### Critical Findings

1. **📈 Revenue Growth**: Revenue increased 5% driven by North region performance
2. **⚠️ Conversion Decline**: Mobile checkout issues causing 0.5% drop
3. **🎯 Regional Gap**: East region declining 5% while others grow

---

## Section 1: Revenue Analysis

### Chart 1.1: Monthly Revenue Trend

```chart
{
  "table": [["Month", "Revenue"], ["Jul", 100], ["Aug", 105], ["Sep", 98], ["Oct", 110], ["Nov", 85], ["Dec", 82]],
  "echarts": {
    "title": {"text": "Monthly Revenue Trend ($000s)"},
    "tooltip": {"trigger": "axis"},
    "xAxis": {"type": "category", "data": ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]},
    "yAxis": {"type": "value", "name": "Revenue ($000s)"},
    "series": [{"type": "line", "data": [100, 105, 98, 110, 85, 82], "smooth": true}]
  }
}
```

**Key Insights:**
- Peak in October at $110K
- Decline in Nov-Dec requires investigation
- Q4 overall down 15% vs Q3

### Chart 1.2: Revenue by Region

| Region | Revenue | Share | Growth | Status |
|--------|---------|-------|--------|--------|
| North | $270K | 45% | +15% | 🟢 Leading |
| South | $150K | 25% | +3% | 🟡 Stable |
| East | $120K | 20% | -5% | 🔴 Declining |
| West | $60K | 10% | +8% | 🟢 Growing |

---

## Key Findings Summary

### Major Wins
- North region outperformed with 15% growth
- West region showing promising 8% growth
- Overall revenue still up 5% YoY

### Areas of Concern
- East region declining 5%
- Nov-Dec drop of 25% from October peak
- Conversion rate down 0.5%

### Recommended Actions
1. **🔴 High**: Investigate East region decline causes
2. **🟡 Medium**: Fix mobile checkout conversion issues
3. **🟢 Low**: Expand successful North region strategies
```

---

## Workflow

1. **Analyze Requirements**
   - Identify dashboard type (Executive/Operational/Analytical)
   - Determine key metrics and KPIs
   - Define time period and data scope

2. **Read Input Files**
   - Load `/tmp/data-{topic}.md`
   - Load `/tmp/insight-{topic}.md` if available
   - Identify available metrics and dimensions

3. **Design Structure**
   - Select 4-8 charts based on data available
   - Group into 2-4 logical sections
   - Plan executive summary content

4. **Generate Charts**
   - Reference `chart-markdown` skill for chart syntax and best practices
   - Create trend charts for time-series data
   - Create bar charts for category comparisons
   - Create tables for detailed comparisons
   - Ensure all charts have insights

5. **Write Executive Summary**
   - Extract key metrics
   - Identify critical findings
   - Add status indicators

6. **Output Dashboard**
   - Write to `/artifacts/dashboard-{topic}.md`
   - Verify all sections complete
   - Check formatting consistency

---

## Common Patterns

### KPI Table Format
```markdown
| Metric | Current | Change | Status | Trend |
|--------|---------|--------|--------|-------|
| **Revenue** | $125K | +5.9% | ✅ On Track | 📈 |
| **Orders** | 1,250 | +4.3% | ✅ On Track | 📈 |
| **Conversion** | 3.2% | -0.5% | ⚠️ Attention | 📉 |
```

### Chart with Insights
```markdown
### Chart: Monthly Trend

```chart
{
  "table": [["Month", "Value"], ["Jan", 100], ["Feb", 120]],
  "echarts": {
    "title": {"text": "Monthly Trend"},
    "tooltip": {"trigger": "axis"},
    "xAxis": {"type": "category", "data": ["Jan", "Feb"]},
    "yAxis": {"type": "value"},
    "series": [{"type": "line", "data": [100, 120]}]
  }
}
```

**Key Insights:**
- [Specific insight with data point]
- [Comparison to previous period]
- [Notable anomaly or pattern]
```

### Finding Format
```markdown
1. **[Emoji] [Title]**: [Specific finding with numbers]
   - [Supporting detail]
   - [Impact assessment]
```