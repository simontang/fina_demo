# Formatting conventions

- **Currency**: Use locale format (e.g. 1,234.56 or 1 234,56); indicate unit in axis title (e.g. "Revenue (CNY)").
- **Percent**: Show as "X%" in labels/tooltip; ensure sum = 100% for pie if showing share.
- **Time**: Same granularity and format across the chart (e.g. "Jan 2024" or "2024-Q1"); use `xAxis.type: "time"` for continuous.
- **Counts**: Prefer integers in labels; use K/M suffix for large numbers (e.g. "1.2M") if needed.
- **Ranking**: Sort by value (descending unless natural order e.g. funnel); limit to top N (e.g. 10) for clarity.

## When in doubt

1. Match the chart to the **primary question** (trend → line, compare → bar, share → pie, relation → scatter).
2. Prefer **fewer series and categories**; aggregate or "Other" instead of cluttering.
3. Use **consistent terminology** with the domain (e.g. "Revenue" vs "Sales" as in the data source).
4. Add a **short, descriptive title** and axis/label names that a business user would recognize.
