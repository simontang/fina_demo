export type TemplateDomainId = "executive" | "finance" | "sales" | "customer" | "operations";

export type IllustrationKey =
  | "ebr"
  | "anomaly"
  | "gross_margin"
  | "budget"
  | "funnel"
  | "forecast"
  | "churn"
  | "segmentation"
  | "inventory"
  | "fulfillment";

export type TemplateDomain = {
  id: TemplateDomainId;
  name: string;
  accent: string;
};

export type AgentTemplate = {
  id: string;
  domain: TemplateDomainId;
  title: string;
  description: string;
  illustrationKey: IllustrationKey;
};

export const domains: TemplateDomain[] = [
  { id: "executive", name: "Executive", accent: "#2F6BFF" },
  { id: "finance", name: "Finance", accent: "#1E9E63" },
  { id: "sales", name: "Sales", accent: "#F58A1F" },
  { id: "customer", name: "Customer", accent: "#11A7A1" },
  { id: "operations", name: "Operations", accent: "#1593C6" },
];

export const templates: AgentTemplate[] = [
  {
    id: "executive-ebr",
    domain: "executive",
    title: "Executive Business Review (EBR)",
    description:
      "A board-ready monthly operating recap. Summarize core KPIs, highlight over/under performance, and draft the narrative.",
    illustrationKey: "ebr",
  },
  {
    id: "executive-anomaly",
    domain: "executive",
    title: "Business Anomaly Detection",
    description:
      "A weekly scan across revenue, cost, AOV, conversion, and inventory to surface the biggest anomalies by BU/region.",
    illustrationKey: "anomaly",
  },
  {
    id: "finance-gmv",
    domain: "finance",
    title: "Gross Margin Variance (Volume / Price / Cost / Mix)",
    description: "Explain margin changes with a variance waterfall and top SKU/region drivers.",
    illustrationKey: "gross_margin",
  },
  {
    id: "finance-budget",
    domain: "finance",
    title: "Budget Variance Analysis",
    description: "Identify departments over budget, detect abnormal spend, and explain cost-structure shifts.",
    illustrationKey: "budget",
  },
  {
    id: "sales-funnel",
    domain: "sales",
    title: "Sales Funnel Diagnosis",
    description:
      "Pinpoint which stage (Leads -> Opportunities -> Deals) drives conversion decline and which reps/segments differ.",
    illustrationKey: "funnel",
  },
  {
    id: "sales-forecast",
    domain: "sales",
    title: "Revenue Forecast",
    description:
      "Forecast quarterly revenue using history, pipeline, and seasonality; quantify target attainment probability and risks.",
    illustrationKey: "forecast",
  },
  {
    id: "customer-churn",
    domain: "customer",
    title: "Churn Risk Detection",
    description:
      "Detect at-risk customers from declining activity, rising complaints, and reduced purchase frequency; summarize likely reasons.",
    illustrationKey: "churn",
  },
  {
    id: "customer-segmentation",
    domain: "customer",
    title: "Customer Segmentation",
    description:
      "Auto-segment into VIP, High-Potential, Price-Sensitive, and Low-Value; compare frequency, AOV, and LTV.",
    illustrationKey: "segmentation",
  },
  {
    id: "ops-inventory",
    domain: "operations",
    title: "Inventory Health",
    description:
      "Spot slow movers, turnover issues, and stockout risks; summarize replenishment and liquidation signals.",
    illustrationKey: "inventory",
  },
  {
    id: "ops-fulfillment",
    domain: "operations",
    title: "Fulfillment Efficiency",
    description:
      "Analyze order cycle time, shipping speed, and logistics cost; locate delay causes and warehouse/carrier gaps.",
    illustrationKey: "fulfillment",
  },
];

