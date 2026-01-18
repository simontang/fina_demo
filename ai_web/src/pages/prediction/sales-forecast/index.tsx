import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Col,
  Collapse,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Switch,
  Table,
  Typography,
  message,
} from "antd";
import type { ColumnsType } from "antd/es/table";

import type { Dataset } from "../../../types/dataset";
import { TOKEN_KEY } from "../../../authProvider";

const { Title, Text } = Typography;

type AvailableModel = {
  id: string;
  name: string;
  version?: string;
  source?: string;
  description?: string;
  status?: string;
  deployed_at?: string;
  path?: string;
};

type SalesHierarchyCategory = {
  category: string;
  sub_categories: string[];
};

type SalesHierarchyResponse = {
  success: boolean;
  data?: {
    dataset_id: string;
    levels: string[];
    categories: SalesHierarchyCategory[];
  };
  detail?: string;
  error?: string;
};

type SalesHierarchySku = {
  sku: string;
  description?: string | null;
  total_revenue?: number;
};

type SalesHierarchySkuListResponse = {
  success: boolean;
  data?: SalesHierarchySku[];
  total?: number;
  detail?: string;
  error?: string;
};

type SalesForecastHistoryPoint = {
  date: string;
  sales: number;
  is_holiday?: number;
};

type SalesForecastRow = {
  date: string;
  predicted_sales: number;
  confidence_interval?: { lower: number; upper: number } | null;
};

type SalesForecastResponse = {
  status: "success" | "error";
  meta?: { model_version?: string; generated_at?: string };
  scope?: Record<string, any>;
  forecast?: SalesForecastRow[];
  simulation_forecast?: SalesForecastRow[];
  scenario?: Record<string, any>;
  trend_summary?: string;
  history?: SalesForecastHistoryPoint[];
  detail?: string;
};

type ScopeLevel = "category" | "sub_category" | "sku";
type PromotionIntensity = "none" | "standard" | "s_tier" | "custom";

type FormValues = {
  datasetId: string;
  modelId: string;
  scopeLevel: ScopeLevel;
  category?: string;
  subCategory?: string;
  skuId?: string;
  horizon: number;
  contextWindowDays: number;
  salesMetric: "quantity" | "revenue";
  baselinePromotionFactor: number;
  holidayCountry: string;
  rounding: "round" | "floor" | "none";

  scenarioEnabled: boolean;
  priceChangePct: number; // percent, e.g. -10 means -10%
  marketingBudget: number;
  marketGrowthPct: number; // percent
  promotionIntensity: PromotionIntensity;
  scenarioPromotionFactor: number;
};

function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function formatNumber(n: unknown): string {
  const num = typeof n === "number" ? n : Number(n);
  if (!Number.isFinite(num)) return "-";
  return num.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

function toISODateUTC(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function minusDaysUTC(dateStr: string, days: number): string {
  const d = new Date(`${dateStr}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() - days);
  return toISODateUTC(d);
}

function sumSales(rows: SalesForecastRow[] | undefined): number {
  if (!rows?.length) return 0;
  return rows.reduce((acc, r) => acc + (Number(r.predicted_sales) || 0), 0);
}

function pctDelta(a: number, b: number): number | null {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  if (Math.abs(a) < 1e-9) return null;
  return ((b - a) / a) * 100.0;
}

type ChartPoint = {
  date: string;
  history?: number;
  baseline?: number;
  sim?: number;
  ciLower?: number;
  ciUpper?: number;
  simCiLower?: number;
  simCiUpper?: number;
};

function buildLinePath(
  pts: ChartPoint[],
  xFor: (i: number) => number,
  yFor: (v: number) => number,
  getVal: (p: ChartPoint) => number | undefined
): string {
  let d = "";
  let started = false;
  for (let i = 0; i < pts.length; i++) {
    const v = getVal(pts[i]);
    if (!Number.isFinite(v as any)) {
      started = false;
      continue;
    }
    const x = xFor(i);
    const y = yFor(Number(v));
    if (!started) {
      d += `M ${x} ${y}`;
      started = true;
    } else {
      d += ` L ${x} ${y}`;
    }
  }
  return d;
}

function buildAreaPath(
  pts: ChartPoint[],
  xFor: (i: number) => number,
  yFor: (v: number) => number,
  getLower: (p: ChartPoint) => number | undefined,
  getUpper: (p: ChartPoint) => number | undefined
): string {
  const tops: Array<[number, number]> = [];
  const bots: Array<[number, number]> = [];
  for (let i = 0; i < pts.length; i++) {
    const lo = getLower(pts[i]);
    const hi = getUpper(pts[i]);
    if (!Number.isFinite(lo as any) || !Number.isFinite(hi as any)) continue;
    const x = xFor(i);
    tops.push([x, yFor(Number(hi))]);
    bots.push([x, yFor(Number(lo))]);
  }
  if (tops.length === 0) return "";
  const first = tops[0]!;
  let d = `M ${first[0]} ${first[1]}`;
  for (let i = 1; i < tops.length; i++) d += ` L ${tops[i]![0]} ${tops[i]![1]}`;
  for (let i = bots.length - 1; i >= 0; i--) d += ` L ${bots[i]![0]} ${bots[i]![1]}`;
  d += " Z";
  return d;
}

const PROMO_INTENSITY_PRESETS: Array<{ label: string; value: PromotionIntensity; factor: number }> = [
  { label: "None", value: "none", factor: 1.0 },
  { label: "Standard", value: "standard", factor: 1.05 },
  { label: "S-tier", value: "s_tier", factor: 1.2 },
  { label: "Custom", value: "custom", factor: 1.0 },
];

export const SalesForecast = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<AvailableModel[]>([]);

  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [hierarchy, setHierarchy] = useState<SalesHierarchyCategory[]>([]);
  const [loadingHierarchy, setLoadingHierarchy] = useState(false);
  const [hierarchyError, setHierarchyError] = useState<string | null>(null);

  const [skuOptions, setSkuOptions] = useState<SalesHierarchySku[]>([]);
  const [loadingSkus, setLoadingSkus] = useState(false);
  const [skuError, setSkuError] = useState<string | null>(null);

  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState<string | null>(null);
  const [forecast, setForecast] = useState<SalesForecastResponse | null>(null);

  const [form] = Form.useForm<FormValues>();

  useEffect(() => {
    const headers = getAuthHeaders();

    const loadDatasets = async () => {
      setLoadingDatasets(true);
      setDatasetsError(null);
      try {
        const res = await fetch("/api/v1/datasets", { headers });
        const data = await res.json();
        if (!data?.success) throw new Error(data?.detail || data?.error || "Failed to fetch datasets");
        setDatasets(data.data || []);
      } catch (e: any) {
        setDatasetsError(e?.message || "Failed to fetch datasets");
      } finally {
        setLoadingDatasets(false);
      }
    };

    const loadModels = async () => {
      setLoadingModels(true);
      setModelsError(null);
      try {
        const res = await fetch("/api/v1/models/available", { headers });
        const data = await res.json();
        if (!data?.success) throw new Error(data?.detail || data?.error || "Failed to fetch models");
        setModels(data.models || []);
      } catch (e: any) {
        setModelsError(e?.message || "Failed to fetch models");
      } finally {
        setLoadingModels(false);
      }
    };

    loadDatasets();
    loadModels();
  }, []);

  useEffect(() => {
    const current = form.getFieldsValue();
    const next: Partial<FormValues> = {};
    if (!current.datasetId && datasets.length > 0) next.datasetId = datasets[0]!.id;
    if (!current.modelId && models.length > 0) {
      const salesForecastModels = models.filter((m) => String(m.id || "").startsWith("sales_forecast:"));
      const preferred =
        (salesForecastModels.length > 0 ? salesForecastModels[salesForecastModels.length - 1] : undefined) ||
        models.find((m) => m.source === "assets") ||
        models[0]!;
      next.modelId = preferred.id;
    }
    if (Object.keys(next).length > 0) form.setFieldsValue(next as any);
  }, [datasets, models, form]);

  const loadHierarchy = async (datasetId: string) => {
    if (!datasetId) return;
    setLoadingHierarchy(true);
    setHierarchyError(null);
    try {
      const res = await fetch(`/api/v1/datasets/${encodeURIComponent(datasetId)}/sales-hierarchy`, {
        headers: getAuthHeaders(),
      });
      const data = (await res.json()) as SalesHierarchyResponse;
      if (!res.ok) throw new Error((data as any)?.detail || `HTTP ${res.status}`);
      if (!data.success) throw new Error(data.detail || data.error || "Failed to fetch hierarchy");
      setHierarchy(data.data?.categories || []);
    } catch (e: any) {
      setHierarchyError(e?.message || "Failed to fetch hierarchy");
      setHierarchy([]);
    } finally {
      setLoadingHierarchy(false);
    }
  };

  const loadSkus = async (datasetId: string, category: string, subCategory?: string) => {
    if (!datasetId || !category) return;
    setLoadingSkus(true);
    setSkuError(null);
    try {
      const q = new URLSearchParams();
      q.set("category", category);
      if (subCategory) q.set("sub_category", subCategory);
      q.set("limit", "200");
      const res = await fetch(`/api/v1/datasets/${encodeURIComponent(datasetId)}/sales-hierarchy/skus?${q.toString()}`, {
        headers: getAuthHeaders(),
      });
      const data = (await res.json()) as SalesHierarchySkuListResponse;
      if (!res.ok) throw new Error((data as any)?.detail || `HTTP ${res.status}`);
      if (!data.success) throw new Error(data.detail || data.error || "Failed to fetch SKUs");
      setSkuOptions(data.data || []);
    } catch (e: any) {
      setSkuError(e?.message || "Failed to fetch SKUs");
      setSkuOptions([]);
    } finally {
      setLoadingSkus(false);
    }
  };

  const selectedCategory = Form.useWatch("category", form);
  const selectedSubCategory = Form.useWatch("subCategory", form);
  const selectedScopeLevel = Form.useWatch("scopeLevel", form);
  const selectedDatasetId = Form.useWatch("datasetId", form);

  useEffect(() => {
    if (selectedDatasetId) loadHierarchy(String(selectedDatasetId));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDatasetId]);

  useEffect(() => {
    if (selectedScopeLevel === "sku" && selectedDatasetId && selectedCategory) {
      loadSkus(String(selectedDatasetId), String(selectedCategory), selectedSubCategory ? String(selectedSubCategory) : undefined);
    } else {
      setSkuOptions([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedScopeLevel, selectedDatasetId, selectedCategory, selectedSubCategory]);

  const categoryOptions = useMemo(
    () => hierarchy.map((c) => ({ label: c.category, value: c.category })),
    [hierarchy]
  );

  useEffect(() => {
    const cur = form.getFieldValue("category");
    if (!cur && hierarchy.length > 0) {
      form.setFieldValue("category", hierarchy[0]!.category);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hierarchy]);

  const subCategoryOptions = useMemo(() => {
    if (!selectedCategory) return [];
    const hit = hierarchy.find((c) => c.category === selectedCategory);
    const subs = hit?.sub_categories || [];
    return subs.map((s) => ({ label: s, value: s }));
  }, [hierarchy, selectedCategory]);

  const onRun = async (values: FormValues) => {
    setForecastLoading(true);
    setForecastError(null);
    setForecast(null);
    const headers = {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
    };

    try {
      const scenario =
        values.scenarioEnabled
          ? {
              enabled: true,
              price_change_pct: Number(values.priceChangePct || 0) / 100.0,
              price_elasticity: -1.2,
              marketing_budget: Number(values.marketingBudget || 0),
              marketing_roi: 3.0,
              promotion_factor: Number(values.scenarioPromotionFactor || 1.0),
              market_growth_pct: Number(values.marketGrowthPct || 0) / 100.0,
            }
          : undefined;

      const res = await fetch(`/api/v1/datasets/${encodeURIComponent(values.datasetId)}/sales-forecast`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model_id: values.modelId,
          scope_level: values.scopeLevel,
          category: values.category,
          sub_category: values.subCategory,
          sku_id: values.skuId,
          forecast_horizon: values.horizon,
          context_window_days: values.contextWindowDays,
          sales_metric: values.salesMetric,
          promotion_factor: values.baselinePromotionFactor,
          holiday_country: values.holidayCountry,
          rounding: values.rounding,
          scenario,
        }),
      });

      const data = (await res.json()) as SalesForecastResponse;
      if (!res.ok) throw new Error((data as any)?.detail || `HTTP ${res.status}`);
      if (data.status !== "success") throw new Error((data as any)?.detail || "Forecast failed");

      setForecast(data);
      message.success(values.scenarioEnabled ? "Simulation completed" : "Forecast completed");
    } catch (e: any) {
      const msg = e?.message || "Forecast failed";
      setForecastError(msg);
      message.error(msg);
    } finally {
      setForecastLoading(false);
    }
  };

  const meta = forecast?.meta;
  const scope = forecast?.scope || {};
  const metaText = useMemo(() => {
    const mv = meta?.model_version ? `Model: ${meta.model_version}` : "";
    const gt = meta?.generated_at ? `Generated at: ${meta.generated_at}` : "";
    const sc =
      scope?.level === "category"
        ? `Scope: Category / ${scope?.category || "-"}`
        : scope?.level === "sub_category"
          ? `Scope: Sub-category / ${scope?.category || "-"} / ${scope?.sub_category || "-"}`
          : scope?.level === "sku"
            ? `Scope: SKU / ${scope?.sku_id || "-"}`
            : "";
    return [mv, sc, gt].filter(Boolean).join(" · ");
  }, [meta, scope]);

  const baselineTotal = useMemo(() => sumSales(forecast?.forecast), [forecast?.forecast]);
  const simTotal = useMemo(() => sumSales(forecast?.simulation_forecast), [forecast?.simulation_forecast]);
  const simDeltaPct = useMemo(() => pctDelta(baselineTotal, simTotal), [baselineTotal, simTotal]);

  const historyMap = useMemo(() => {
    const m = new Map<string, number>();
    for (const h of forecast?.history || []) m.set(h.date, Number(h.sales) || 0);
    return m;
  }, [forecast?.history]);

  type GridRow = {
    date: string;
    scope: string;
    baseline: number;
    simulated?: number;
    deltaPct?: number | null;
    lastWeek?: number;
  };

  const gridRows: GridRow[] = useMemo(() => {
    const base = forecast?.forecast || [];
    const sim = forecast?.simulation_forecast || [];
    const label =
      scope?.level === "category"
        ? String(scope?.category || "Category")
        : scope?.level === "sub_category"
          ? String(scope?.sub_category || "Sub-category")
          : scope?.level === "sku"
            ? String(scope?.sku_id || "SKU")
            : "Scope";

    return base.map((b, i) => {
      const s = sim[i];
      const baseline = Number(b.predicted_sales) || 0;
      const simulated = s ? Number(s.predicted_sales) || 0 : undefined;
      const lastWeek = historyMap.get(minusDaysUTC(b.date, 7));
      return {
        date: b.date,
        scope: label,
        baseline,
        simulated,
        deltaPct: simulated != null ? pctDelta(baseline, simulated) : null,
        lastWeek,
      };
    });
  }, [forecast?.forecast, forecast?.simulation_forecast, historyMap, scope]);

  const gridColumns: ColumnsType<GridRow> = useMemo(
    () => [
      { title: "Date", dataIndex: "date", key: "date", width: 120 },
      { title: "Scope", dataIndex: "scope", key: "scope", width: 160 },
      {
        title: "Baseline Revenue",
        dataIndex: "baseline",
        key: "baseline",
        render: (v) => formatNumber(v),
      },
      {
        title: "Simulated Revenue",
        dataIndex: "simulated",
        key: "simulated",
        render: (v) => formatNumber(v),
      },
      {
        title: "Delta (%)",
        dataIndex: "deltaPct",
        key: "deltaPct",
        width: 120,
        render: (v) => (typeof v === "number" ? `${v.toFixed(2)}%` : "-"),
      },
      {
        title: "Last Week (Actual)",
        dataIndex: "lastWeek",
        key: "lastWeek",
        render: (v) => formatNumber(v),
      },
    ],
    []
  );

  const chartPoints: ChartPoint[] = useMemo(() => {
    const pts: ChartPoint[] = [];
    for (const h of forecast?.history || []) {
      pts.push({ date: h.date, history: Number(h.sales) || 0 });
    }
    const sim = forecast?.simulation_forecast || [];
    for (let i = 0; i < (forecast?.forecast || []).length; i++) {
      const b = forecast?.forecast?.[i];
      if (!b) continue;
      const s = sim[i];
      pts.push({
        date: b.date,
        baseline: Number(b.predicted_sales) || 0,
        ciLower: b.confidence_interval?.lower,
        ciUpper: b.confidence_interval?.upper,
        sim: s ? Number(s.predicted_sales) || 0 : undefined,
        simCiLower: s?.confidence_interval?.lower,
        simCiUpper: s?.confidence_interval?.upper,
      });
    }
    // Ensure chronological order
    pts.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
    return pts;
  }, [forecast?.history, forecast?.forecast, forecast?.simulation_forecast]);

  const Chart = () => {
    const svgRef = useRef<SVGSVGElement | null>(null);
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);
    const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null);

    const width = 900;
    const height = 320;
    const pad = 40;

    const n = chartPoints.length;
    const xFor = (i: number) => (n <= 1 ? pad : pad + (i * (width - 2 * pad)) / (n - 1));

    const ys = chartPoints.flatMap((p) => [
      p.history,
      p.baseline,
      p.sim,
      p.ciLower,
      p.ciUpper,
      p.simCiLower,
      p.simCiUpper,
    ]);
    const yMax = Math.max(1, ...ys.map((v) => (Number.isFinite(v as any) ? Number(v) : 0)));
    const yFor = (v: number) => height - pad - (Math.max(0, v) / yMax) * (height - 2 * pad);

    const area = buildAreaPath(chartPoints, xFor, yFor, (p) => p.ciLower, (p) => p.ciUpper);
    const lineHistory = buildLinePath(chartPoints, xFor, yFor, (p) => p.history);
    const lineBaseline = buildLinePath(chartPoints, xFor, yFor, (p) => p.baseline);
    const lineSim = buildLinePath(chartPoints, xFor, yFor, (p) => p.sim);

    const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg || n <= 0) return;
      const rect = svg.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const idx = Math.max(0, Math.min(n - 1, Math.round((x / rect.width) * (n - 1))));
      setHoverIndex(idx);
      setHoverPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    };

    const onLeave = () => {
      setHoverIndex(null);
      setHoverPos(null);
    };

    const hp = hoverIndex != null ? chartPoints[hoverIndex] : null;

    return (
      <div style={{ position: "relative" }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          width="100%"
          height={height}
          onMouseMove={onMove}
          onMouseLeave={onLeave}
          style={{ display: "block", background: "#fff" }}
        >
          {/* axes */}
          <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#e5e7eb" />
          <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#e5e7eb" />

          {/* CI area (baseline) */}
          {area ? <path d={area} fill="rgba(59,130,246,0.12)" stroke="none" /> : null}

          {/* Lines */}
          {lineHistory ? <path d={lineHistory} fill="none" stroke="#6b7280" strokeWidth={2} /> : null}
          {lineBaseline ? (
            <path d={lineBaseline} fill="none" stroke="#2563eb" strokeWidth={2} strokeDasharray="6 4" />
          ) : null}
          {lineSim ? <path d={lineSim} fill="none" stroke="#f97316" strokeWidth={2} strokeDasharray="6 4" /> : null}

          {/* Hover guide */}
          {hoverIndex != null ? (
            <line x1={xFor(hoverIndex)} y1={pad} x2={xFor(hoverIndex)} y2={height - pad} stroke="#d1d5db" />
          ) : null}
        </svg>

        {hp && hoverPos ? (
          <div
            style={{
              position: "absolute",
              left: Math.min(hoverPos.x + 12, 520),
              top: Math.max(8, hoverPos.y - 10),
              background: "rgba(0,0,0,0.78)",
              color: "#fff",
              padding: "8px 10px",
              borderRadius: 6,
              fontSize: 12,
              pointerEvents: "none",
              width: 260,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 6 }}>{hp.date}</div>
            <div>History: {formatNumber(hp.history)}</div>
            <div>Baseline: {formatNumber(hp.baseline)}</div>
            <div>Scenario: {formatNumber(hp.sim)}</div>
            <div>CI: {formatNumber(hp.ciLower)} – {formatNumber(hp.ciUpper)}</div>
          </div>
        ) : null}
      </div>
    );
  };

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Title level={3} style={{ margin: 0 }}>
            Sales Forecast & Scenario Planner
          </Title>
          <Text type="secondary">
            Forecast revenue by category/sub-category/SKU, then run a scenario simulation (price, promo, marketing).
          </Text>
        </Col>

        <Col xs={24} md={8}>
          <Card title="Configuration">
            <Form<FormValues>
              form={form}
              layout="vertical"
              initialValues={{
                datasetId: "",
                modelId: "",
                scopeLevel: "category",
                category: undefined,
                subCategory: undefined,
                skuId: undefined,
                horizon: 30,
                contextWindowDays: 60,
                salesMetric: "revenue",
                baselinePromotionFactor: 1.0,
                holidayCountry: "GB",
                rounding: "none",
                scenarioEnabled: true,
                priceChangePct: 0,
                marketingBudget: 0,
                marketGrowthPct: 0,
                promotionIntensity: "none",
                scenarioPromotionFactor: 1.0,
              }}
              onFinish={onRun}
              onValuesChange={(changed, all) => {
                if ("promotionIntensity" in changed) {
                  const v = (changed as any).promotionIntensity as PromotionIntensity;
                  const preset = PROMO_INTENSITY_PRESETS.find((p) => p.value === v);
                  if (preset && v !== "custom") {
                    form.setFieldValue("scenarioPromotionFactor", preset.factor);
                  }
                }
                if ("category" in changed) {
                  form.setFieldValue("subCategory", undefined);
                  form.setFieldValue("skuId", undefined);
                }
                if ("subCategory" in changed) {
                  form.setFieldValue("skuId", undefined);
                }
                if ("scopeLevel" in changed) {
                  const lvl = (changed as any).scopeLevel as ScopeLevel;
                  if (lvl === "category") {
                    form.setFieldValue("subCategory", undefined);
                    form.setFieldValue("skuId", undefined);
                  }
                  if (lvl === "sub_category") {
                    form.setFieldValue("skuId", undefined);
                  }
                }
              }}
            >
              <Form.Item label="Dataset" name="datasetId" rules={[{ required: true, message: "Please select a dataset" }]}>
                <Select
                  loading={loadingDatasets}
                  placeholder="Select a dataset"
                  options={datasets.map((d) => ({ label: `${d.name} (${d.id})`, value: d.id }))}
                  disabled={datasets.length === 0}
                  onChange={(v) => {
                    setForecast(null);
                    setForecastError(null);
                    setHierarchy([]);
                    setSkuOptions([]);
                    if (v) loadHierarchy(String(v));
                  }}
                />
              </Form.Item>
              {datasetsError && <Alert type="error" showIcon message={datasetsError} style={{ marginBottom: 12 }} />}

              <Form.Item label="Model" name="modelId" rules={[{ required: true, message: "Please select a model" }]}>
                <Select
                  loading={loadingModels}
                  placeholder="Select a model"
                  options={models.map((m) => ({
                    label: `${m.name} (${m.id})${m.source ? ` · ${m.source}` : ""}`,
                    value: m.id,
                  }))}
                  disabled={models.length === 0}
                />
              </Form.Item>
              {modelsError && <Alert type="error" showIcon message={modelsError} style={{ marginBottom: 12 }} />}

              <Form.Item label="Scope Level" name="scopeLevel" rules={[{ required: true }]}>
                <Select
                  options={[
                    { label: "Category", value: "category" },
                    { label: "Sub-category (Top-down)", value: "sub_category" },
                    { label: "SKU", value: "sku" },
                  ]}
                />
              </Form.Item>

              <Form.Item
                label="Category"
                name="category"
                rules={[
                  ({ getFieldValue }) => ({
                    validator: (_, v) => {
                      const lvl = getFieldValue("scopeLevel") as ScopeLevel;
                      if (lvl === "sku" || lvl === "category" || lvl === "sub_category") {
                        return v ? Promise.resolve() : Promise.reject(new Error("Please select a category"));
                      }
                      return Promise.resolve();
                    },
                  }),
                ]}
              >
                <Select
                  loading={loadingHierarchy}
                  placeholder={loadingHierarchy ? "Loading..." : "Select a category"}
                  options={categoryOptions}
                  disabled={!selectedDatasetId || hierarchy.length === 0}
                />
              </Form.Item>
              {hierarchyError && <Alert type="error" showIcon message={hierarchyError} style={{ marginBottom: 12 }} />}

              {(selectedScopeLevel === "sub_category" || selectedScopeLevel === "sku") && (
                <Form.Item
                  label="Sub-category"
                  name="subCategory"
                  rules={[
                    ({ getFieldValue }) => ({
                      validator: (_, v) => {
                        const lvl = getFieldValue("scopeLevel") as ScopeLevel;
                        if (lvl === "sub_category") {
                          return v ? Promise.resolve() : Promise.reject(new Error("Please select a sub-category"));
                        }
                        return Promise.resolve();
                      },
                    }),
                  ]}
                >
                  <Select
                    placeholder="Select a sub-category"
                    options={subCategoryOptions}
                    disabled={!selectedCategory || subCategoryOptions.length === 0}
                  />
                </Form.Item>
              )}

              {selectedScopeLevel === "sku" && (
                <Form.Item
                  label="SKU"
                  name="skuId"
                  rules={[{ required: true, message: "Please select a SKU" }]}
                  extra="SKU list is filtered by the selected category/sub-category (top 200 by revenue)."
                >
                  <Select
                    showSearch
                    loading={loadingSkus}
                    placeholder="Select a SKU"
                    options={skuOptions.map((s) => ({
                      label: `${s.sku}${s.description ? ` · ${s.description}` : ""}`,
                      value: s.sku,
                    }))}
                    disabled={!selectedCategory || skuOptions.length === 0}
                    filterOption={(input, option) =>
                      String(option?.label || "")
                        .toLowerCase()
                        .includes(String(input || "").toLowerCase())
                    }
                  />
                </Form.Item>
              )}
              {skuError && <Alert type="error" showIcon message={skuError} style={{ marginBottom: 12 }} />}

              <Row gutter={12}>
                <Col xs={24} md={12}>
                  <Form.Item label="Horizon (days)" name="horizon" rules={[{ required: true }]}>
                    <InputNumber min={1} max={365} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item label="History Window (days)" name="contextWindowDays" rules={[{ required: true }]}>
                    <InputNumber min={7} max={3650} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={12}>
                <Col xs={24} md={12}>
                  <Form.Item label="Metric" name="salesMetric" rules={[{ required: true }]}>
                    <Select
                      options={[
                        { label: "Revenue (Quantity x UnitPrice)", value: "revenue" },
                        { label: "Quantity", value: "quantity" },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item label="Baseline Promotion Factor" name="baselinePromotionFactor" rules={[{ required: true }]}>
                    <InputNumber min={0.01} step={0.05} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={12}>
                <Col xs={24} md={12}>
                  <Form.Item label="Holiday Country Code" name="holidayCountry" rules={[{ required: true }]}>
                    <Input placeholder="e.g. GB / US" />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item label="Rounding" name="rounding" rules={[{ required: true }]}>
                    <Select
                      options={[
                        { label: "Round", value: "round" },
                        { label: "Floor", value: "floor" },
                        { label: "None", value: "none" },
                      ]}
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Collapse
                items={[
                  {
                    key: "scenario",
                    label: "Scenario Simulation",
                    children: (
                      <Space direction="vertical" style={{ width: "100%" }} size={12}>
                        <Form.Item name="scenarioEnabled" valuePropName="checked" style={{ marginBottom: 0 }}>
                          <Switch checkedChildren="On" unCheckedChildren="Off" />
                        </Form.Item>

                        <Row gutter={12}>
                          <Col xs={24} md={12}>
                            <Form.Item label="Price Change (%)" name="priceChangePct">
                              <InputNumber min={-50} max={50} step={1} style={{ width: "100%" }} />
                            </Form.Item>
                          </Col>
                          <Col xs={24} md={12}>
                            <Form.Item label="Marketing Budget" name="marketingBudget">
                              <InputNumber min={0} step={100} style={{ width: "100%" }} />
                            </Form.Item>
                          </Col>
                        </Row>

                        <Row gutter={12}>
                          <Col xs={24} md={12}>
                            <Form.Item label="Market Growth (%)" name="marketGrowthPct">
                              <InputNumber min={-50} max={200} step={1} style={{ width: "100%" }} />
                            </Form.Item>
                          </Col>
                          <Col xs={24} md={12}>
                            <Form.Item label="Promotion Intensity" name="promotionIntensity">
                              <Select
                                options={PROMO_INTENSITY_PRESETS.map((p) => ({ label: p.label, value: p.value }))}
                              />
                            </Form.Item>
                          </Col>
                        </Row>

                        <Form.Item label="Scenario Promotion Factor" name="scenarioPromotionFactor">
                          <InputNumber min={0.01} step={0.05} style={{ width: "100%" }} />
                        </Form.Item>
                      </Space>
                    ),
                  },
                ]}
              />

              <Space style={{ marginTop: 16 }}>
                <Button type="primary" htmlType="submit" loading={forecastLoading}>
                  Run
                </Button>
                <Button
                  onClick={() => {
                    form.setFieldsValue({
                      datasetId: datasets[0]?.id || "",
                      modelId: models[0]?.id || "baseline_moving_average",
                      scopeLevel: "category",
                      category: undefined,
                      subCategory: undefined,
                      skuId: undefined,
                      horizon: 30,
                      contextWindowDays: 60,
                      salesMetric: "revenue",
                      baselinePromotionFactor: 1.0,
                      holidayCountry: "GB",
                      rounding: "none",
                      scenarioEnabled: true,
                      priceChangePct: 0,
                      marketingBudget: 0,
                      marketGrowthPct: 0,
                      promotionIntensity: "none",
                      scenarioPromotionFactor: 1.0,
                    });
                    setForecast(null);
                    setForecastError(null);
                  }}
                >
                  Reset
                </Button>
              </Space>
            </Form>
          </Card>
        </Col>

        <Col xs={24} md={16}>
          <Card title="Analysis" extra={metaText ? <Text type="secondary">{metaText}</Text> : null}>
            {forecastError && <Alert type="error" showIcon message={forecastError} style={{ marginBottom: 12 }} />}

            {forecastLoading ? (
              <Spin />
            ) : forecast?.status === "success" ? (
              <Space direction="vertical" style={{ width: "100%" }} size={16}>
                <Row gutter={[16, 16]}>
                  <Col xs={24} md={6}>
                    <Statistic title="Trend (Baseline)" value={forecast.trend_summary || "-"} />
                  </Col>
                  <Col xs={24} md={6}>
                    <Statistic title="Baseline Total" value={formatNumber(baselineTotal)} />
                  </Col>
                  <Col xs={24} md={6}>
                    <Statistic title="Scenario Total" value={forecast.simulation_forecast ? formatNumber(simTotal) : "-"} />
                  </Col>
                  <Col xs={24} md={6}>
                    <Statistic
                      title="Scenario Delta"
                      value={simDeltaPct == null ? "-" : `${simDeltaPct.toFixed(2)}%`}
                    />
                  </Col>
                </Row>

                <Card size="small" title="Chart (History + Forecast + Scenario)">
                  <Chart />
                </Card>

                <Card size="small" title="Forecast Grid (Baseline vs Scenario)">
                  <Table
                    rowKey={(r) => r.date}
                    size="small"
                    pagination={false}
                    columns={gridColumns}
                    dataSource={gridRows}
                    scroll={{ x: "max-content" }}
                  />
                </Card>
              </Space>
            ) : (
              <Text type="secondary">Select scope + parameters on the left, then click Run.</Text>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};
