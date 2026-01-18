import React, { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Col,
  Divider,
  Form,
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
import MDEditor from "@uiw/react-md-editor/nohighlight";

import type { Dataset } from "../../../types/dataset";
import { TOKEN_KEY } from "../../../authProvider";

const { Title, Text } = Typography;

type SegmentationFormValues = {
  datasetId: string;
  timeWindowDays: number;
  kMin: number;
  kMax: number;
  randomSeed: number;
  outlierThreshold: number;
  enableAiInsight: boolean;
};

type ClusterCharacteristic = { mean: number; std: number };
type ClusterSummary = {
  cluster_id: number;
  size: number;
  percentage: string;
  characteristics: Record<string, ClusterCharacteristic>;
  label_suggestion?: string;
};

type SegmentationModelInfo = {
  best_k: number;
  best_silhouette_score: number;
  features_used: string[];
};

type ElbowPoint = { k: number; sse: number };
type FeatureDefinition = { name: string; description: string };
type CategoryOverviewRow = {
  category: string;
  revenue: number;
  revenue_share_pct: number;
  line_items: number;
};

type SeasonOverviewRow = {
  season: string;
  revenue: number;
  revenue_share_pct: number;
  orders: number;
};

type ClusterPlotPoint = {
  x: number;
  y: number;
  cluster_id: number;
  user_id?: string;
};

type ClusterPlotCentroid = {
  cluster_id: number;
  x: number;
  y: number;
};

type ClusterPlot = {
  method: string;
  x_label: string;
  y_label: string;
  explained_variance_ratio?: Array<number | null>;
  n_points_total: number;
  n_points: number;
  points: ClusterPlotPoint[];
  centroids?: ClusterPlotCentroid[];
};

type SegmentationResponse = {
  status: "success" | "error";
  message?: string;
  dataset_id?: string;
  reference_date?: string;
  time_window_days?: number;
  feature_mode?: "category_mix" | "rfm";
  model_info?: SegmentationModelInfo;
  clusters_summary?: ClusterSummary[];
  data_preview?: Record<string, any>[];
  elbow_curve_data?: ElbowPoint[];
  warnings?: string[];
  log_transform?: Record<string, { skew: number; shift: number }>;
  feature_definitions?: FeatureDefinition[];
  category_overview?: CategoryOverviewRow[];
  season_overview?: SeasonOverviewRow[];
  insight_markdown?: string;
  cluster_plot?: ClusterPlot;
};

function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function formatNumber(n: unknown): string {
  const num = typeof n === "number" ? n : Number(n);
  if (!Number.isFinite(num)) return "-";
  return num.toLocaleString("en-US");
}

function formatFloat(n: unknown, digits = 4): string {
  const num = typeof n === "number" ? n : Number(n);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function formatFeatureValue(feature: string, value: unknown): string {
  const num = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(num)) return "-";
  if (feature.startsWith("cat_share_") || feature.startsWith("season_share_")) return `${(num * 100).toFixed(1)}%`;
  if (feature === "frequency") return num.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (feature === "monetary") return num.toLocaleString("en-US", { maximumFractionDigits: 2 });
  return formatNumber(num);
}

function clusterColor(clusterId: number): string {
  const palette = [
    "#1D70B8",
    "#FF7043",
    "#43A047",
    "#8E24AA",
    "#00897B",
    "#F9A825",
    "#3949AB",
    "#D81B60",
    "#546E7A",
    "#6D4C41",
  ];
  const idx = Math.abs(Number(clusterId)) % palette.length;
  return palette[idx]!;
}

function prettifyShareKey(key: string, prefix: "cat_share_" | "season_share_"): string {
  const raw = key.startsWith(prefix) ? key.slice(prefix.length) : key;
  const words = raw
    .split("_")
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1));
  return words.join(" ");
}

function topSharesSummary(
  characteristics: Record<string, ClusterCharacteristic> | undefined,
  prefix: "cat_share_" | "season_share_",
  n: number
): string {
  if (!characteristics) return "-";
  const items: Array<{ k: string; mean: number }> = [];
  for (const [k, v] of Object.entries(characteristics)) {
    if (!k.startsWith(prefix)) continue;
    const m = typeof v?.mean === "number" ? v.mean : Number((v as any)?.mean);
    if (!Number.isFinite(m)) continue;
    items.push({ k, mean: m });
  }
  items.sort((a, b) => b.mean - a.mean);
  return items
    .slice(0, Math.max(0, n))
    .map((it) => `${prettifyShareKey(it.k, prefix)} ${(it.mean * 100).toFixed(1)}%`)
    .join(", ") || "-";
}

const ClusterScatter: React.FC<{
  plot: ClusterPlot;
  labels?: Record<string, string>;
  activeClusterId: number | null;
  onSelectCluster: (clusterId: number) => void;
}> = ({ plot, labels, activeClusterId, onSelectCluster }) => {
  const w = 1000;
  const h = 420;
  const pad = 36;

  const pts = plot.points || [];
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const minX = xs.length ? Math.min(...xs) : -1;
  const maxX = xs.length ? Math.max(...xs) : 1;
  const minY = ys.length ? Math.min(...ys) : -1;
  const maxY = ys.length ? Math.max(...ys) : 1;
  const dx = maxX - minX || 1;
  const dy = maxY - minY || 1;

  const xScale = (x: number) => pad + ((x - minX) / dx) * (w - pad * 2);
  const yScale = (y: number) => pad + ((maxY - y) / dy) * (h - pad * 2);

  const evr = plot.explained_variance_ratio;
  const evrText =
    evr && evr.length >= 2 && typeof evr[0] === "number" && typeof evr[1] === "number"
      ? `Explained variance: ${(evr[0] * 100).toFixed(1)}% + ${(evr[1] * 100).toFixed(1)}%`
      : null;

  return (
    <div>
      <Text type="secondary">
        {plot.x_label} vs {plot.y_label} · Showing {formatNumber(plot.n_points)} of {formatNumber(plot.n_points_total)}{" "}
        users
        {evrText ? ` · ${evrText}` : ""}
      </Text>
      <div style={{ marginTop: 12, width: "100%", overflow: "hidden" }}>
        <svg
          viewBox={`0 0 ${w} ${h}`}
          style={{
            width: "100%",
            height: 420,
            borderRadius: 8,
            background: "rgba(148, 163, 184, 0.08)",
          }}
        >
          <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="rgba(15, 23, 42, 0.35)" />
          <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="rgba(15, 23, 42, 0.35)" />

          {pts.map((p, i) => {
            const active = activeClusterId !== null;
            const isActive = active && p.cluster_id === activeClusterId;
            const opacity = !active ? 0.75 : isActive ? 0.9 : 0.15;
            const r = !active ? 2 : isActive ? 2.4 : 1.8;
            const label = labels?.[String(p.cluster_id)] || `Cluster ${p.cluster_id}`;
            return (
              <circle
                key={i}
                cx={xScale(p.x)}
                cy={yScale(p.y)}
                r={r}
                fill={clusterColor(p.cluster_id)}
                opacity={opacity}
                onClick={() => onSelectCluster(p.cluster_id)}
                style={{ cursor: "pointer" }}
              >
                <title>
                  {label}
                  {p.user_id ? `\nuser_id: ${p.user_id}` : ""}
                </title>
              </circle>
            );
          })}

          {(plot.centroids || []).map((c) => {
            const label = labels?.[String(c.cluster_id)] || `Cluster ${c.cluster_id}`;
            const active = activeClusterId !== null;
            const isActive = active && c.cluster_id === activeClusterId;
            return (
              <circle
                key={`c-${c.cluster_id}`}
                cx={xScale(c.x)}
                cy={yScale(c.y)}
                r={isActive ? 8 : 7}
                fill={clusterColor(c.cluster_id)}
                stroke="rgba(15, 23, 42, 0.85)"
                strokeWidth={2}
                onClick={() => onSelectCluster(c.cluster_id)}
                style={{ cursor: "pointer" }}
              >
                <title>{label} (centroid)</title>
              </circle>
            );
          })}
        </svg>
      </div>
    </div>
  );
};

export const Segmentation: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  const [runLoading, setRunLoading] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<SegmentationResponse | null>(null);
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);

  const [form] = Form.useForm<SegmentationFormValues>();

  useEffect(() => {
    let cancelled = false;
    const loadDatasets = async () => {
      setDatasetsLoading(true);
      setDatasetsError(null);
      try {
        const res = await fetch("/api/v1/datasets", { headers: getAuthHeaders() });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err?.detail || `HTTP ${res.status}`);
        }
        const json = await res.json();
        if (!json?.success) throw new Error(json?.error || json?.message || "Failed to load datasets");
        if (!cancelled) setDatasets(json.data || []);
      } catch (e: any) {
        if (!cancelled) setDatasetsError(e?.message || "Failed to load datasets");
      } finally {
        if (!cancelled) setDatasetsLoading(false);
      }
    };
    loadDatasets();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const current = form.getFieldsValue();
    if (!current.datasetId && datasets.length > 0) {
      form.setFieldsValue({ datasetId: datasets[0]!.id } as any);
    }
  }, [datasets, form]);

  useEffect(() => {
    const first = result?.clusters_summary?.[0]?.cluster_id;
    if (typeof first === "number") setSelectedClusterId(first);
  }, [result]);

  const onRun = async (values: SegmentationFormValues) => {
    setRunLoading(true);
    setRunError(null);
    setResult(null);
    try {
      const datasetId = values.datasetId;
      if (!datasetId) throw new Error("Please select a dataset");

      const body = {
        feature_mode: "category_mix",
        time_window_days: Number(values.timeWindowDays),
        k_range: [Number(values.kMin), Number(values.kMax)],
        random_seed: Number(values.randomSeed),
        outlier_threshold: Number(values.outlierThreshold),
        enable_ai_insight: Boolean(values.enableAiInsight),
      };

      const res = await fetch(`/api/v1/datasets/${encodeURIComponent(datasetId)}/segmentation`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders(),
        },
        body: JSON.stringify(body),
      });

      const json = (await res.json().catch(() => ({}))) as any;
      if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`);
      if (json?.status !== "success") throw new Error(json?.message || "Segmentation failed");

      setResult(json as SegmentationResponse);
      message.success("Segmentation completed");
    } catch (e: any) {
      const msg = e?.message || "Segmentation failed";
      setRunError(msg);
      message.error(msg);
    } finally {
      setRunLoading(false);
    }
  };

  const elbowColumns: ColumnsType<ElbowPoint> = useMemo(
    () => [
      { title: "K", dataIndex: "k", key: "k", width: 80 },
      { title: "SSE", dataIndex: "sse", key: "sse", render: (v) => formatNumber(v) },
    ],
    []
  );

  const categoryColumns: ColumnsType<CategoryOverviewRow> = useMemo(
    () => [
      { title: "Category", dataIndex: "category", key: "category", width: 180 },
      { title: "Revenue", dataIndex: "revenue", key: "revenue", width: 140, render: (v) => formatNumber(v) },
      {
        title: "Revenue Share",
        dataIndex: "revenue_share_pct",
        key: "revenue_share_pct",
        width: 140,
        render: (v) => `${formatFloat(v, 2)}%`,
      },
      { title: "Line Items", dataIndex: "line_items", key: "line_items", width: 120, render: (v) => formatNumber(v) },
    ],
    []
  );

  const seasonColumns: ColumnsType<SeasonOverviewRow> = useMemo(
    () => [
      { title: "Season", dataIndex: "season", key: "season", width: 140 },
      { title: "Revenue", dataIndex: "revenue", key: "revenue", width: 160, render: (v) => formatNumber(v) },
      {
        title: "Revenue Share",
        dataIndex: "revenue_share_pct",
        key: "revenue_share_pct",
        width: 160,
        render: (v) => `${formatFloat(v, 2)}%`,
      },
      { title: "Orders", dataIndex: "orders", key: "orders", width: 120, render: (v) => formatNumber(v) },
    ],
    []
  );

  const featureColumns: ColumnsType<FeatureDefinition> = useMemo(
    () => [
      { title: "Feature", dataIndex: "name", key: "name", width: 220 },
      { title: "Definition", dataIndex: "description", key: "description" },
    ],
    []
  );

  const clusterColumns: ColumnsType<ClusterSummary> = useMemo(() => {
    const cols: ColumnsType<ClusterSummary> = [
      { title: "Cluster", dataIndex: "cluster_id", key: "cluster_id", width: 90 },
      { title: "Users", dataIndex: "size", key: "size", width: 100, render: (v) => formatNumber(v) },
      { title: "Share", dataIndex: "percentage", key: "percentage", width: 90 },
      { title: "Suggested Label", dataIndex: "label_suggestion", key: "label_suggestion", width: 220 },
      {
        title: "Top Categories",
        key: "top_categories",
        width: 260,
        render: (_, r) => topSharesSummary(r.characteristics, "cat_share_", 2),
      },
      {
        title: "Top Seasons",
        key: "top_seasons",
        width: 220,
        render: (_, r) => topSharesSummary(r.characteristics, "season_share_", 1),
      },
      {
        title: "Avg Frequency",
        key: "avg_frequency",
        width: 140,
        render: (_, r) => formatFeatureValue("frequency", r.characteristics?.frequency?.mean),
      },
      {
        title: "Avg Monetary",
        key: "avg_monetary",
        width: 160,
        render: (_, r) => formatFeatureValue("monetary", r.characteristics?.monetary?.mean),
      },
    ];
    return cols;
  }, [result]);

  const previewColumns: ColumnsType<Record<string, any>> = useMemo(() => {
    const rec = result?.data_preview?.[0];
    if (!rec) return [];
    return Object.keys(rec).map((k) => ({
      title: k,
      dataIndex: k,
      key: k,
      render: (v) => (typeof v === "number" ? formatNumber(v) : String(v)),
    }));
  }, [result]);

  const silhouetteHint = useMemo(() => {
    const s = result?.model_info?.best_silhouette_score;
    if (typeof s !== "number" || !Number.isFinite(s)) return null;
    if (s > 0.5) return <Text type="success">Strong separation</Text>;
    if (s >= 0.25) return <Text type="warning">Moderate separation</Text>;
    return <Text type="danger">Weak separation</Text>;
  }, [result]);

  const selectedCluster = useMemo(() => {
    const list = result?.clusters_summary || [];
    if (selectedClusterId === null) return null;
    return list.find((c) => c.cluster_id === selectedClusterId) || null;
  }, [result, selectedClusterId]);

  const selectedClusterFeatureRows = useMemo(() => {
    if (!selectedCluster || !result?.model_info?.features_used) return [];
    return result.model_info.features_used.map((f) => ({
      feature: f,
      mean: selectedCluster.characteristics?.[f]?.mean,
      std: selectedCluster.characteristics?.[f]?.std,
    }));
  }, [result, selectedCluster]);

  const selectedClusterFeatureCols: ColumnsType<{ feature: string; mean: number; std: number }> = useMemo(
    () => [
      { title: "Feature", dataIndex: "feature", key: "feature", width: 220 },
      { title: "Mean", dataIndex: "mean", key: "mean", width: 160, render: (v, r) => formatFeatureValue(r.feature, v) },
      { title: "Std", dataIndex: "std", key: "std", width: 160, render: (v, r) => formatFeatureValue(r.feature, v) },
    ],
    []
  );

  const clusterLabels = useMemo(() => {
    const out: Record<string, string> = {};
    for (const c of result?.clusters_summary || []) {
      out[String(c.cluster_id)] = c.label_suggestion || `Cluster ${c.cluster_id}`;
    }
    return out;
  }, [result]);

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Title level={3} style={{ margin: 0 }}>
            Segmentation (Category-Based Customer Clustering)
          </Title>
          <Text type="secondary">
            Step 1: clean product descriptions into categories. Step 2: build customer features (category mix, purchase
            seasonality, frequency, and monetary). Step 3: run K-Means and pick the best K by silhouette score.
          </Text>
        </Col>

        <Col span={24}>
          <Card title="Configuration">
            {datasetsError && (
              <Alert
                type="error"
                showIcon
                message="Failed to load datasets"
                description={datasetsError}
                style={{ marginBottom: 12 }}
              />
            )}

            <Form<SegmentationFormValues>
              form={form}
              layout="vertical"
              initialValues={{
                datasetId: "",
                timeWindowDays: 365,
                kMin: 3,
                kMax: 6,
                randomSeed: 42,
                outlierThreshold: 3.0,
                enableAiInsight: true,
              }}
              onFinish={onRun}
            >
              <Row gutter={16}>
                <Col xs={24} md={8}>
                  <Form.Item label="Dataset" name="datasetId" rules={[{ required: true, message: "Select a dataset" }]}>
                    <Select
                      placeholder={datasetsLoading ? "Loading..." : "Select a dataset"}
                      loading={datasetsLoading}
                      options={datasets.map((d) => ({ label: `${d.name} (${d.id})`, value: d.id }))}
                      showSearch
                      optionFilterProp="label"
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={8}>
                  <Form.Item label="Time Window (days)" name="timeWindowDays" rules={[{ required: true }]}>
                    <InputNumber min={1} max={3650} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={8}>
                  <Form.Item label="AI Interpretation" name="enableAiInsight" valuePropName="checked">
                    <Switch checkedChildren="On" unCheckedChildren="Off" />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col xs={24} md={6}>
                  <Form.Item label="K Min" name="kMin" rules={[{ required: true }]}>
                    <InputNumber min={2} max={50} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="K Max" name="kMax" rules={[{ required: true }]}>
                    <InputNumber min={2} max={50} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="Random Seed" name="randomSeed" rules={[{ required: true }]}>
                    <InputNumber min={0} max={1_000_000} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="Outlier Threshold (Z-score)" name="outlierThreshold" rules={[{ required: true }]}>
                    <InputNumber min={0} max={10} step={0.1} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
              </Row>

              <Space style={{ width: "100%", justifyContent: "flex-end" }}>
                <Button type="primary" htmlType="submit" loading={runLoading} disabled={datasetsLoading}>
                  Run Segmentation
                </Button>
              </Space>
            </Form>
          </Card>
        </Col>

        <Col span={24}>
          {runLoading && (
            <Card>
              <Spin />
            </Card>
          )}

          {runError && (
            <Alert type="error" showIcon message="Segmentation failed" description={runError} style={{ marginBottom: 16 }} />
          )}

          {result && (
            <Space direction="vertical" style={{ width: "100%" }} size={16}>
              <Card title="Overview">
                <Row gutter={[16, 16]}>
                  <Col xs={12} md={6}>
                    <Statistic title="Best K" value={result.model_info?.best_k ?? "-"} />
                  </Col>
                  <Col xs={12} md={6}>
                    <Statistic title="Best Silhouette" value={result.model_info?.best_silhouette_score ?? "-"} precision={4} />
                    {silhouetteHint}
                  </Col>
                  <Col xs={24} md={12}>
                    <Text type="secondary">
                      Dataset: {result.dataset_id || "-"} · reference_date: {result.reference_date || "-"} · window:{" "}
                      {result.time_window_days ?? "-"} days · feature_mode: {result.feature_mode || "-"}
                    </Text>
                    <Divider style={{ margin: "8px 0" }} />
                    <Text type="secondary">features_used: {(result.model_info?.features_used || []).join(", ")}</Text>
                  </Col>
                </Row>

                {result.warnings && result.warnings.length > 0 && (
                  <Alert
                    style={{ marginTop: 12 }}
                    type="warning"
                    showIcon
                    message="Warnings"
                    description={
                      <div>
                        {result.warnings.map((w, idx) => (
                          <div key={idx}>{w}</div>
                        ))}
                      </div>
                    }
                  />
                )}
              </Card>

              {result.feature_definitions && result.feature_definitions.length > 0 && (
                <Card title="Feature Definitions" style={{ borderRadius: 8 }}>
                  <Table<FeatureDefinition>
                    rowKey={(r) => r.name}
                    size="small"
                    columns={featureColumns}
                    dataSource={result.feature_definitions}
                    pagination={false}
                    scroll={{ x: "max-content" }}
                  />
                </Card>
              )}

              {result.category_overview && result.category_overview.length > 0 && (
                <Card title="Category Cleaning (Revenue by Category)" style={{ borderRadius: 8 }}>
                  <Table<CategoryOverviewRow>
                    rowKey={(r) => r.category}
                    size="small"
                    columns={categoryColumns}
                    dataSource={result.category_overview}
                    pagination={false}
                    scroll={{ x: "max-content" }}
                  />
                </Card>
              )}

              {result.season_overview && result.season_overview.length > 0 && (
                <Card title="Season Overview (Revenue by Season)" style={{ borderRadius: 8 }}>
                  <Table<SeasonOverviewRow>
                    rowKey={(r) => r.season}
                    size="small"
                    columns={seasonColumns}
                    dataSource={result.season_overview}
                    pagination={false}
                    scroll={{ x: "max-content" }}
                  />
                </Card>
              )}

              <Row gutter={[16, 16]}>
                <Col xs={24} md={12}>
                  <Card title="Elbow Curve Data (SSE)" style={{ borderRadius: 8 }}>
                    <Table<ElbowPoint>
                      rowKey={(r) => String(r.k)}
                      size="small"
                      columns={elbowColumns}
                      dataSource={result.elbow_curve_data || []}
                      pagination={false}
                    />
                  </Card>
                </Col>
                <Col xs={24} md={12}>
                  <Card title="Data Preview (with cluster labels)" style={{ borderRadius: 8 }}>
                    <Table<Record<string, any>>
                      rowKey={(_, i) => String(i)}
                      size="small"
                      columns={previewColumns}
                      dataSource={result.data_preview || []}
                      pagination={false}
                      scroll={{ x: "max-content" }}
                    />
                  </Card>
                </Col>
              </Row>

              {result.cluster_plot && (
                <Card title="Cluster Scatter Plot (PCA)" style={{ borderRadius: 8 }}>
                  <ClusterScatter
                    plot={result.cluster_plot}
                    labels={clusterLabels}
                    activeClusterId={selectedClusterId}
                    onSelectCluster={(id) => setSelectedClusterId(id)}
                  />
                </Card>
              )}

              <Card title="Cluster Profiles (Centroids)" style={{ borderRadius: 8 }}>
                <Table<ClusterSummary>
                  rowKey={(r) => String(r.cluster_id)}
                  size="small"
                  columns={clusterColumns}
                  dataSource={result.clusters_summary || []}
                  pagination={false}
                  scroll={{ x: "max-content" }}
                  onRow={(record) => ({
                    onClick: () => setSelectedClusterId(record.cluster_id),
                  })}
                />
              </Card>

              {selectedCluster && (
                <Card
                  title={`Cluster Detail: ${selectedCluster.cluster_id}${selectedCluster.label_suggestion ? ` · ${selectedCluster.label_suggestion}` : ""}`}
                  style={{ borderRadius: 8 }}
                  extra={
                    <Select
                      style={{ width: 220 }}
                      value={selectedClusterId ?? undefined}
                      onChange={(v) => setSelectedClusterId(Number(v))}
                      options={(result.clusters_summary || []).map((c) => ({
                        label: `Cluster ${c.cluster_id}${c.label_suggestion ? ` · ${c.label_suggestion}` : ""}`,
                        value: c.cluster_id,
                      }))}
                    />
                  }
                >
                  <Text type="secondary">
                    Size: {formatNumber(selectedCluster.size)} · Share: {selectedCluster.percentage}
                  </Text>
                  <Divider style={{ margin: "12px 0" }} />
                  <Table<{ feature: string; mean: number; std: number }>
                    rowKey={(r) => r.feature}
                    size="small"
                    columns={selectedClusterFeatureCols}
                    dataSource={selectedClusterFeatureRows}
                    pagination={false}
                    scroll={{ x: "max-content" }}
                  />
                </Card>
              )}

              {result.insight_markdown && (
                <Card title="AI Cluster Interpretation (Markdown)" style={{ borderRadius: 8 }}>
                  <MDEditor.Markdown source={result.insight_markdown} />
                </Card>
              )}

              {result.log_transform && (
                <Card title="Log Transform" style={{ borderRadius: 8 }}>
                  <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(result.log_transform, null, 2)}</pre>
                </Card>
              )}
            </Space>
          )}
        </Col>
      </Row>
    </div>
  );
};
