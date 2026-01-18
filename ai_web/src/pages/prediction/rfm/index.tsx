import React, { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Col,
  Divider,
  Drawer,
  Form,
  InputNumber,
  Radio,
  Row,
  Select,
  Slider,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import type { ColumnsType, TablePaginationConfig } from "antd/es/table";
import { PlayCircleOutlined, SettingOutlined, ThunderboltOutlined } from "@ant-design/icons";
import MDEditor from "@uiw/react-md-editor/nohighlight";

import type { Dataset } from "../../../types/dataset";
import type {
  RFMAnalysisData,
  RFMRunRequest,
  RFMSegmentCustomer,
  RFMSegmentDetailResponse,
  RFMSegmentationMethod,
  RFMSegmentSummary,
} from "../../../types/rfm";
import { TOKEN_KEY } from "../../../authProvider";

const { Title, Text } = Typography;

type Size = { width: number; height: number };

function useElementSize<T extends HTMLElement>(): { ref: React.RefCallback<T>; size: Size } {
  const [el, setEl] = useState<T | null>(null);
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });

  const ref = useCallback((node: T | null) => {
    setEl(node);
  }, []);

  useLayoutEffect(() => {
    if (!el) return;

    const update = () => {
      const { width, height } = el.getBoundingClientRect();
      setSize({ width, height });
    };

    update();

    if (typeof ResizeObserver !== "undefined") {
      const ro = new ResizeObserver(() => update());
      ro.observe(el);
      return () => ro.disconnect();
    }

    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [el]);

  return { ref, size };
}

type TreemapItem = {
  key: string;
  label: string;
  value: number;
  color: string;
  raw: RFMSegmentSummary;
};

type TreemapRect = TreemapItem & { x: number; y: number; width: number; height: number };

function computeTreemap(items: TreemapItem[], width: number, height: number, padding = 2): TreemapRect[] {
  const filtered = items.filter((i) => i.value > 0);
  const total = filtered.reduce((acc, i) => acc + i.value, 0);
  if (!total || width <= 0 || height <= 0) return [];

  const area = width * height;
  const normalized = filtered
    .map((i) => ({ ...i, _area: (i.value / total) * area }))
    .sort((a, b) => b._area - a._area);

  const rects: TreemapRect[] = [];

  const worst = (row: Array<(TreemapItem & { _area: number })>, side: number) => {
    const areas = row.map((r) => r._area);
    const sum = areas.reduce((a, v) => a + v, 0);
    const max = Math.max(...areas);
    const min = Math.min(...areas);
    const s2 = sum * sum;
    const side2 = side * side;
    return Math.max((side2 * max) / s2, s2 / (side2 * min));
  };

  const layoutRow = (
    row: Array<(TreemapItem & { _area: number })>,
    x: number,
    y: number,
    w: number,
    h: number
  ) => {
    const rowArea = row.reduce((a, r) => a + r._area, 0);
    const horizontal = w >= h;

    if (horizontal) {
      const rowH = rowArea / w;
      let offsetX = x;
      for (const r of row) {
        const itemW = r._area / rowH;
        rects.push({
          ...r,
          x: offsetX + padding,
          y: y + padding,
          width: Math.max(0, itemW - padding * 2),
          height: Math.max(0, rowH - padding * 2),
        });
        offsetX += itemW;
      }
      return { x, y: y + rowH, w, h: h - rowH };
    }

    const rowW = rowArea / h;
    let offsetY = y;
    for (const r of row) {
      const itemH = r._area / rowW;
      rects.push({
        ...r,
        x: x + padding,
        y: offsetY + padding,
        width: Math.max(0, rowW - padding * 2),
        height: Math.max(0, itemH - padding * 2),
      });
      offsetY += itemH;
    }
    return { x: x + rowW, y, w: w - rowW, h };
  };

  let row: Array<(TreemapItem & { _area: number })> = [];
  let x = 0;
  let y = 0;
  let w = width;
  let h = height;

  const remaining = [...normalized];
  while (remaining.length > 0) {
    const next = remaining[0]!;
    if (row.length === 0) {
      row.push(next);
      remaining.shift();
      continue;
    }

    const side = Math.min(w, h);
    const currentWorst = worst(row, side);
    const nextWorst = worst([...row, next], side);
    if (nextWorst <= currentWorst) {
      row.push(next);
      remaining.shift();
    } else {
      const nextRect = layoutRow(row, x, y, w, h);
      x = nextRect.x;
      y = nextRect.y;
      w = nextRect.w;
      h = nextRect.h;
      row = [];
    }
  }

  if (row.length > 0) {
    layoutRow(row, x, y, w, h);
  }

  return rects;
}

function formatCompactNumber(n: number): string {
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("zh-CN");
}

function formatMoney(n: number): string {
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("zh-CN", { maximumFractionDigits: 2 });
}

function segmentStrategy(segment: string): string {
  switch (segment) {
    case "Champions":
      return "推新品/高客单组合包，叠加会员权益防止流失。";
    case "Loyal Customers":
      return "上新提醒 + 复购激励，提升客单并培育为 Champions。";
    case "Potential Loyalist":
      return "用满减/加价购引导二购，叠加会员/积分体系提升频次。";
    case "Need Attention":
      return "中等活跃：用内容/新品/个性化推荐做唤醒，避免滑向流失。";
    case "Promising":
      return "新近但低频：首购后 7-21 天重点二次触达，培养复购习惯。";
    case "Can't Lose Them":
      return "高频但久未购买：优先人工/高触达召回（专属券/关怀），避免高价值流失。";
    case "At Risk":
      return "优先召回：限时券/关怀触达，结合偏好做个性化唤醒。";
    case "New Customers":
      return "首购后 7-14 天内二次触达，推动二购形成习惯。";
    case "Hibernating":
      return "低成本触达 + 低门槛优惠，测试可唤醒比例。";
    default:
      return "分层再营销（推荐/满减），推动向 Loyal/Champions 迁移。";
  }
}

const CONFIG_STORAGE_PREFIX = "rfm_engine:config:";

const defaultConfig: RFMRunRequest = {
  time_window_days: 365,
  scoring_scale: 5,
  segmentation_method: "quantiles",
  weights: { r: 1, f: 1, m: 1 },
};

export const RFMEngine: React.FC = () => {
  const [configForm] = Form.useForm();

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  const [datasetId, setDatasetId] = useState<string>();

  const [config, setConfig] = useState<RFMRunRequest>(defaultConfig);
  const [configOpen, setConfigOpen] = useState(false);

  const [analysis, setAnalysis] = useState<RFMAnalysisData | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const [activeSegment, setActiveSegment] = useState<string | null>(null);
  const [segmentOpen, setSegmentOpen] = useState(false);
  const [segmentLoading, setSegmentLoading] = useState(false);
  const [segmentError, setSegmentError] = useState<string | null>(null);
  const [segmentData, setSegmentData] = useState<RFMSegmentDetailResponse | null>(null);

  const [visualMode, setVisualMode] = useState<"matrix" | "treemap">("matrix");

  const { ref: treemapRef, size: treemapSize } = useElementSize<HTMLDivElement>();

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setDatasetsLoading(true);
      setDatasetsError(null);
      try {
        const token = localStorage.getItem(TOKEN_KEY);
        const headers: Record<string, string> = {};
        if (token) headers.Authorization = `Bearer ${token}`;
        const res = await fetch("/api/v1/datasets", { headers });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const json = await res.json();
        if (!json?.success) {
          throw new Error(json?.error || json?.message || "加载数据集失败");
        }
        if (!cancelled) setDatasets(json.data || []);
      } catch (e) {
        if (!cancelled) setDatasetsError(e instanceof Error ? e.message : "加载数据集失败");
      } finally {
        if (!cancelled) setDatasetsLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!datasetId) return;

    // Load last saved config per dataset (fallback to defaults).
    const raw = localStorage.getItem(`${CONFIG_STORAGE_PREFIX}${datasetId}`);
    if (!raw) {
      setConfig(defaultConfig);
      return;
    }
    try {
      const parsed = JSON.parse(raw) as Partial<RFMRunRequest>;
      setConfig({
        time_window_days: parsed.time_window_days ?? defaultConfig.time_window_days,
        scoring_scale: parsed.scoring_scale ?? defaultConfig.scoring_scale,
        segmentation_method: (parsed.segmentation_method as RFMSegmentationMethod) ?? defaultConfig.segmentation_method,
        weights: {
          r: parsed.weights?.r ?? defaultConfig.weights.r,
          f: parsed.weights?.f ?? defaultConfig.weights.f,
          m: parsed.weights?.m ?? defaultConfig.weights.m,
        },
      });
    } catch {
      setConfig(defaultConfig);
    }
  }, [datasetId]);

  useEffect(() => {
    if (!configOpen) return;
    configForm.setFieldsValue({
      time_window_days: config.time_window_days,
      scoring_scale: config.scoring_scale,
      segmentation_method: config.segmentation_method,
      w_r: config.weights.r,
      w_f: config.weights.f,
      w_m: config.weights.m,
    });
  }, [config, configForm, configOpen]);

  const saveConfig = (next: RFMRunRequest) => {
    if (!datasetId) return;
    localStorage.setItem(`${CONFIG_STORAGE_PREFIX}${datasetId}`, JSON.stringify(next));
  };

  const runAnalysis = async () => {
    if (!datasetId) {
      message.warning("请先选择一个数据集");
      return;
    }
    setAnalysisLoading(true);
    setAnalysisError(null);
    setAnalysis(null);
    setActiveSegment(null);
    setSegmentOpen(false);
    setSegmentData(null);
    try {
      saveConfig(config);

      const token = localStorage.getItem(TOKEN_KEY);
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`/api/v1/datasets/${encodeURIComponent(datasetId)}/rfm`, {
        method: "POST",
        headers,
        body: JSON.stringify(config),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const json = await res.json();
      if (!json?.success || !json?.data) {
        throw new Error(json?.error || json?.message || "运行分析失败");
      }
      setAnalysis(json.data as RFMAnalysisData);
    } catch (e) {
      setAnalysisError(e instanceof Error ? e.message : "运行分析失败");
    } finally {
      setAnalysisLoading(false);
    }
  };

  const fetchSegmentDetail = async (segmentName: string, pagination: { page: number; pageSize: number }) => {
    if (!analysis || !datasetId) return;
    setSegmentLoading(true);
    setSegmentError(null);
    try {
      const token = localStorage.getItem(TOKEN_KEY);
      const headers: Record<string, string> = {};
      if (token) headers.Authorization = `Bearer ${token}`;

      const url =
        `/api/v1/datasets/${encodeURIComponent(datasetId)}` +
        `/rfm/${encodeURIComponent(analysis.analysis_id)}` +
        `/segment/${encodeURIComponent(segmentName)}` +
        `?page=${pagination.page}&pageSize=${pagination.pageSize}`;

      const res = await fetch(url, { headers });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const json = await res.json();
      if (!json?.success || !json?.data) {
        throw new Error(json?.error || json?.message || "加载人群详情失败");
      }
      setSegmentData(json.data as RFMSegmentDetailResponse);
    } catch (e) {
      setSegmentError(e instanceof Error ? e.message : "加载人群详情失败");
      setSegmentData(null);
    } finally {
      setSegmentLoading(false);
    }
  };

  const openSegment = (segmentName: string) => {
    setActiveSegment(segmentName);
    setSegmentOpen(true);
    fetchSegmentDetail(segmentName, { page: 1, pageSize: 20 });
  };

  const treemapItems: TreemapItem[] = useMemo(() => {
    if (!analysis) return [];
    return analysis.segments.map((s) => ({
      key: s.segment,
      label: s.segment,
      value: s.count,
      color: s.color,
      raw: s,
    }));
  }, [analysis]);

  const treemapRects = useMemo(() => {
    if (!analysis) return [];
    const w = treemapSize.width;
    const h = treemapSize.height;
    if (!w || !h) return [];
    return computeTreemap(treemapItems, w, h);
  }, [analysis, treemapItems, treemapSize.height, treemapSize.width]);

  const segmentColumns: ColumnsType<RFMSegmentSummary> = [
    {
      title: "人群",
      dataIndex: "segment",
      key: "segment",
      render: (v: string, r) => (
        <Space size={8}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: r.color, display: "inline-block" }} />
          <Text strong>{v}</Text>
        </Space>
      ),
    },
    { title: "人数", dataIndex: "count", key: "count", render: (v: number) => formatCompactNumber(v) },
    { title: "占比", dataIndex: "share_pct", key: "share_pct", render: (v: number) => `${v.toFixed(1)}%` },
    { title: "营收", dataIndex: "revenue", key: "revenue", render: (v: number) => formatMoney(v) },
    { title: "营收占比", dataIndex: "revenue_share_pct", key: "revenue_share_pct", render: (v: number) => `${v.toFixed(1)}%` },
    {
      title: "Avg R(天)",
      dataIndex: "avg_recency_days",
      key: "avg_recency_days",
      render: (v: number) => v.toFixed(1),
    },
    {
      title: "Avg F",
      dataIndex: "avg_frequency",
      key: "avg_frequency",
      render: (v: number) => v.toFixed(1),
    },
    {
      title: "Avg M",
      dataIndex: "avg_monetary",
      key: "avg_monetary",
      render: (v: number) => formatMoney(v),
    },
    { title: "Avg RFM", dataIndex: "avg_rfm_score", key: "avg_rfm_score", render: (v: number) => v.toFixed(2) },
  ];

  const customerColumns: ColumnsType<RFMSegmentCustomer> = [
    { title: "user_id", dataIndex: "user_id", key: "user_id", width: 120 },
    { title: "R(天)", dataIndex: "recency_days", key: "recency_days", width: 90, sorter: true },
    { title: "F", dataIndex: "frequency", key: "frequency", width: 80, sorter: true },
    { title: "M", dataIndex: "monetary", key: "monetary", width: 120, render: (v: number) => formatMoney(v), sorter: true },
    { title: "R", dataIndex: "r_score", key: "r_score", width: 70 },
    { title: "F", dataIndex: "f_score", key: "f_score", width: 70 },
    { title: "M", dataIndex: "m_score", key: "m_score", width: 70 },
    { title: "RFM", dataIndex: "rfm_score", key: "rfm_score", width: 90, render: (v: number) => v.toFixed(2) },
  ];

  const scoreChart = (title: string, counts: Record<string, number>, scale: number) => {
    const values = Array.from({ length: scale }, (_, idx) => counts[String(idx + 1)] || 0);
    const max = Math.max(...values, 0);
    return (
      <Card size="small" title={title} style={{ borderRadius: 6 }}>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 8, height: 90 }}>
          {values.map((v, i) => {
            const h = max > 0 ? (v / max) * 70 : 0;
            return (
              <div key={i} style={{ textAlign: "center", width: 28 }}>
                <div
                  style={{
                    height: 70,
                    display: "flex",
                    alignItems: "flex-end",
                    justifyContent: "center",
                  }}
                >
                  <div style={{ width: 18, height: h, background: "#1D70B8", borderRadius: 3 }} />
                </div>
                <div style={{ fontSize: 12, color: "#64748B", marginTop: 6 }}>{i + 1}</div>
                <div style={{ fontSize: 11, color: "#94A3B8" }}>{v}</div>
              </div>
            );
          })}
        </div>
      </Card>
    );
  };

  const configSummary = useMemo(() => {
    const w = config.weights;
    return `窗口 ${config.time_window_days} 天 · 评分 ${config.scoring_scale} · ${config.segmentation_method} · 权重 ${w.r}:${w.f}:${w.m}`;
  }, [config]);

  const matrixCellMap = useMemo(() => {
    const map = new Map<string, any>();
    if (!analysis?.matrix?.cells) return map;
    for (const c of analysis.matrix.cells) {
      map.set(`${c.r_level}:${c.f_level}`, c);
    }
    return map;
  }, [analysis]);

  const softBg = (hex: string, alphaHex: string) => {
    if (typeof hex === "string" && hex.startsWith("#") && hex.length === 7) return `${hex}${alphaHex}`;
    return hex;
  };

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <Card style={{ borderRadius: 8, borderTop: "3px solid #1D70B8" }}>
          <Space direction="vertical" size={12} style={{ width: "100%" }}>
            <Space align="center" style={{ justifyContent: "space-between", width: "100%" }}>
              <div>
                <Title level={3} style={{ margin: 0, color: "#0F3460" }}>
                  RFM 精准引擎
                </Title>
                <Text type="secondary">{configSummary}</Text>
              </div>
              <Space>
                <Button icon={<SettingOutlined />} onClick={() => setConfigOpen(true)}>
                  配置参数
                </Button>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={runAnalysis}
                  disabled={!datasetId || datasetsLoading}
                  loading={analysisLoading}
                >
                  运行分析
                </Button>
              </Space>
            </Space>

            <Space wrap>
              <Text strong>选择数据集：</Text>
              <Select
                style={{ minWidth: 320 }}
                placeholder="请选择一个数据集"
                loading={datasetsLoading}
                value={datasetId}
                onChange={(v) => {
                  setDatasetId(v);
                  setAnalysis(null);
                  setAnalysisError(null);
                }}
                options={datasets.map((d) => ({
                  value: d.id,
                  label: `${d.name} (${d.id})`,
                }))}
              />
              {datasetsError && <Text type="danger">{datasetsError}</Text>}
            </Space>
          </Space>
        </Card>

        {analysisError && <Alert type="error" showIcon message="运行失败" description={analysisError} />}

        {analysisLoading && (
          <Card style={{ borderRadius: 8 }}>
            <div style={{ textAlign: "center", padding: 24 }}>
              <Spin />
              <div style={{ marginTop: 12, color: "#64748B" }}>正在计算 RFM...</div>
            </div>
          </Card>
        )}

        {analysis && (
          <>
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} md={6}>
                <Card style={{ borderRadius: 8 }}>
                  <Statistic title="用户数" value={analysis.overview.total_users} />
                </Card>
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Card style={{ borderRadius: 8 }}>
                  <Statistic title="订单数" value={analysis.overview.total_orders} />
                </Card>
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Card style={{ borderRadius: 8 }}>
                  <Statistic title="总营收" value={analysis.overview.total_revenue} precision={2} />
                </Card>
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Card style={{ borderRadius: 8 }}>
                  <Statistic
                    title="环比营收"
                    value={analysis.mom?.total_revenue_change_pct ?? 0}
                    precision={1}
                    suffix="%"
                    valueStyle={{
                      color:
                        (analysis.mom?.total_revenue_change_pct ?? 0) >= 0 ? "#1D70B8" : "#FF7043",
                    }}
                  />
                </Card>
              </Col>
            </Row>

            <Row gutter={[16, 16]}>
              <Col xs={24} lg={14}>
                <Card
                  title={visualMode === "matrix" ? "经典 RFM 矩阵（9 宫格）" : "人群分布（Treemap）"}
                  style={{ borderRadius: 8 }}
                  extra={
                    <Radio.Group
                      value={visualMode}
                      onChange={(e) => setVisualMode(e.target.value as "matrix" | "treemap")}
                      optionType="button"
                      buttonStyle="solid"
                    >
                      <Radio.Button value="matrix">矩阵</Radio.Button>
                      <Radio.Button value="treemap">Treemap</Radio.Button>
                    </Radio.Group>
                  }
                >
                  {visualMode === "matrix" ? (
                    <>
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "110px repeat(3, 1fr)",
                          gap: 10,
                        }}
                      >
                        <div />
                        {analysis.matrix.cols.map((c) => (
                          <div
                            key={c.id}
                            style={{
                              fontWeight: 700,
                              fontSize: 12,
                              color: "#0F3460",
                              textAlign: "center",
                              padding: "6px 8px",
                              background: "rgba(148, 163, 184, 0.08)",
                              borderRadius: 6,
                            }}
                          >
                            {c.label}
                          </div>
                        ))}

                        {analysis.matrix.rows.map((r) => (
                          <React.Fragment key={r.id}>
                            <div
                              style={{
                                fontWeight: 700,
                                fontSize: 12,
                                color: "#0F3460",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                padding: "6px 8px",
                                background: "rgba(148, 163, 184, 0.08)",
                                borderRadius: 6,
                              }}
                            >
                              {r.label}
                            </div>
                            {analysis.matrix.cols.map((c) => {
                              const cell = matrixCellMap.get(`${r.id}:${c.id}`);
                              if (!cell) return <div key={`${r.id}:${c.id}`} />;
                              const clickable = cell.count > 0;
                              const active = activeSegment === cell.segment;
                              return (
                                <div
                                  key={`${r.id}:${c.id}`}
                                  role={clickable ? "button" : undefined}
                                  tabIndex={clickable ? 0 : undefined}
                                  onClick={clickable ? () => openSegment(cell.segment) : undefined}
                                  onKeyDown={(e) => {
                                    if (!clickable) return;
                                    if (e.key === "Enter") openSegment(cell.segment);
                                  }}
                                  title={`${cell.segment}\n人数: ${formatCompactNumber(cell.count)} (${cell.share_pct.toFixed(
                                    1
                                  )}%)\n营收占比: ${cell.revenue_share_pct.toFixed(1)}%\nAvg M: ${formatMoney(
                                    cell.avg_monetary
                                  )}\nAvg RFM: ${cell.avg_rfm_score.toFixed(2)}`}
                                  style={{
                                    cursor: clickable ? "pointer" : "not-allowed",
                                    userSelect: "none",
                                    borderRadius: 8,
                                    padding: 10,
                                    minHeight: 92,
                                    background: softBg(cell.color, "1A"),
                                    border: `1px solid ${softBg(cell.color, "66")}`,
                                    opacity: clickable ? 1 : 0.55,
                                    outline: active ? "2px solid rgba(15, 52, 96, 0.5)" : "none",
                                  }}
                                >
                                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                    <span
                                      style={{
                                        width: 10,
                                        height: 10,
                                        borderRadius: 2,
                                        background: cell.color,
                                        display: "inline-block",
                                      }}
                                    />
                                    <Text strong style={{ color: "#0B1220" }}>
                                      {cell.segment}
                                    </Text>
                                  </div>
                                  <div style={{ marginTop: 6, fontSize: 12, color: "#334155" }}>
                                    {formatCompactNumber(cell.count)} 人 · {cell.share_pct.toFixed(1)}%
                                  </div>
                                  <div style={{ marginTop: 2, fontSize: 12, color: "#334155" }}>
                                    Avg M {formatMoney(cell.avg_monetary)}
                                  </div>
                                  <div style={{ marginTop: 2, fontSize: 12, color: "#334155" }}>
                                    Avg RFM {cell.avg_rfm_score.toFixed(2)}
                                  </div>
                                </div>
                              );
                            })}
                          </React.Fragment>
                        ))}
                      </div>

                      <div style={{ marginTop: 12, color: "#64748B", fontSize: 12 }}>
                        规则：score ≤ {analysis.matrix.thresholds.low_end} 为 Low；score ≥ {analysis.matrix.thresholds.high_start} 为 High；
                        其余为 Mid（基于 1-{analysis.matrix.thresholds.scale} 分制）。
                        <span style={{ marginLeft: 8 }}>点击格子可查看该人群明细；M 维度体现在 Avg M / 营收贡献。</span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div
                        ref={treemapRef}
                        style={{
                          height: 320,
                          width: "100%",
                          position: "relative",
                          background: "rgba(148, 163, 184, 0.08)",
                          borderRadius: 8,
                          overflow: "hidden",
                        }}
                      >
                        {analysis.segments.length > 0 && treemapRects.length === 0 && (
                          <div
                            style={{
                              position: "absolute",
                              inset: 0,
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              color: "#64748B",
                            }}
                          >
                            正在渲染 Treemap...
                          </div>
                        )}
                        {treemapRects.map((r) => {
                          const showLabel = r.width > 90 && r.height > 42;
                          return (
                            <div
                              key={r.key}
                              role="button"
                              tabIndex={0}
                              onClick={() => openSegment(r.raw.segment)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") openSegment(r.raw.segment);
                              }}
                              style={{
                                position: "absolute",
                                left: r.x,
                                top: r.y,
                                width: r.width,
                                height: r.height,
                                background: r.color,
                                borderRadius: 6,
                                cursor: "pointer",
                                padding: 10,
                                boxSizing: "border-box",
                                opacity: activeSegment === r.raw.segment ? 0.9 : 1,
                                outline:
                                  activeSegment === r.raw.segment ? "2px solid rgba(15, 52, 96, 0.5)" : "none",
                              }}
                              title={`${r.raw.segment}\n人数: ${formatCompactNumber(r.raw.count)}\nAvg M: ${formatMoney(
                                r.raw.avg_monetary
                              )}`}
                            >
                              {showLabel && (
                                <div style={{ color: "#0B1220" }}>
                                  <div style={{ fontWeight: 700, fontSize: 14 }}>{r.raw.segment}</div>
                                  <div style={{ marginTop: 4, fontSize: 12 }}>
                                    {formatCompactNumber(r.raw.count)} 人 · Avg M {formatMoney(r.raw.avg_monetary)}
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                      <div style={{ marginTop: 12, color: "#64748B", fontSize: 12 }}>
                        Hover 查看指标，Click 选中人群。
                      </div>
                    </>
                  )}
                </Card>
              </Col>
              <Col xs={24} lg={10}>
                <Card title="评分分布（1-N）" style={{ borderRadius: 8 }}>
                  <Row gutter={[12, 12]}>
                    <Col span={24}>
                      {scoreChart("R Score 分布", analysis.score_distributions.r, analysis.scoring_scale)}
                    </Col>
                    <Col span={24}>
                      {scoreChart("F Score 分布", analysis.score_distributions.f, analysis.scoring_scale)}
                    </Col>
                    <Col span={24}>
                      {scoreChart("M Score 分布", analysis.score_distributions.m, analysis.scoring_scale)}
                    </Col>
                  </Row>
                </Card>
              </Col>
            </Row>

            <Card
              title="AI 洞察（Markdown）"
              style={{ borderRadius: 8 }}
              extra={
                <Tag icon={<ThunderboltOutlined />} color="blue">
                  Next Best Action
                </Tag>
              }
            >
              <MDEditor.Markdown source={analysis.insight_markdown} />
            </Card>

            <Card title="人群明细" style={{ borderRadius: 8 }}>
              <Table<RFMSegmentSummary>
                rowKey={(r) => r.segment}
                size="small"
                columns={segmentColumns}
                dataSource={analysis.segments}
                pagination={false}
                onRow={(record) => ({
                  onClick: () => openSegment(record.segment),
                })}
              />
            </Card>
          </>
        )}
      </Space>

      <Drawer
        title="参数配置"
        open={configOpen}
        onClose={() => setConfigOpen(false)}
        width={420}
        destroyOnClose
      >
        <Form
          form={configForm}
          layout="vertical"
          onValuesChange={(_, values) => {
            const next: RFMRunRequest = {
              time_window_days: Number(values.time_window_days ?? defaultConfig.time_window_days),
              scoring_scale: Number(values.scoring_scale ?? defaultConfig.scoring_scale),
              segmentation_method: values.segmentation_method as RFMSegmentationMethod,
              weights: {
                r: Number(values.w_r ?? defaultConfig.weights.r),
                f: Number(values.w_f ?? defaultConfig.weights.f),
                m: Number(values.w_m ?? defaultConfig.weights.m),
              },
            };
            setConfig(next);
          }}
        >
          <Form.Item label="Time Window（天）" name="time_window_days">
            <InputNumber min={1} max={3650} style={{ width: "100%" }} />
          </Form.Item>

          <Form.Item label="Scoring Scale（3-10）" name="scoring_scale">
            <InputNumber min={3} max={10} style={{ width: "100%" }} />
          </Form.Item>

          <Form.Item label="Segmentation Method" name="segmentation_method">
            <Radio.Group>
              <Radio value="quantiles">Quantiles（等频分箱）</Radio>
              <Radio value="kmeans">K-Means（自然聚类）</Radio>
            </Radio.Group>
          </Form.Item>

          <Divider />

          <Title level={5} style={{ marginTop: 0 }}>
            Weights（权重）
          </Title>

          <Form.Item label={`R 权重：${config.weights.r.toFixed(1)}`} name="w_r">
            <Slider min={0.5} max={3} step={0.1} />
          </Form.Item>
          <Form.Item label={`F 权重：${config.weights.f.toFixed(1)}`} name="w_f">
            <Slider min={0.5} max={3} step={0.1} />
          </Form.Item>
          <Form.Item label={`M 权重：${config.weights.m.toFixed(1)}`} name="w_m">
            <Slider min={0.5} max={3} step={0.1} />
          </Form.Item>

          <Space style={{ width: "100%", justifyContent: "space-between" }}>
            <Button
              onClick={() => {
                setConfig(defaultConfig);
                configForm.setFieldsValue({
                  time_window_days: defaultConfig.time_window_days,
                  scoring_scale: defaultConfig.scoring_scale,
                  segmentation_method: defaultConfig.segmentation_method,
                  w_r: defaultConfig.weights.r,
                  w_f: defaultConfig.weights.f,
                  w_m: defaultConfig.weights.m,
                });
                message.success("已恢复默认配置");
              }}
            >
              恢复默认
            </Button>
            <Button
              type="primary"
              onClick={() => {
                saveConfig(config);
                message.success("配置已保存");
                setConfigOpen(false);
              }}
              disabled={!datasetId}
            >
              保存配置
            </Button>
          </Space>
          {!datasetId && (
            <Alert
              style={{ marginTop: 16 }}
              type="info"
              showIcon
              message="提示"
              description="选择数据集后，将按数据集维度保存配置。"
            />
          )}
        </Form>
      </Drawer>

      <Drawer
        title={activeSegment ? `人群详情：${activeSegment}` : "人群详情"}
        open={segmentOpen}
        onClose={() => setSegmentOpen(false)}
        width={780}
      >
        {activeSegment && (
          <>
            <Alert
              type="info"
              showIcon
              message="推荐策略"
              description={segmentStrategy(activeSegment)}
              style={{ marginBottom: 16 }}
            />
            <Space style={{ marginBottom: 12 }}>
              <Button
                type="primary"
                icon={<ThunderboltOutlined />}
                onClick={() => message.info("“激活策略”已触发（当前为 Demo，未接入 CRM/MA）")}
              >
                激活策略
              </Button>
            </Space>
          </>
        )}

        {segmentError && <Alert type="error" showIcon message="加载失败" description={segmentError} />}
        {segmentLoading && (
          <div style={{ textAlign: "center", padding: 24 }}>
            <Spin />
          </div>
        )}
        {segmentData && (
          <Table<RFMSegmentCustomer>
            rowKey={(r) => `${r.user_id}`}
            size="small"
            columns={customerColumns}
            dataSource={segmentData.records}
            pagination={{
              current: segmentData.page,
              pageSize: segmentData.pageSize,
              total: segmentData.total,
              showSizeChanger: true,
              pageSizeOptions: [10, 20, 50, 100],
              onChange: (page, pageSize) => {
                if (activeSegment) fetchSegmentDetail(activeSegment, { page, pageSize });
              },
            }}
          />
        )}
      </Drawer>
    </div>
  );
};
