import React, { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Progress,
  Radio,
  Row,
  Select,
  Slider,
  Space,
  Spin,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
  message,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import { PlayCircleOutlined } from "@ant-design/icons";
import MDEditor from "@uiw/react-md-editor/nohighlight";
import { TOKEN_KEY } from "../../../authProvider";

const { Title, Text } = Typography;

const DATASET_ID = "store_sales_data";

type Size = { width: number; height: number };
function useElementSize<T extends HTMLElement>(): { ref: React.RefCallback<T>; size: Size } {
  const [el, setEl] = useState<T | null>(null);
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });

  const ref = useCallback((node: T | null) => setEl(node), []);

  useLayoutEffect(() => {
    if (!el) return;
    const update = () => {
      const r = el.getBoundingClientRect();
      setSize({ width: r.width, height: r.height });
    };
    update();
    if (typeof ResizeObserver !== "undefined") {
      const ro = new ResizeObserver(update);
      ro.observe(el);
      return () => ro.disconnect();
    }
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [el]);

  return { ref, size };
}

type ProductOption = {
  product_name: string;
  category: string;
  sub_category: string;
  total_orders: number;
  total_revenue: number;
};

type AllocationCityRow = {
  city: string;
  region: string;
  lat: number;
  lon: number;
  margin_tier: "High" | "Med" | "Low";
  margin_rate: number;
  forecast_demand: number;
  allocated: number;
  fill_rate: number;
  status: "Sufficient" | "Adjusted" | "Dropped";
};

type TrendPoint = {
  date: string;
  forecast_demand: number;
  planned_inventory: number;
};

type StockAllocationResult = {
  dataset_id: string;
  model_version: string;
  reference_date: string;
  product: {
    name: string;
    category: string;
    sub_category: string;
    forecast_scope: string;
    product_share_within_sub_category: number;
  };
  supply: {
    available_inventory: number;
    inbound: { days: number; quantity: number };
    gap_pct: number;
    supply_total: number;
  };
  kpis: {
    profit: number;
    profit_delta_vs_profit_max: number;
    fill_rate: number;
    lost_sales: number;
    risk_store_count: number;
  };
  ai_explanation_markdown?: string;
  ai_explanation_source?: "ai" | "fallback" | "none";
  ai_explanation_error?: string | null;
  visuals: {
    trend: { points: TrendPoint[]; out_of_stock_date: string | null };
  };
  cities: AllocationCityRow[];
};

function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function formatInt(n: number): string {
  if (!Number.isFinite(n)) return "-";
  return Math.round(n).toLocaleString("en-US");
}

function formatMoney(n: number): string {
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function fillColor(fr: number): string {
  if (!Number.isFinite(fr)) return "#94A3B8";
  if (fr >= 0.8) return "#4CAF50";
  if (fr >= 0.3) return "#D4E157";
  return "#FF7043";
}

function statusTag(status: AllocationCityRow["status"]) {
  if (status === "Sufficient") return <Tag color="green">Sufficient</Tag>;
  if (status === "Adjusted") return <Tag color="orange">Adjusted</Tag>;
  return <Tag color="red">Dropped</Tag>;
}

function marginTag(tier: AllocationCityRow["margin_tier"]) {
  if (tier === "High") return <Tag color="blue">High</Tag>;
  if (tier === "Low") return <Tag color="volcano">Low</Tag>;
  return <Tag color="gold">Med</Tag>;
}

function MapView({
  cities,
  width,
  height,
}: {
  cities: AllocationCityRow[];
  width: number;
  height: number;
}) {
  const w = Math.max(320, Math.floor(width || 0));
  const h = Math.max(240, Math.floor(height || 0));
  const padding = 14;

  // US bounds (approx)
  const minLon = -124;
  const maxLon = -67;
  const minLat = 25;
  const maxLat = 49;

  const maxDemand = Math.max(...cities.map((c) => c.forecast_demand), 1);

  const xFor = (lon: number) =>
    padding + ((lon - minLon) / (maxLon - minLon)) * (w - padding * 2);
  const yFor = (lat: number) =>
    padding + ((maxLat - lat) / (maxLat - minLat)) * (h - padding * 2);

  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <rect x={0} y={0} width={w} height={h} fill="rgba(148, 163, 184, 0.08)" rx={8} />
      {cities.map((c) => {
        const r = 4 + Math.sqrt(c.forecast_demand / maxDemand) * 16;
        const x = xFor(c.lon);
        const y = yFor(c.lat);
        const color = fillColor(c.fill_rate);
        return (
          <g key={c.city}>
            <circle cx={x} cy={y} r={r} fill={color} opacity={0.85} stroke="rgba(15, 52, 96, 0.35)" />
            <title>
              {c.city}
              {"\n"}Demand: {formatInt(c.forecast_demand)}
              {"\n"}Allocated: {formatInt(c.allocated)}
              {"\n"}Fill rate: {(c.fill_rate * 100).toFixed(0)}%
            </title>
          </g>
        );
      })}
    </svg>
  );
}

function BarChartView({
  cities,
  width,
  height,
}: {
  cities: AllocationCityRow[];
  width: number;
  height: number;
}) {
  const w = Math.max(320, Math.floor(width || 0));
  const h = Math.max(240, Math.floor(height || 0));
  const padL = 30;
  const padR = 14;
  const padT = 14;
  const padB = 40;

  const data = cities.slice(0, 12);
  const maxV = Math.max(...data.flatMap((c) => [c.forecast_demand, c.allocated]), 1);
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;
  const band = innerW / Math.max(1, data.length);
  const barW = Math.max(6, band * 0.32);
  const x0 = (i: number) => padL + i * band + band / 2;
  const yFor = (v: number) => padT + (1 - v / maxV) * innerH;

  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <rect x={0} y={0} width={w} height={h} fill="rgba(148, 163, 184, 0.08)" rx={8} />
      {data.map((c, i) => {
        const x = x0(i);
        const yD = yFor(c.forecast_demand);
        const yA = yFor(c.allocated);
        const hD = padT + innerH - yD;
        const hA = padT + innerH - yA;
        return (
          <g key={c.city}>
            <rect x={x - barW - 2} y={yD} width={barW} height={hD} fill="#94A3B8" opacity={0.75} />
            <rect x={x + 2} y={yA} width={barW} height={hA} fill="#1D70B8" opacity={0.9} />
            <text
              x={x}
              y={padT + innerH + 22}
              textAnchor="middle"
              fontSize={10}
              fill="#334155"
              transform={`rotate(20 ${x} ${padT + innerH + 22})`}
            >
              {c.city.length > 10 ? `${c.city.slice(0, 10)}…` : c.city}
            </text>
            <title>
              {c.city}
              {"\n"}Demand: {formatInt(c.forecast_demand)}
              {"\n"}Allocated: {formatInt(c.allocated)}
            </title>
          </g>
        );
      })}
      <text x={padL} y={padT + 10} fontSize={11} fill="#64748B">
        Demand (gray) vs Allocated (blue)
      </text>
    </svg>
  );
}

function TrendView({
  points,
  outOfStockDate,
  width,
  height,
}: {
  points: TrendPoint[];
  outOfStockDate: string | null;
  width: number;
  height: number;
}) {
  const w = Math.max(320, Math.floor(width || 0));
  const h = Math.max(240, Math.floor(height || 0));
  const padL = 34;
  const padR = 14;
  const padT = 14;
  const padB = 30;

  const maxV = Math.max(...points.flatMap((p) => [p.forecast_demand, p.planned_inventory]), 1);
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;
  const xFor = (i: number) => padL + (i / Math.max(1, points.length - 1)) * innerW;
  const yFor = (v: number) => padT + (1 - v / maxV) * innerH;

  const linePath = (getVal: (p: TrendPoint) => number) => {
    let d = "";
    points.forEach((p, i) => {
      const x = xFor(i);
      const y = yFor(getVal(p));
      d += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
    });
    return d;
  };

  const oosIdx = outOfStockDate ? points.findIndex((p) => p.date === outOfStockDate) : -1;
  const oosX = oosIdx >= 0 ? xFor(oosIdx) : null;

  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <rect x={0} y={0} width={w} height={h} fill="rgba(148, 163, 184, 0.08)" rx={8} />
      {oosX !== null && (
        <line x1={oosX} x2={oosX} y1={padT} y2={padT + innerH} stroke="#FF7043" strokeDasharray="6 4" />
      )}
      <path d={linePath((p) => p.forecast_demand)} fill="none" stroke="#94A3B8" strokeWidth={2} />
      <path d={linePath((p) => p.planned_inventory)} fill="none" stroke="#1D70B8" strokeWidth={2} />
      <text x={padL} y={padT + 10} fontSize={11} fill="#64748B">
        7-day Forecast Demand (gray) vs Planned Inventory (blue)
      </text>
      {oosX !== null && (
        <text x={Math.min(w - 6, oosX + 6)} y={padT + 22} fontSize={11} fill="#FF7043">
          Out of stock
        </text>
      )}
    </svg>
  );
}

export const InventoryAllocation: React.FC = () => {
  const [products, setProducts] = useState<ProductOption[]>([]);
  const [productsLoading, setProductsLoading] = useState(false);
  const [productsError, setProductsError] = useState<string | null>(null);

  const [productName, setProductName] = useState<string>();

  const [objective, setObjective] = useState<"profit" | "fairness" | "strategic">("profit");
  const [vipNY, setVipNY] = useState(true);
  const [minShipment, setMinShipment] = useState(true);
  const [minFillRatePct, setMinFillRatePct] = useState(30);

  const [demandShockPct, setDemandShockPct] = useState(0);
  const [inventoryShrinkPct, setInventoryShrinkPct] = useState(0);

  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<StockAllocationResult | null>(null);

  const { ref: mapRef, size: mapSize } = useElementSize<HTMLDivElement>();
  const { ref: chartRef, size: chartSize } = useElementSize<HTMLDivElement>();
  const { ref: trendRef, size: trendSize } = useElementSize<HTMLDivElement>();

  const totalDemand = useMemo(() => {
    if (!result) return 0;
    return result.cities.reduce((acc, c) => acc + (Number(c.forecast_demand) || 0), 0);
  }, [result]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setProductsLoading(true);
      setProductsError(null);
      try {
        const res = await fetch(`/api/v1/datasets/${encodeURIComponent(DATASET_ID)}/stock-allocation/products?limit=80`, {
          headers: getAuthHeaders(),
        });
        const json = await res.json().catch(() => ({}));
        if (!res.ok || !json?.success) throw new Error(json?.detail || json?.error || `HTTP ${res.status}`);
        const items = (json.data || []) as ProductOption[];
        if (!cancelled) {
          setProducts(items);
          const preferred = items.find((p) => p.product_name === "Staple envelope")?.product_name;
          setProductName(preferred || items[0]?.product_name);
        }
      } catch (e: any) {
        if (!cancelled) setProductsError(e?.message || "Failed to load products");
      } finally {
        if (!cancelled) setProductsLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const run = async () => {
    if (!productName) {
      message.warning("Please select a product first.");
      return;
    }
    setRunning(true);
    setRunError(null);
    try {
      const body = {
        product_name: productName,
        objective,
        enable_vip_new_york: vipNY,
        enable_min_shipment: minShipment,
        min_fill_rate: minFillRatePct / 100,
        demand_shock_pct: demandShockPct,
        inventory_shrink_pct: inventoryShrinkPct,
        include_ai_explanation: true,
      };
      const res = await fetch(`/api/v1/datasets/${encodeURIComponent(DATASET_ID)}/stock-allocation/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() } as any,
        body: JSON.stringify(body),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok || !json?.success) throw new Error(json?.detail || json?.error || `HTTP ${res.status}`);
      setResult(json.data as StockAllocationResult);
    } catch (e: any) {
      setRunError(e?.message || "Failed to run optimization");
      setResult(null);
    } finally {
      setRunning(false);
    }
  };

  // Auto-run once after first product is selected for a ready-to-demo screen.
  useEffect(() => {
    if (!productName) return;
    if (result) return;
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [productName]);

  const cityColumns: ColumnsType<AllocationCityRow> = [
    { title: "City", dataIndex: "city", key: "city", fixed: "left", width: 180 },
    { title: "Demand", dataIndex: "forecast_demand", key: "forecast_demand", render: (v: number) => formatInt(v) },
    {
      title: "Margin",
      dataIndex: "margin_tier",
      key: "margin_tier",
      width: 110,
      render: (v: AllocationCityRow["margin_tier"]) => marginTag(v),
    },
    { title: "Allocated", dataIndex: "allocated", key: "allocated", render: (v: number) => formatInt(v) },
    {
      title: "Fill Rate",
      dataIndex: "fill_rate",
      key: "fill_rate",
      render: (v: number) => `${(v * 100).toFixed(0)}%`,
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (_: any, r) => statusTag(r.status),
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <div>
          <Title level={2} style={{ marginBottom: 0, color: "#0F3460" }}>
            Intelligent Supply Chain Control Tower
          </Title>
          <Text type="secondary">Stock Allocation Simulation powered by Optimization + Machine Learning</Text>
        </div>

        <Row gutter={[16, 16]}>
          <Col xs={24} lg={7}>
            <Card style={{ borderRadius: 8, borderTop: "3px solid #1D70B8" }} title="Control Panel">
              <Space direction="vertical" size={14} style={{ width: "100%" }}>
                <div>
                  <Text strong>Product</Text>
                  <Select
                    style={{ width: "100%", marginTop: 8 }}
                    loading={productsLoading}
                    value={productName}
                    placeholder="Select a product"
                    onChange={(v) => {
                      setProductName(v);
                      setResult(null);
                      setRunError(null);
                    }}
                    options={products.map((p) => ({
                      value: p.product_name,
                      label: `${p.product_name}`,
                    }))}
                  />
                  {productsError && <Alert style={{ marginTop: 12 }} type="error" showIcon message={productsError} />}
                </div>

                {result && (
                  <Card size="small" style={{ borderRadius: 8, background: "rgba(148, 163, 184, 0.08)" }}>
                    <Space direction="vertical" size={8} style={{ width: "100%" }}>
                      <div>
                        <Text type="secondary">Forecast Demand (7d)</Text>
                        <div style={{ fontWeight: 700, fontSize: 18, color: "#0B1220" }}>{formatInt(totalDemand)} units</div>
                      </div>
                      <div>
                        <Text type="secondary">Available Inventory</Text>
                        <div style={{ fontWeight: 700, fontSize: 18, color: "#0B1220" }}>
                          {formatInt(result.supply.available_inventory)} units
                          <Text type="secondary" style={{ marginLeft: 8 }}>
                            (gap {result.supply.gap_pct.toFixed(0)}%)
                          </Text>
                        </div>
                      </div>
                      <div>
                        <Text type="secondary">Inbound Replenishment</Text>
                        <div style={{ marginTop: 6 }}>
                          <Progress
                            percent={Math.min(100, Math.max(0, (result.supply.inbound.days / 7) * 100))}
                            showInfo={false}
                          />
                          <div style={{ marginTop: 6, fontSize: 12, color: "#64748B" }}>
                            {formatInt(result.supply.inbound.quantity)} units arriving in {result.supply.inbound.days} days
                          </div>
                        </div>
                      </div>
                    </Space>
                  </Card>
                )}

                <div>
                  <Text strong>Objective</Text>
                  <Radio.Group
                    style={{ width: "100%", marginTop: 8 }}
                    value={objective}
                    onChange={(e) => setObjective(e.target.value)}
                  >
                    <Space direction="vertical">
                      <Radio value="profit">
                        <Text strong style={{ color: "#FF7043" }}>
                          Profit Max
                        </Text>
                        <Text type="secondary"> — protect high-margin cities</Text>
                      </Radio>
                      <Radio value="fairness">
                        <Text strong style={{ color: "#1D70B8" }}>
                          Fairness
                        </Text>
                        <Text type="secondary"> — ensure a minimum fill for everyone</Text>
                      </Radio>
                      <Radio value="strategic">
                        <Text strong style={{ color: "#10B981" }}>
                          Strategic
                        </Text>
                        <Text type="secondary"> — prioritize core cities (e.g., San Francisco)</Text>
                      </Radio>
                    </Space>
                  </Radio.Group>
                </div>

                <div>
                  <Text strong>Constraints</Text>
                  <div style={{ marginTop: 8 }}>
                    <Space direction="vertical" size={10}>
                      <Checkbox checked={vipNY} onChange={(e) => setVipNY(e.target.checked)}>
                        New York City min fill 50% (VIP)
                      </Checkbox>
                      <Checkbox checked={minShipment} onChange={(e) => setMinShipment(e.target.checked)}>
                        Minimum shipment size ≥ 10 units
                      </Checkbox>
                    </Space>
                  </div>

                  <div style={{ marginTop: 16 }}>
                    <Text type="secondary">Minimum fill rate (Fairness objective)</Text>
                    <Slider
                      min={0}
                      max={100}
                      value={minFillRatePct}
                      onChange={(v) => setMinFillRatePct(v)}
                      disabled={objective !== "fairness"}
                    />
                    <div style={{ fontSize: 12, color: "#64748B" }}>Current: {minFillRatePct}%</div>
                  </div>
                </div>

                <div>
                  <Text strong>Simulation</Text>
                  <div style={{ marginTop: 10 }}>
                    <Text type="secondary">Demand shock</Text>
                    <Slider min={-50} max={50} value={demandShockPct} onChange={(v) => setDemandShockPct(v)} />
                    <div style={{ fontSize: 12, color: "#64748B" }}>{demandShockPct}%</div>
                  </div>
                  <div style={{ marginTop: 10 }}>
                    <Text type="secondary">Inventory shrink</Text>
                    <Slider min={-20} max={0} value={inventoryShrinkPct} onChange={(v) => setInventoryShrinkPct(v)} />
                    <div style={{ fontSize: 12, color: "#64748B" }}>{inventoryShrinkPct}%</div>
                  </div>
                </div>

                <Button
                  type="primary"
                  size="large"
                  icon={<PlayCircleOutlined />}
                  onClick={run}
                  loading={running}
                  disabled={!productName || productsLoading}
                  style={{ width: "100%" }}
                >
                  Run Optimization
                </Button>
              </Space>
            </Card>
          </Col>

          <Col xs={24} lg={17}>
            {runError && <Alert type="error" showIcon message="Run Failed" description={runError} />}

            {!result && running && (
              <Card style={{ borderRadius: 8 }}>
                <div style={{ textAlign: "center", padding: 24 }}>
                  <Spin />
                  <div style={{ marginTop: 12, color: "#64748B" }}>Running optimization...</div>
                </div>
              </Card>
            )}

            {result && (
              <Space direction="vertical" size="large" style={{ width: "100%" }}>
                <Row gutter={[16, 16]}>
                  <Col xs={24} sm={12} md={6}>
                    <Card style={{ borderRadius: 8 }}>
                      <Statistic
                        title="Profit ($)"
                        value={result.kpis.profit}
                        precision={0}
                        valueStyle={{ color: result.kpis.profit_delta_vs_profit_max >= 0 ? "#1D70B8" : "#FF7043" }}
                        suffix={
                          objective !== "profit" ? (
                            <span style={{ marginLeft: 8, fontSize: 12, color: "#64748B" }}>
                              ({result.kpis.profit_delta_vs_profit_max.toFixed(0)} vs Profit Max)
                            </span>
                          ) : undefined
                        }
                      />
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={6}>
                    <Card style={{ borderRadius: 8 }}>
                      <Statistic title="Fill Rate" value={result.kpis.fill_rate * 100} precision={0} suffix="%" />
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={6}>
                    <Card style={{ borderRadius: 8 }}>
                      <Statistic title="Lost Sales ($)" value={result.kpis.lost_sales} precision={0} />
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={6}>
                    <Card style={{ borderRadius: 8 }}>
                      <Statistic title="Risk Stores" value={result.kpis.risk_store_count} />
                    </Card>
                  </Col>
                </Row>

                <Card
                  title="AI Explanation (Markdown)"
                  style={{ borderRadius: 8 }}
                  extra={
                    result.ai_explanation_source ? (
                      <Tag color={result.ai_explanation_source === "ai" ? "blue" : "gold"}>
                        {result.ai_explanation_source === "ai" ? "Volcengine" : "Fallback"}
                      </Tag>
                    ) : undefined
                  }
                >
                  {result.ai_explanation_error && (
                    <Alert
                      type="warning"
                      showIcon
                      message="AI explanation failed; showing fallback."
                      description={result.ai_explanation_error}
                      style={{ marginBottom: 12 }}
                    />
                  )}
                  <MDEditor.Markdown source={result.ai_explanation_markdown || "No explanation available."} />
                </Card>

                <Row gutter={[16, 16]}>
                  <Col xs={24} xl={14}>
                    <Card
                      style={{ borderRadius: 8 }}
                      title="Visualization"
                      extra={
                        <Tag color="blue">
                          Forecast scope: {result.product.sub_category} (via Sub-Category model)
                        </Tag>
                      }
                    >
                      <Tabs
                        items={[
                          {
                            key: "map",
                            label: "Geo Map",
                            children: (
                              <div ref={mapRef} style={{ width: "100%", height: 340 }}>
                                <MapView cities={result.cities} width={mapSize.width} height={mapSize.height} />
                                <div style={{ marginTop: 10, fontSize: 12, color: "#64748B" }}>
                                  Green = high fill; yellow = medium; red = severe shortage.
                                </div>
                              </div>
                            ),
                          },
                          {
                            key: "bar",
                            label: "Demand vs Allocation",
                            children: (
                              <div ref={chartRef} style={{ width: "100%", height: 340 }}>
                                <BarChartView cities={result.cities} width={chartSize.width} height={chartSize.height} />
                              </div>
                            ),
                          },
                        ]}
                      />
                    </Card>
                  </Col>
                  <Col xs={24} xl={10}>
                    <Card style={{ borderRadius: 8 }} title="7-Day Trend">
                      <div ref={trendRef} style={{ width: "100%", height: 340 }}>
                        <TrendView
                          points={result.visuals.trend.points}
                          outOfStockDate={result.visuals.trend.out_of_stock_date}
                          width={trendSize.width}
                          height={trendSize.height}
                        />
                        {result.visuals.trend.out_of_stock_date && (
                          <div style={{ marginTop: 10, fontSize: 12, color: "#FF7043" }}>
                            Out of stock point: {result.visuals.trend.out_of_stock_date}
                          </div>
                        )}
                      </div>
                    </Card>
                  </Col>
                </Row>

                <Card style={{ borderRadius: 8 }} title="Allocation Details">
                  <Table<AllocationCityRow>
                    rowKey={(r) => r.city}
                    columns={cityColumns}
                    dataSource={result.cities}
                    pagination={{ pageSize: 10, showSizeChanger: true, pageSizeOptions: [10, 15, 30] }}
                    scroll={{ x: 820 }}
                  />
                </Card>
              </Space>
            )}
          </Col>
        </Row>
      </Space>
    </div>
  );
};
