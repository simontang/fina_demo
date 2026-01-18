import React, { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
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
  forecast?: SalesForecastRow[];
  trend_summary?: string;
  history?: SalesForecastHistoryPoint[];
};

type FormValues = {
  datasetId: string;
  modelId: string;
  targetEntityId: string;
  horizon: number;
  contextWindowDays: number;
  salesMetric: "quantity" | "revenue";
  promotionFactor: number;
  holidayCountry: string;
  rounding: "round" | "floor" | "none";
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

export const SalesForecast = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<AvailableModel[]>([]);

  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);

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
    if (!current.modelId && models.length > 0) next.modelId = models[0]!.id;
    if (Object.keys(next).length > 0) form.setFieldsValue(next as any);
  }, [datasets, models, form]);

  const historyColumns: ColumnsType<SalesForecastHistoryPoint> = useMemo(
    () => [
      { title: "Date", dataIndex: "date", key: "date", width: 120 },
      {
        title: "Sales",
        dataIndex: "sales",
        key: "sales",
        render: (v) => formatNumber(v),
      },
      {
        title: "Holiday",
        dataIndex: "is_holiday",
        key: "is_holiday",
        width: 90,
        render: (v) => (Number(v) ? "Yes" : "No"),
      },
    ],
    []
  );

  const forecastColumns: ColumnsType<SalesForecastRow> = useMemo(
    () => [
      { title: "Date", dataIndex: "date", key: "date", width: 120 },
      {
        title: "Predicted",
        dataIndex: "predicted_sales",
        key: "predicted_sales",
        render: (v) => formatNumber(v),
      },
      {
        title: "CI Lower",
        key: "ci_lower",
        render: (_, r) => formatNumber(r.confidence_interval?.lower),
      },
      {
        title: "CI Upper",
        key: "ci_upper",
        render: (_, r) => formatNumber(r.confidence_interval?.upper),
      },
    ],
    []
  );

  const onRunForecast = async (values: FormValues) => {
    setForecastLoading(true);
    setForecastError(null);
    setForecast(null);
    const headers = {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
    };

    try {
      const res = await fetch(`/api/v1/datasets/${encodeURIComponent(values.datasetId)}/sales-forecast`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model_id: values.modelId,
          target_entity_id: values.targetEntityId,
          forecast_horizon: values.horizon,
          context_window_days: values.contextWindowDays,
          sales_metric: values.salesMetric,
          promotion_factor: values.promotionFactor,
          holiday_country: values.holidayCountry,
          rounding: values.rounding,
        }),
      });

      const data = (await res.json()) as SalesForecastResponse;
      if (!res.ok) throw new Error((data as any)?.detail || `HTTP ${res.status}`);
      if (data.status !== "success") throw new Error((data as any)?.detail || "Forecast failed");

      setForecast(data);
      message.success("Forecast completed");
    } catch (e: any) {
      const msg = e?.message || "Forecast failed";
      setForecastError(msg);
      message.error(msg);
    } finally {
      setForecastLoading(false);
    }
  };

  const meta = forecast?.meta;
  const metaText = useMemo(() => {
    const mv = meta?.model_version ? `Model: ${meta.model_version}` : "";
    const gt = meta?.generated_at ? `Generated at: ${meta.generated_at}` : "";
    return [mv, gt].filter(Boolean).join(" · ");
  }, [meta]);

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Title level={3} style={{ margin: 0 }}>
            Sales Forecast
          </Title>
          <Text type="secondary">Select a model and dataset, then forecast future days.</Text>
        </Col>

        <Col span={24}>
          <Card title="Configuration">
            <Form<FormValues>
              form={form}
              layout="vertical"
              initialValues={{
                datasetId: "",
                modelId: "",
                targetEntityId: "",
                horizon: 7,
                contextWindowDays: 60,
                salesMetric: "revenue",
                promotionFactor: 1.0,
                holidayCountry: "GB",
                rounding: "none",
              }}
              onFinish={onRunForecast}
            >
              <Row gutter={16}>
                <Col xs={24} md={8}>
                  <Form.Item
                    label="Dataset"
                    name="datasetId"
                    rules={[{ required: true, message: "Please select a dataset" }]}
                  >
                    <Select
                      loading={loadingDatasets}
                      placeholder="Select a dataset"
                      options={datasets.map((d) => ({ label: `${d.name} (${d.id})`, value: d.id }))}
                      disabled={datasets.length === 0}
                    />
                  </Form.Item>
                  {datasetsError && <Alert type="error" showIcon message={datasetsError} />}
                </Col>

                <Col xs={24} md={8}>
                  <Form.Item
                    label="Model"
                    name="modelId"
                    rules={[{ required: true, message: "Please select a model" }]}
                  >
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
                  {modelsError && <Alert type="error" showIcon message={modelsError} />}
                </Col>

                <Col xs={24} md={8}>
                  <Form.Item
                    label="Target Entity ID (e.g. StockCode)"
                    name="targetEntityId"
                    rules={[{ required: true, message: "Please enter a target entity id" }]}
                  >
                    <Input placeholder="e.g. 85123A" />
                  </Form.Item>
                </Col>

                <Col xs={24} md={6}>
                  <Form.Item label="Horizon (days)" name="horizon" rules={[{ required: true }]}>
                    <InputNumber min={1} max={365} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="History Window (days)" name="contextWindowDays" rules={[{ required: true }]}>
                    <InputNumber min={7} max={3650} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="Metric" name="salesMetric" rules={[{ required: true }]}>
                    <Select
                      options={[
                        { label: "Quantity", value: "quantity" },
                        { label: "Revenue (Quantity x UnitPrice)", value: "revenue" },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="Promotion Factor" name="promotionFactor" rules={[{ required: true }]}>
                    <InputNumber min={0.01} step={0.05} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>

                <Col xs={24} md={6}>
                  <Form.Item label="Holiday Country Code" name="holidayCountry" rules={[{ required: true }]}>
                    <Input placeholder="e.g. GB / US" />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
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
                <Col xs={24} md={12} style={{ display: "flex", alignItems: "end" }}>
                  <Form.Item style={{ marginBottom: 0 }}>
                    <Space>
                      <Button type="primary" htmlType="submit" loading={forecastLoading}>
                        Run Forecast
                      </Button>
                      <Button
                        onClick={() => {
                          form.setFieldsValue({
                            datasetId: datasets[0]?.id || "",
                            modelId: models[0]?.id || "baseline_moving_average",
                            targetEntityId: "",
                            horizon: 7,
                            contextWindowDays: 60,
                            salesMetric: "revenue",
                            promotionFactor: 1.0,
                            holidayCountry: "GB",
                            rounding: "none",
                          });
                          setForecast(null);
                          setForecastError(null);
                        }}
                      >
                        Reset
                      </Button>
                    </Space>
                  </Form.Item>
                </Col>
              </Row>
            </Form>
          </Card>
        </Col>

        <Col span={24}>
          <Card title="Results" extra={metaText ? <Text type="secondary">{metaText}</Text> : null}>
            {forecastError && <Alert type="error" showIcon message={forecastError} style={{ marginBottom: 12 }} />}

            {forecastLoading ? (
              <Spin />
            ) : forecast?.status === "success" ? (
              <Row gutter={[16, 16]}>
                <Col xs={24} md={6}>
                  <Statistic title="Trend" value={forecast.trend_summary || "-"} />
                </Col>
                <Col xs={24} md={18}>
                  <Space direction="vertical" style={{ width: "100%" }} size={16}>
                    <Card size="small" title="History (Last 30 days)">
                      <Table
                        rowKey={(r) => r.date}
                        size="small"
                        pagination={false}
                        columns={historyColumns}
                        dataSource={forecast.history || []}
                        scroll={{ x: "max-content" }}
                      />
                    </Card>

                    <Card size="small" title="Forecast">
                      <Table
                        rowKey={(r) => r.date}
                        size="small"
                        pagination={false}
                        columns={forecastColumns}
                        dataSource={forecast.forecast || []}
                        scroll={{ x: "max-content" }}
                      />
                    </Card>
                  </Space>
                </Col>
              </Row>
            ) : (
              <Text type="secondary">Configure parameters and run a forecast.</Text>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};
