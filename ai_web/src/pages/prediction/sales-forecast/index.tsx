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
  return num.toLocaleString("zh-CN");
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
        if (!data?.success) throw new Error(data?.detail || data?.error || "获取数据集失败");
        setDatasets(data.data || []);
      } catch (e: any) {
        setDatasetsError(e?.message || "获取数据集失败");
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
        if (!data?.success) throw new Error(data?.detail || data?.error || "获取模型列表失败");
        setModels(data.models || []);
      } catch (e: any) {
        setModelsError(e?.message || "获取模型列表失败");
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
      { title: "日期", dataIndex: "date", key: "date", width: 120 },
      {
        title: "销量",
        dataIndex: "sales",
        key: "sales",
        render: (v) => formatNumber(v),
      },
      {
        title: "节假日",
        dataIndex: "is_holiday",
        key: "is_holiday",
        width: 90,
        render: (v) => (Number(v) ? "是" : "否"),
      },
    ],
    []
  );

  const forecastColumns: ColumnsType<SalesForecastRow> = useMemo(
    () => [
      { title: "日期", dataIndex: "date", key: "date", width: 120 },
      {
        title: "预测销量",
        dataIndex: "predicted_sales",
        key: "predicted_sales",
        render: (v) => formatNumber(v),
      },
      {
        title: "区间下限",
        key: "ci_lower",
        render: (_, r) => formatNumber(r.confidence_interval?.lower),
      },
      {
        title: "区间上限",
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
      if (data.status !== "success") throw new Error((data as any)?.detail || "预测失败");

      setForecast(data);
      message.success("预测完成");
    } catch (e: any) {
      const msg = e?.message || "预测失败";
      setForecastError(msg);
      message.error(msg);
    } finally {
      setForecastLoading(false);
    }
  };

  const meta = forecast?.meta;
  const metaText = useMemo(() => {
    const mv = meta?.model_version ? `模型: ${meta.model_version}` : "";
    const gt = meta?.generated_at ? `生成时间: ${meta.generated_at}` : "";
    return [mv, gt].filter(Boolean).join(" · ");
  }, [meta]);

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Title level={3} style={{ margin: 0 }}>
            销量预测
          </Title>
          <Text type="secondary">选择模型与数据集，按未来天数生成预测结果。</Text>
        </Col>

        <Col span={24}>
          <Card title="配置">
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
                    label="数据集"
                    name="datasetId"
                    rules={[{ required: true, message: "请选择数据集" }]}
                  >
                    <Select
                      loading={loadingDatasets}
                      placeholder="请选择数据集"
                      options={datasets.map((d) => ({ label: `${d.name} (${d.id})`, value: d.id }))}
                      disabled={datasets.length === 0}
                    />
                  </Form.Item>
                  {datasetsError && <Alert type="error" showIcon message={datasetsError} />}
                </Col>

                <Col xs={24} md={8}>
                  <Form.Item
                    label="模型"
                    name="modelId"
                    rules={[{ required: true, message: "请选择模型" }]}
                  >
                    <Select
                      loading={loadingModels}
                      placeholder="请选择模型"
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
                    label="预测对象 ID（例如 StockCode）"
                    name="targetEntityId"
                    rules={[{ required: true, message: "请输入目标 ID" }]}
                  >
                    <Input placeholder="例如: 85123A" />
                  </Form.Item>
                </Col>

                <Col xs={24} md={6}>
                  <Form.Item label="预测天数" name="horizon" rules={[{ required: true }]}>
                    <InputNumber min={1} max={365} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="历史窗口（天）" name="contextWindowDays" rules={[{ required: true }]}>
                    <InputNumber min={7} max={3650} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="销量口径" name="salesMetric" rules={[{ required: true }]}>
                    <Select
                      options={[
                        { label: "Quantity", value: "quantity" },
                        { label: "Revenue (Quantity x UnitPrice)", value: "revenue" },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="促销系数" name="promotionFactor" rules={[{ required: true }]}>
                    <InputNumber min={0.01} step={0.05} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>

                <Col xs={24} md={6}>
                  <Form.Item label="节假日国家码" name="holidayCountry" rules={[{ required: true }]}>
                    <Input placeholder="例如: CN / US" />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="取整方式" name="rounding" rules={[{ required: true }]}>
                    <Select
                      options={[
                        { label: "四舍五入", value: "round" },
                        { label: "向下取整", value: "floor" },
                        { label: "不取整", value: "none" },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12} style={{ display: "flex", alignItems: "end" }}>
                  <Form.Item style={{ marginBottom: 0 }}>
                    <Space>
                      <Button type="primary" htmlType="submit" loading={forecastLoading}>
                        运行预测
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
                        重置
                      </Button>
                    </Space>
                  </Form.Item>
                </Col>
              </Row>
            </Form>
          </Card>
        </Col>

        <Col span={24}>
          <Card title="结果" extra={metaText ? <Text type="secondary">{metaText}</Text> : null}>
            {forecastError && <Alert type="error" showIcon message={forecastError} style={{ marginBottom: 12 }} />}

            {forecastLoading ? (
              <Spin />
            ) : forecast?.status === "success" ? (
              <Row gutter={[16, 16]}>
                <Col xs={24} md={6}>
                  <Statistic title="趋势" value={forecast.trend_summary || "-"} />
                </Col>
                <Col xs={24} md={18}>
                  <Space direction="vertical" style={{ width: "100%" }} size={16}>
                    <Card size="small" title="历史（最近 30 天）">
                      <Table
                        rowKey={(r) => r.date}
                        size="small"
                        pagination={false}
                        columns={historyColumns}
                        dataSource={forecast.history || []}
                        scroll={{ x: "max-content" }}
                      />
                    </Card>

                    <Card size="small" title="预测结果">
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
              <Text type="secondary">请先配置参数并运行预测。</Text>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};
