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
  Table,
  Typography,
  message,
} from "antd";
import type { ColumnsType } from "antd/es/table";

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
  selectedFeatures: string[];
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

type SegmentationResponse = {
  status: "success" | "error";
  message?: string;
  dataset_id?: string;
  reference_date?: string;
  time_window_days?: number;
  model_info?: SegmentationModelInfo;
  clusters_summary?: ClusterSummary[];
  data_preview?: Record<string, any>[];
  elbow_curve_data?: ElbowPoint[];
  warnings?: string[];
  log_transform?: Record<string, { skew: number; shift: number }>;
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

function formatFloat(n: unknown, digits = 4): string {
  const num = typeof n === "number" ? n : Number(n);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

export const Segmentation: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  const [runLoading, setRunLoading] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<SegmentationResponse | null>(null);

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
        if (!json?.success) throw new Error(json?.error || json?.message || "获取数据集失败");
        if (!cancelled) setDatasets(json.data || []);
      } catch (e: any) {
        if (!cancelled) setDatasetsError(e?.message || "获取数据集失败");
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

  const onRun = async (values: SegmentationFormValues) => {
    setRunLoading(true);
    setRunError(null);
    setResult(null);
    try {
      const datasetId = values.datasetId;
      if (!datasetId) throw new Error("请选择数据集");

      const body = {
        time_window_days: Number(values.timeWindowDays),
        selected_features: values.selectedFeatures,
        k_range: [Number(values.kMin), Number(values.kMax)],
        random_seed: Number(values.randomSeed),
        outlier_threshold: Number(values.outlierThreshold),
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
      if (!res.ok) {
        throw new Error(json?.detail || `HTTP ${res.status}`);
      }
      if (json?.status !== "success") {
        throw new Error(json?.message || "聚类分析失败");
      }
      setResult(json as SegmentationResponse);
      message.success("聚类分析完成");
    } catch (e: any) {
      const msg = e?.message || "聚类分析失败";
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

  const clusterColumns: ColumnsType<ClusterSummary> = useMemo(() => {
    const features = result?.model_info?.features_used || [];
    const cols: ColumnsType<ClusterSummary> = [
      { title: "Cluster", dataIndex: "cluster_id", key: "cluster_id", width: 90 },
      { title: "用户数", dataIndex: "size", key: "size", width: 100, render: (v) => formatNumber(v) },
      { title: "占比", dataIndex: "percentage", key: "percentage", width: 90 },
      { title: "标签建议", dataIndex: "label_suggestion", key: "label_suggestion", width: 140 },
    ];

    for (const f of features) {
      cols.push({
        title: `${f} 均值`,
        key: `${f}_mean`,
        render: (_, r) => formatNumber(r.characteristics?.[f]?.mean),
      });
      cols.push({
        title: `${f} Std`,
        key: `${f}_std`,
        render: (_, r) => formatNumber(r.characteristics?.[f]?.std),
      });
    }
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
    if (s > 0.5) return <Text type="success">结构清晰</Text>;
    if (s >= 0.25) return <Text type="warning">结构尚可</Text>;
    return <Text type="danger">聚类效果偏弱</Text>;
  }, [result]);

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Title level={3} style={{ margin: 0 }}>
            Segmentation（客户聚类）
          </Title>
          <Text type="secondary">
            基于 RFM（recency_days / frequency / monetary）特征，对客户进行 K-Means 聚类，并自动在 K 范围内寻优。
          </Text>
        </Col>

        <Col span={24}>
          <Card title="配置">
            {datasetsError && (
              <Alert
                type="error"
                showIcon
                message="数据集加载失败"
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
                selectedFeatures: ["recency_days", "frequency", "monetary"],
              }}
              onFinish={onRun}
            >
              <Row gutter={16}>
                <Col xs={24} md={8}>
                  <Form.Item label="数据集" name="datasetId" rules={[{ required: true, message: "请选择数据集" }]}>
                    <Select
                      placeholder={datasetsLoading ? "加载中..." : "请选择数据集"}
                      loading={datasetsLoading}
                      options={datasets.map((d) => ({ label: `${d.name} (${d.id})`, value: d.id }))}
                      showSearch
                      optionFilterProp="label"
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={8}>
                  <Form.Item label="时间窗口（天）" name="timeWindowDays" rules={[{ required: true }]}>
                    <InputNumber min={1} max={3650} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={8}>
                  <Form.Item label="特征选择" name="selectedFeatures" rules={[{ required: true, message: "请选择至少一个特征" }]}>
                    <Select
                      mode="multiple"
                      options={[
                        { label: "recency_days（最近一次购买距今天数）", value: "recency_days" },
                        { label: "frequency（购买次数）", value: "frequency" },
                        { label: "monetary（消费金额）", value: "monetary" },
                      ]}
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col xs={24} md={6}>
                  <Form.Item label="K 最小值" name="kMin" rules={[{ required: true }]}>
                    <InputNumber min={2} max={50} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="K 最大值" name="kMax" rules={[{ required: true }]}>
                    <InputNumber min={2} max={50} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="随机种子" name="randomSeed" rules={[{ required: true }]}>
                    <InputNumber min={0} max={1_000_000} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={6}>
                  <Form.Item label="异常值阈值（Z-score）" name="outlierThreshold" rules={[{ required: true }]}>
                    <InputNumber min={0} max={10} step={0.1} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
              </Row>

              <Space style={{ width: "100%", justifyContent: "flex-end" }}>
                <Button type="primary" htmlType="submit" loading={runLoading} disabled={datasetsLoading}>
                  运行聚类
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
            <Alert type="error" showIcon message="聚类分析失败" description={runError} style={{ marginBottom: 16 }} />
          )}

          {result && (
            <Space direction="vertical" style={{ width: "100%" }} size={16}>
              <Card title="结果概览">
                <Row gutter={[16, 16]}>
                  <Col xs={12} md={6}>
                    <Statistic title="Best K" value={result.model_info?.best_k ?? "-"} />
                  </Col>
                  <Col xs={12} md={6}>
                    <Statistic
                      title="Best Silhouette"
                      value={result.model_info?.best_silhouette_score ?? "-"}
                      precision={4}
                    />
                    {silhouetteHint}
                  </Col>
                  <Col xs={24} md={12}>
                    <Text type="secondary">
                      数据集: {result.dataset_id || "-"} · reference_date: {result.reference_date || "-"} · window:{" "}
                      {result.time_window_days ?? "-"} 天
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

              <Row gutter={[16, 16]}>
                <Col xs={24} md={12}>
                  <Card title="Elbow Curve Data（SSE）" style={{ borderRadius: 8 }}>
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
                  <Card title="数据预览（带标签）" style={{ borderRadius: 8 }}>
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

              <Card title="簇画像（Centroids）" style={{ borderRadius: 8 }}>
                <Table<ClusterSummary>
                  rowKey={(r) => String(r.cluster_id)}
                  size="small"
                  columns={clusterColumns}
                  dataSource={result.clusters_summary || []}
                  pagination={false}
                  scroll={{ x: "max-content" }}
                />
              </Card>

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
