import React, { useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, Space, Table, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { ReloadOutlined } from "@ant-design/icons";

import { TOKEN_KEY } from "../../../authProvider";

const { Title, Text } = Typography;

type ModelAsset = {
  id: string;
  name: string;
  version: string;
  framework?: string | null;
  task?: string | null;
  target_metric?: string | null;
  trained_at?: string | null;
  metrics?: Record<string, any> | null;
};

type ListResponse = {
  success: boolean;
  data?: ModelAsset[];
  total?: number;
  page?: number;
  pageSize?: number;
  totalPages?: number;
  detail?: string;
  error?: string;
};

function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function formatMetric(v: unknown): string {
  const n = typeof v === "number" ? v : Number(v);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("zh-CN", { maximumFractionDigits: 4 });
}

export const Models = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [items, setItems] = useState<ModelAsset[]>([]);
  const [total, setTotal] = useState(0);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/v1/model-assets?page=1&pageSize=200`, {
        headers: getAuthHeaders(),
      });
      const data = (await res.json()) as ListResponse;
      if (!res.ok) throw new Error((data as any)?.detail || `HTTP ${res.status}`);
      if (!data.success) throw new Error(data.detail || data.error || "获取模型列表失败");
      setItems(data.data || []);
      setTotal(data.total || 0);
    } catch (e: any) {
      setError(e?.message || "获取模型列表失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const columns: ColumnsType<ModelAsset> = useMemo(
    () => [
      { title: "模型名", dataIndex: "name", key: "name", width: 220 },
      { title: "版本", dataIndex: "version", key: "version", width: 140 },
      { title: "框架", dataIndex: "framework", key: "framework", width: 140 },
      { title: "任务", dataIndex: "task", key: "task", width: 160 },
      { title: "目标", dataIndex: "target_metric", key: "target_metric", width: 140 },
      { title: "训练时间", dataIndex: "trained_at", key: "trained_at", width: 200 },
      {
        title: "MAE",
        key: "mae",
        width: 120,
        render: (_, r) => formatMetric(r.metrics?.mae),
      },
      {
        title: "MAPE",
        key: "mape",
        width: 120,
        render: (_, r) => formatMetric(r.metrics?.mape),
      },
    ],
    []
  );

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: "100%" }} size={16}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            Models
          </Title>
          <Text type="secondary">从仓库根目录的 models/ 扫描模型与版本（metadata.json）。</Text>
        </div>

        <Card
          title={`模型列表（${total.toLocaleString("zh-CN")}）`}
          extra={
            <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
              刷新
            </Button>
          }
        >
          {error && <Alert type="error" showIcon message={error} style={{ marginBottom: 12 }} />}
          <Table
            rowKey={(r) => r.id}
            loading={loading}
            size="small"
            columns={columns}
            dataSource={items}
            pagination={false}
            scroll={{ x: "max-content" }}
          />
        </Card>
      </Space>
    </div>
  );
};

