import React, { useState, useEffect } from "react";
import type { ReactNode } from "react";
import { useShow } from "@refinedev/core";
import { Show } from "@refinedev/antd";
import { useParams } from "react-router";
import {
  Card,
  Descriptions,
  Tag,
  Collapse,
  Table,
  Space,
  Typography,
  Statistic,
  Row,
  Col,
  Spin,
  Alert,
} from "antd";
import {
  DatabaseOutlined,
  CalendarOutlined,
  BarChartOutlined,
} from "@ant-design/icons";
import type { DatasetDetail as DatasetDetailType, DatasetPreview } from "../../../types/dataset";
import { TOKEN_KEY } from "../../../authProvider";

const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;

export const DatasetDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const showResult = useShow<DatasetDetailType>({
    resource: "datasets",
    id: id,
  });

  // refine v5: useShow 返回 { query, result, showId, setShowId }
  const queryResult = showResult.query;

  // 调试 useShow 返回值
  useEffect(() => {
    console.log("useShow result:", {
      showResult,
      queryResult,
      hasQueryResult: !!queryResult,
    });
  }, [showResult, queryResult]);

  const { data, isLoading, isError, error } = queryResult;
  const dataset = showResult.result;

  // 独立管理列统计信息状态
  const [columnStats, setColumnStats] = useState<any[]>([]);
  const [statsLoading, setStatsLoading] = useState(false);

  // 获取主要数据后，异步获取统计信息
  useEffect(() => {
    if (dataset?.id) {
      setStatsLoading(true);
      const token = localStorage.getItem(TOKEN_KEY);
      fetch(`/api/v1/datasets/${dataset.id}/stats`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      })
        .then((res) => res.json())
        .then((res) => {
          if (res.success && res.data) {
            setColumnStats(res.data);
          }
        })
        .catch((err) => console.error("Failed to load stats:", err))
        .finally(() => setStatsLoading(false));
    }
  }, [dataset?.id]);

  // 调试信息
  useEffect(() => {
    console.log("DatasetDetail Debug:", {
      id,
      dataset,
      isLoading,
      isError,
      hasData: !!data,
      dataStructure: data ? Object.keys(data) : null,
    });
  }, [id, dataset, isLoading, isError, data]);

  const [previewData, setPreviewData] = useState<DatasetPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewPage, setPreviewPage] = useState(1);
  const [previewPageSize, setPreviewPageSize] = useState(20);

  // 获取预览数据 - 使用 dataset.id 或 URL 中的 id
  useEffect(() => {
    const datasetId = dataset?.id || id;
    if (datasetId) {
      console.log("Fetching preview for dataset:", datasetId);
      fetchPreviewData(datasetId, previewPage, previewPageSize);
    }
  }, [dataset?.id, id, previewPage, previewPageSize]);

  const fetchPreviewData = async (datasetId: string, page: number, pageSize: number) => {
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const token = localStorage.getItem(TOKEN_KEY);
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (token) {
        headers.Authorization = `Bearer ${token}`;
      }
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      let response: Response;
      try {
        response = await fetch(
          `/api/v1/datasets/${datasetId}/preview?page=${page}&pageSize=${pageSize}`,
          { headers, signal: controller.signal }
        );
      } finally {
        clearTimeout(timeoutId);
      }

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const result = await response.json();
      console.log("Preview API response:", {
        success: result.success,
        hasData: !!result.data,
        recordsCount: result.data?.records?.length,
        total: result.data?.total,
        page: result.data?.page,
        pageSize: result.data?.pageSize,
      });

      if (result.success && result.data) {
        setPreviewData(result.data);
        console.log("Preview data set successfully:", {
          records: result.data.records?.length,
          total: result.data.total,
          firstRecordKeys: result.data.records?.[0] ? Object.keys(result.data.records[0]).length : 0,
        });
      } else {
        setPreviewError("预览数据格式异常");
      }
    } catch (e) {
      const isAbortError =
        typeof e === "object" && e !== null && "name" in e && (e as any).name === "AbortError";
      const msg = isAbortError
        ? "预览请求超时，请检查后端服务/数据库连接"
        : e instanceof Error
          ? e.message
          : "加载预览失败";
      setPreviewError(msg);
      setPreviewData(null);
      console.error("Error fetching preview data:", e);
    } finally {
      setPreviewLoading(false);
    }
  };

  // 即使 dataset 还没有完全加载，只要有 id 就可以先显示基本信息
  // 使用 id 作为后备，这样即使 dataset 还在加载，也能先显示预览数据
  const displayDataset: Partial<DatasetDetailType> = dataset || {
    id: id ?? "",
    name: id ?? "",
  };

  // 格式化值根据列类型
  const formatValue = (value: any, columnType: string): ReactNode => {
    if (value === null || value === undefined) {
      return <Text type="secondary">-</Text>;
    }

    // 日期时间类型
    if (
      columnType.includes("date") ||
      columnType.includes("time") ||
      columnType === "timestamp" ||
      columnType === "timestamp without time zone" ||
      columnType === "timestamp with time zone"
    ) {
      try {
        const date = new Date(value);
        if (isNaN(date.getTime())) {
          return String(value);
        }
        return (
          <Text>
            {date.toLocaleString("zh-CN", {
              year: "numeric",
              month: "2-digit",
              day: "2-digit",
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            })}
          </Text>
        );
      } catch {
        return String(value);
      }
    }

    // 数值类型
    if (
      columnType.includes("int") ||
      columnType.includes("numeric") ||
      columnType.includes("decimal") ||
      columnType.includes("real") ||
      columnType.includes("double") ||
      columnType === "float" ||
      columnType === "bigint"
    ) {
      if (typeof value === "number") {
        // 如果是整数，不显示小数
        if (Number.isInteger(value)) {
          return <Text>{value.toLocaleString()}</Text>;
        }
        // 浮点数，保留2位小数
        return <Text>{value.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</Text>;
      }
      // 字符串形式的数字
      const num = parseFloat(value);
      if (!isNaN(num)) {
        if (Number.isInteger(num)) {
          return <Text>{num.toLocaleString()}</Text>;
        }
        return <Text>{num.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</Text>;
      }
    }

    // 布尔类型
    if (columnType === "boolean" || columnType === "bool") {
      return <Tag color={value ? "green" : "default"}>{value ? "是" : "否"}</Tag>;
    }

    // 文本类型 - 如果太长则截断
    const str = String(value);
    if (str.length > 100) {
      return (
        <Text title={str} ellipsis={{ tooltip: str }}>
          {str.substring(0, 100)}...
        </Text>
      );
    }

    return <Text>{str}</Text>;
  };

  // 构建表格列，根据列类型优化显示
  // 如果 dataset.columns 为空，从预览数据的第一条记录推断列
  const tableColumns = React.useMemo(() => {
    console.log("Building table columns:", {
      hasDataset: !!dataset,
      hasColumns: !!dataset?.columns,
      columnsCount: dataset?.columns?.length || 0,
      hasPreviewData: !!previewData,
      recordsCount: previewData?.records?.length || 0,
    });

    if (!dataset?.columns || dataset.columns.length === 0) {
      // 如果还没有列信息，从预览数据推断
      if (previewData?.records && previewData.records.length > 0) {
        const firstRecord = previewData.records[0];
        return Object.keys(firstRecord).map((key) => {
          const value = firstRecord[key];
          const isNumeric = typeof value === "number";
          const isDate = value && typeof value === "string" && /^\d{4}-\d{2}-\d{2}/.test(value);

          return {
            title: key,
            dataIndex: key,
            key: key,
            width: isDate ? 180 : isNumeric ? 120 : 200,
            sorter: isNumeric
              ? (a: any, b: any) => {
                const aVal = parseFloat(a[key]) || 0;
                const bVal = parseFloat(b[key]) || 0;
                return aVal - bVal;
              }
              : undefined,
            render: (val: any) => {
              if (val === null || val === undefined) {
                return <Text type="secondary">-</Text>;
              }
              if (typeof val === "number") {
                return <Text>{val.toLocaleString()}</Text>;
              }
              if (isDate) {
                try {
                  const date = new Date(val);
                  if (!isNaN(date.getTime())) {
                    return (
                      <Text>
                        {date.toLocaleString("zh-CN", {
                          year: "numeric",
                          month: "2-digit",
                          day: "2-digit",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </Text>
                    );
                  }
                } catch { }
              }
              const str = String(val);
              if (str.length > 100) {
                return (
                  <Text title={str} ellipsis={{ tooltip: str }}>
                    {str.substring(0, 100)}...
                  </Text>
                );
              }
              return <Text>{str}</Text>;
            },
          };
        });
      }
      return [];
    }

    return dataset.columns.map((col) => {
      const isNumeric =
        col.type.includes("int") ||
        col.type.includes("numeric") ||
        col.type.includes("decimal") ||
        col.type.includes("real") ||
        col.type.includes("double") ||
        col.type === "float" ||
        col.type === "bigint";

      const isDate =
        col.type.includes("date") ||
        col.type.includes("time") ||
        col.type === "timestamp" ||
        col.type === "timestamp without time zone" ||
        col.type === "timestamp with time zone";

      return {
        title: (
          <Space direction="vertical" size={0}>
            <Text strong>{col.name}</Text>
            <Tag color="blue" style={{ fontSize: 11, margin: 0 }}>
              {col.type}
            </Tag>
          </Space>
        ),
        dataIndex: col.name,
        key: col.name,
        width: isDate ? 180 : isNumeric ? 120 : 200,
        sorter: isNumeric
          ? (a: any, b: any) => {
            const aVal = parseFloat(a[col.name]) || 0;
            const bVal = parseFloat(b[col.name]) || 0;
            return aVal - bVal;
          }
          : isDate
            ? (a: any, b: any) => {
              const aDate = new Date(a[col.name]).getTime() || 0;
              const bDate = new Date(b[col.name]).getTime() || 0;
              return aDate - bDate;
            }
            : undefined,
        render: (value: any) => formatValue(value, col.type),
        ellipsis: true,
      };
    });
  }, [dataset?.columns, previewData?.records]);

  // 不能在 hooks（如 useMemo）之前 return，否则会触发 Hooks 顺序变化错误
  // 首屏加载：等待 useShow 完成首次请求
  if (isLoading && !dataset) {
    return (
      <div style={{ padding: 24, textAlign: "center" }}>
        <Spin size="large" />
        <div style={{ marginTop: 16, color: "#999", fontSize: 14 }}>
          正在加载数据集详情...
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div style={{ padding: 24 }}>
        <Alert
          message="加载失败"
          description={`无法加载数据集详情: ${
            error instanceof Error ? error.message : String(error || "未知错误")
          }`}
          type="error"
          showIcon
        />
      </div>
    );
  }

  return (
    <Show
      title={false}
      headerButtons={false}
      contentProps={{
        style: {
          padding: 24,
        },
      }}
    >
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        {/* 数据集头部信息 */}
        <Card
          style={{
            borderTop: "3px solid #1D70B8",
            borderRadius: 6,
          }}
        >
          <Space direction="vertical" size="small" style={{ width: "100%" }}>
            <Title level={3} style={{ margin: 0, color: "#0F3460" }}>
              {displayDataset.name || id}
            </Title>
            {displayDataset.description && (
              <Paragraph style={{ margin: 0, color: "#64748B" }}>
                {displayDataset.description}
              </Paragraph>
            )}
            {displayDataset.tags && displayDataset.tags.length > 0 && (
              <Space size={[0, 8]} wrap>
                {displayDataset.tags.map((tag: string) => (
                  <Tag key={tag} style={{ borderRadius: 4 }}>
                    {tag}
                  </Tag>
                ))}
              </Space>
            )}
          </Space>
        </Card>

        {/* 基本信息 */}
        <Card
          title={
            <Space>
              <DatabaseOutlined style={{ color: "#1D70B8" }} />
              <span>基本信息</span>
            </Space>
          }
          style={{ borderRadius: 6, borderTop: "3px solid #1D70B8" }}
        >
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12} md={6}>
              <Statistic
                title="记录数"
                value={displayDataset.row_count || 0}
                valueStyle={{ color: "#1D70B8", fontSize: 24 }}
                suffix="条"
              />
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Statistic
                title="列数"
                value={displayDataset.column_count || 0}
                valueStyle={{ color: "#1D70B8", fontSize: 24 }}
                suffix="列"
              />
            </Col>
            {displayDataset.time_range && (
              <Col xs={24} sm={24} md={12}>
                <Space direction="vertical" size={0}>
                  <Text type="secondary" style={{ fontSize: 14 }}>
                    时间范围
                  </Text>
                  <Space>
                    <CalendarOutlined style={{ color: "#1D70B8" }} />
                    <Text strong>
                      {displayDataset.time_range.min
                        ? new Date(displayDataset.time_range.min).toLocaleDateString(
                          "zh-CN"
                        )
                        : "-"}
                    </Text>
                    <Text>至</Text>
                    <Text strong>
                      {displayDataset.time_range.max
                        ? new Date(displayDataset.time_range.max).toLocaleDateString(
                          "zh-CN"
                        )
                        : "-"}
                    </Text>
                  </Space>
                </Space>
              </Col>
            )}
          </Row>
        </Card>

        {/* 列统计信息 - 可选显示，如果 columns 不存在就显示加载状态 */}
        {dataset?.columns && dataset.columns.length > 0 ? (
          <Card
            title={
              <Space>
                <BarChartOutlined style={{ color: "#1D70B8" }} />
                <span>列定义与统计</span>
              </Space>
            }
            style={{ borderRadius: 6, borderTop: "3px solid #1D70B8" }}
          >
            <Collapse
              defaultActiveKey={dataset.columns.slice(0, 3).map((_, i) => i)}
              style={{ background: "transparent" }}
            >
              {dataset.columns.map((column, index) => {
                const statInfo = columnStats.find(c => c.name === column.name)?.stats || column.stats;

                return (
                  <Panel
                    header={
                      <Space>
                        <Text strong>{column.name}</Text>
                        <Tag color="blue" style={{ borderRadius: 4 }}>
                          {column.type}
                        </Tag>
                      </Space>
                    }
                    key={index}
                  >
                    {statsLoading && !statInfo ? (
                      <div style={{ padding: 12, textAlign: "center" }}>
                        <Spin size="small" /> <Text type="secondary" style={{ fontSize: 12, marginLeft: 8 }}>正在加载统计信息...</Text>
                      </div>
                    ) : statInfo ? (
                      <Descriptions column={1} size="small">
                        <Descriptions.Item label="空值数量">
                          {statInfo.nullCount.toLocaleString()}
                        </Descriptions.Item>
                        {statInfo.uniqueCount !== undefined && (
                          <Descriptions.Item label="唯一值数量">
                            <Text strong style={{ color: "#1D70B8", fontSize: 16 }}>
                              {statInfo.uniqueCount.toLocaleString()}
                            </Text>
                          </Descriptions.Item>
                        )}
                        {statInfo.distribution && (
                          <>
                            <Descriptions.Item label="最小值">
                              {statInfo.distribution.min.toLocaleString()}
                            </Descriptions.Item>
                            <Descriptions.Item label="最大值">
                              {statInfo.distribution.max.toLocaleString()}
                            </Descriptions.Item>
                            <Descriptions.Item label="平均值">
                              {statInfo.distribution.mean.toFixed(2)}
                            </Descriptions.Item>
                            <Descriptions.Item label="中位数">
                              {statInfo.distribution.median.toFixed(2)}
                            </Descriptions.Item>
                            <Descriptions.Item label="分位数 (25%, 50%, 75%)">
                              {statInfo.distribution.quartiles
                                .map((q: number) => q.toFixed(2))
                                .join(", ")}
                            </Descriptions.Item>
                          </>
                        )}
                      </Descriptions>
                    ) : (
                      <Text type="secondary">暂无统计信息</Text>
                    )}
                  </Panel>
                )
              })}
            </Collapse>
          </Card>
        ) : isLoading ? (
          <Card
            title={
              <Space>
                <BarChartOutlined style={{ color: "#1D70B8" }} />
                <span>列定义与统计</span>
              </Space>
            }
            style={{ borderRadius: 6, borderTop: "3px solid #1D70B8" }}
          >
            <div style={{ textAlign: "center", padding: "24px 0" }}>
              <Spin size="large" />
              <div style={{ marginTop: 16, color: "#999" }}>
                正在加载列定义与统计信息...
              </div>
            </div>
          </Card>
        ) : null}

        {/* 数据预览 */}
        <Card
          title={
            <Space>
              <span>数据预览</span>
            </Space>
          }
          style={{ borderRadius: 6, borderTop: "3px solid #1D70B8" }}
          extra={
            previewData && (
              <Text strong style={{ color: "#1D70B8" }}>
                共 {previewData.total.toLocaleString()} 条记录
              </Text>
            )
          }
        >
          <Spin spinning={previewLoading}>
            {previewError && (
              <Alert
                type="error"
                message="预览加载失败"
                description={previewError}
                showIcon
                style={{ marginBottom: 16 }}
              />
            )}
            {previewData && previewData.records && previewData.records.length > 0 ? (
              <>
                {tableColumns.length > 0 ? (
                  <Table
                    columns={tableColumns}
                    dataSource={previewData.records}
                    rowKey={(record, index) => {
                      // 优先使用 id 字段，否则使用索引
                      return record.id !== undefined ? `row-${record.id}` : `row-${index}`;
                    }}
                    pagination={{
                      current: previewPage,
                      pageSize: previewPageSize,
                      total: previewData.total,
                      showSizeChanger: true,
                      showQuickJumper: previewData.total > 1000,
                      pageSizeOptions: ["10", "20", "50", "100"],
                      showTotal: (total, range) =>
                        `第 ${range[0]}-${range[1]} 条，共 ${total.toLocaleString()} 条`,
                      onChange: (page, size) => {
                        setPreviewPage(page);
                        if (size !== previewPageSize) {
                          setPreviewPageSize(size);
                        }
                      },
                      onShowSizeChange: (current, size) => {
                        setPreviewPageSize(size);
                        setPreviewPage(1); // 改变每页大小时重置到第一页
                      },
                    }}
                    scroll={{ x: "max-content", y: "calc(100vh - 500px)" }}
                    size="small"
                    bordered
                    sticky={{ offsetHeader: 0 }}
                  />
                ) : (
                  <Alert
                    type="info"
                    message="正在加载列定义..."
                    description="请稍候，表格列信息正在加载中"
                    showIcon
                  />
                )}
              </>
            ) : previewData && (!previewData.records || previewData.records.length === 0) ? (
              <Alert
                type="info"
                message="暂无数据"
                description="当前页没有数据，请尝试其他页码"
                showIcon
              />
            ) : !previewData && !previewLoading && !previewError ? (
              <Text type="secondary">等待数据加载...</Text>
            ) : null}
          </Spin>
        </Card>
      </Space>
    </Show>
  );
};
