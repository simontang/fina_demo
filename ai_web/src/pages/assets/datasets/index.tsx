import React from "react";
import { useList, useNavigation } from "@refinedev/core";
import { List, useTable } from "@refinedev/antd";
import { Table, Space, Tag, Button, Typography } from "antd";
import { EyeOutlined } from "@ant-design/icons";
import type { Dataset } from "../../../types/dataset";

const { Text } = Typography;

export const Datasets: React.FC = () => {
  const { show } = useNavigation();
  const { tableProps } = useTable<Dataset>({
    resource: "datasets",
    pagination: {
      pageSize: 10,
    },
  });

  return (
    <List>
      <Table
        {...tableProps}
        rowKey="id"
        style={{ marginTop: 16 }}
        onRow={(record) => ({
          onClick: () => show("datasets", record.id),
          style: { cursor: "pointer" },
        })}
        rowClassName={() => "dataset-table-row"}
      >
        <Table.Column
          dataIndex="name"
          title="数据集名称"
          render={(value, record: Dataset) => (
            <Space direction="vertical" size={0}>
              <Text strong style={{ color: "#1D70B8" }}>
                {value}
              </Text>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {record.description}
              </Text>
            </Space>
          )}
        />
        <Table.Column
          dataIndex="type"
          title="类型"
          width={120}
          render={(value) => (
            <Tag color="blue" style={{ borderRadius: 4 }}>
              {value}
            </Tag>
          )}
        />
        <Table.Column
          dataIndex="row_count"
          title="记录数"
          width={120}
          render={(value) => (
            <Text strong style={{ color: "#0F3460" }}>
              {value?.toLocaleString() || 0}
            </Text>
          )}
        />
        <Table.Column
          dataIndex="tags"
          title="标签"
          width={200}
          render={(tags: string[]) => (
            <Space size={[0, 8]} wrap>
              {tags?.map((tag) => (
                <Tag key={tag} style={{ borderRadius: 4 }}>
                  {tag}
                </Tag>
              ))}
            </Space>
          )}
        />
        <Table.Column
          dataIndex="updated_at"
          title="最后更新"
          width={180}
          render={(value) => {
            if (!value) return "-";
            const date = new Date(value);
            return date.toLocaleString("zh-CN", {
              year: "numeric",
              month: "2-digit",
              day: "2-digit",
              hour: "2-digit",
              minute: "2-digit",
            });
          }}
        />
        <Table.Column
          title="操作"
          width={100}
          fixed="right"
          render={(_, record: Dataset) => (
            <Button
              type="link"
              icon={<EyeOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                show("datasets", record.id);
              }}
              style={{ color: "#1D70B8" }}
            >
              查看
            </Button>
          )}
        />
      </Table>
    </List>
  );
};
