import { Alert, Button, Empty, Spin, Table, Typography } from "antd";
import type { TableColumnsType } from "antd";
import { ArrowLeftOutlined, CloseOutlined } from "@ant-design/icons";
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useTableDetailWorkbenchContext } from "./tableDetailWorkbenchContext";
import { TABLE_DETAIL_WORKBENCH_LAYOUT } from "./tableDetailWorkbenchLayout";

const { Text } = Typography;

function useCompactWorkbench() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [compact, setCompact] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    const updateLayout = () => {
      setCompact(container.clientWidth < TABLE_DETAIL_WORKBENCH_LAYOUT.compactBreakpoint);
    };
    updateLayout();

    const observer = new ResizeObserver(updateLayout);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  return { containerRef, compact };
}

export interface TableDetailWorkbenchProps<RecordType extends object> {
  title: string;
  records: RecordType[];
  columns: TableColumnsType<RecordType>;
  getRecordId: (record: RecordType) => number;
  selectedId: number | null;
  loading: boolean;
  error: string | null;
  emptyDescription: string;
  detailLoading?: boolean;
  detailError?: string | null;
  onSelect: (record: RecordType) => void;
  onCloseDetail: () => void;
  renderDetail: () => ReactNode;
}

export function TableDetailWorkbench<RecordType extends object>({
  title,
  records,
  columns,
  getRecordId,
  selectedId,
  loading,
  error,
  emptyDescription,
  detailLoading = false,
  detailError = null,
  onSelect,
  onCloseDetail,
  renderDetail,
}: TableDetailWorkbenchProps<RecordType>) {
  const { detailExpanded, setDetailExpanded } = useTableDetailWorkbenchContext();
  const { containerRef, compact } = useCompactWorkbench();
  const hadSelectionRef = useRef(selectedId !== null);

  useEffect(() => {
    if (selectedId !== null) {
      hadSelectionRef.current = true;
      setDetailExpanded(true);
      return;
    }

    if (hadSelectionRef.current) {
      hadSelectionRef.current = false;
      setDetailExpanded(false);
    }
  }, [selectedId, setDetailExpanded]);

  const selectRecord = (record: RecordType) => {
    setDetailExpanded(true);
    onSelect(record);
  };

  const closeDetail = () => {
    onCloseDetail();
    setDetailExpanded(false);
  };

  const tablePanel = (split: boolean) => (
    <section
      aria-label={title}
      style={{
        display: "flex",
        flexDirection: "column",
        flex: split ? `0 0 ${TABLE_DETAIL_WORKBENCH_LAYOUT.tablePaneWidth}px` : 1,
        minWidth: 0,
        minHeight: 0,
        overflow: "hidden",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingBottom: 12 }}>
        <Text strong>{title}</Text>
        <Text type="secondary" style={{ fontSize: 12 }}>{records.length} total</Text>
      </div>
      {loading ? (
        <div style={{ flex: 1, display: "grid", placeItems: "center" }}><Spin /></div>
      ) : error ? (
        <Alert type="error" showIcon message={`${title} unavailable`} description={error} />
      ) : records.length === 0 ? (
        <div style={{ flex: 1, display: "grid", placeItems: "center" }}><Empty description={emptyDescription} /></div>
      ) : (
        <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
          <Table<RecordType>
            columns={columns}
            dataSource={records}
            size="small"
            pagination={false}
            rowKey={getRecordId}
            scroll={split ? { x: "max-content" } : undefined}
            onRow={(record) => {
              const recordId = getRecordId(record);
              const selected = selectedId === recordId;
              return {
                onClick: () => selectRecord(record),
                onKeyDown: (event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    selectRecord(record);
                  }
                },
                tabIndex: 0,
                "aria-selected": selected,
                style: {
                  cursor: "pointer",
                  background: selected ? TABLE_DETAIL_WORKBENCH_LAYOUT.selectedRowBackground : undefined,
                },
              };
            }}
          />
        </div>
      )}
    </section>
  );

  const detailPanel = (singleColumn: boolean) => (
    <section
      aria-label={`${title} detail`}
      style={{
        flex: 1,
        minWidth: 0,
        minHeight: 0,
        overflowY: "auto",
        paddingLeft: singleColumn ? 0 : TABLE_DETAIL_WORKBENCH_LAYOUT.contentGap,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: singleColumn ? "flex-start" : "flex-end",
          paddingBottom: 8,
          position: "sticky",
          top: 0,
          zIndex: 2,
          background: "#fff",
        }}
      >
        <Button
          type="text"
          size="small"
          icon={singleColumn ? <ArrowLeftOutlined /> : <CloseOutlined />}
          onClick={closeDetail}
          aria-label={singleColumn ? "Back to table" : "Close detail"}
        >
          {singleColumn ? "Back to table" : null}
        </Button>
      </div>
      {detailLoading ? (
        <div style={{ minHeight: 180, display: "grid", placeItems: "center" }}><Spin tip="Loading detail..." /></div>
      ) : detailError ? (
        <Alert type="error" showIcon message="Detail unavailable" description={detailError} />
      ) : (
        renderDetail()
      )}
    </section>
  );

  const showSplitLayout = detailExpanded && !compact;

  return (
    <div
      ref={containerRef}
      style={{
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        height: "100%",
        minHeight: 0,
        overflow: "hidden",
        padding: TABLE_DETAIL_WORKBENCH_LAYOUT.contentPadding,
      }}
    >
      {showSplitLayout ? (
        <div style={{ display: "flex", flex: 1, minHeight: 0, gap: TABLE_DETAIL_WORKBENCH_LAYOUT.contentGap }}>
          {tablePanel(true)}
          {detailPanel(false)}
        </div>
      ) : detailExpanded ? (
        detailPanel(true)
      ) : (
        tablePanel(false)
      )}
    </div>
  );
}
