import { Alert, Button, Select, Spin, Tooltip, Typography } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import { useMemo } from "react";
import type { ReactNode, UIEvent } from "react";

const { Text } = Typography;
const LOAD_MORE_THRESHOLD = 24;
const RICH_OPTION_HEIGHT = 72;

interface ArtifactSelectorProps<Item> {
  ariaLabel: string;
  placeholder: string;
  items: Item[];
  selectedItem: Item | null;
  getId: (item: Item) => number;
  getName: (item: Item) => string;
  renderOption?: (item: Item) => ReactNode;
  renderSummary?: (item: Item) => ReactNode;
  searchValue: string;
  searchActive: boolean;
  overallTotal: number;
  matchedTotal: number;
  loading: boolean;
  loadingMore: boolean;
  error: string | null;
  onSearch: (value: string) => void;
  onSelect: (item: Item) => void;
  onLoadMore: () => void;
  onRetry: () => void;
}

export function ArtifactSelector<Item>({
  ariaLabel,
  placeholder,
  items,
  selectedItem,
  getId,
  getName,
  renderOption,
  renderSummary,
  searchValue,
  searchActive,
  overallTotal,
  matchedTotal,
  loading,
  loadingMore,
  error,
  onSearch,
  onSelect,
  onLoadMore,
  onRetry,
}: ArtifactSelectorProps<Item>) {
  const selectorItems = useMemo(() => {
    if (!selectedItem || items.some((item) => getId(item) === getId(selectedItem))) {
      return items;
    }
    return [selectedItem, ...items];
  }, [getId, items, selectedItem]);

  const options = useMemo(
    () => selectorItems.map((item) => ({
      value: getId(item),
      label: getName(item),
      title: getName(item),
    })),
    [getId, getName, selectorItems],
  );

  const itemById = useMemo(
    () => new Map(selectorItems.map((item) => [getId(item), item])),
    [getId, selectorItems],
  );

  const handlePopupScroll = (event: UIEvent<HTMLDivElement>) => {
    const target = event.currentTarget;
    const remaining = target.scrollHeight - target.scrollTop - target.clientHeight;
    if (remaining <= LOAD_MORE_THRESHOLD && !loading && !loadingMore && !error && items.length < matchedTotal) {
      onLoadMore();
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, flexShrink: 0 }}>
      <Select<number>
        aria-label={ariaLabel}
        value={selectedItem ? getId(selectedItem) : undefined}
        options={options}
        loading={loading}
        showSearch
        filterOption={false}
        searchValue={searchValue}
        placeholder={placeholder}
        notFoundContent={loading ? <Spin size="small" /> : searchActive ? "No matches" : "No artifacts"}
        onSearch={onSearch}
        onPopupScroll={handlePopupScroll}
        listItemHeight={renderOption ? RICH_OPTION_HEIGHT : undefined}
        onChange={(itemId) => {
          const item = selectorItems.find((candidate) => getId(candidate) === itemId);
          if (item) onSelect(item);
        }}
        optionRender={(option) => {
          const item = typeof option.value === "number" ? itemById.get(option.value) : undefined;
          if (item && renderOption) return renderOption(item);
          return (
            <Text ellipsis={{ tooltip: String(option.label) }} style={{ display: "block" }}>
              {option.label}
            </Text>
          );
        }}
        popupRender={(menu) => (
          <>
            {menu}
            {searchActive && matchedTotal === 0 ? (
              <div style={{ padding: "8px 12px", textAlign: "center" }}>
                <Text type="secondary" style={{ fontSize: 12 }}>No matches</Text>
              </div>
            ) : null}
            {loadingMore ? (
              <div style={{ padding: 8, textAlign: "center" }}><Spin size="small" /></div>
            ) : null}
          </>
        )}
        style={{ width: "100%", minWidth: 0 }}
      />
      {selectedItem && renderSummary ? (
        <div style={{ minWidth: 0 }}>
          {renderSummary(selectedItem)}
        </div>
      ) : null}
      <Text type="secondary" style={{ fontSize: 12 }}>
        {searchActive ? `${matchedTotal} matches · ${overallTotal} total` : `${overallTotal} total`}
      </Text>
      {error ? (
        <Alert
          type="error"
          showIcon
          message={error}
          action={(
            <Tooltip title="Retry">
              <Button
                type="text"
                size="small"
                icon={<ReloadOutlined />}
                aria-label="Retry artifact list"
                onClick={onRetry}
              />
            </Tooltip>
          )}
        />
      ) : null}
    </div>
  );
}
