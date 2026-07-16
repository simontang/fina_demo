import type { ReactNode } from "react";
import { Collapse, Descriptions, Empty, Space, Table, Tag, Typography } from "antd";
import type { CollapseProps, TableColumnsType } from "antd";
import type {
  CampaignAbTest,
  CampaignChannel,
  CampaignCondition,
  CampaignContentChannelStrategy,
  CampaignControlGroup,
  CampaignOffer,
  CampaignOfferStrategy,
  CampaignPresentation,
  CampaignSegmentation,
  CampaignSubSegment,
  CampaignVariant,
  CampaignWave,
  CampaignWaveStrategy,
  JsonValue,
  MarketingCampaignVO,
} from "./types";

const { Text } = Typography;

function hasJsonContent(value: JsonValue | undefined): value is JsonValue {
  if (value == null) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return true;
}

function humanizeKey(key: string): string {
  const words = key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .trim()
    .toLowerCase();
  return words ? `${words[0].toUpperCase()}${words.slice(1)}` : key;
}

function formatPercent(value: number): string {
  const percent = value * 100;
  return `${percent.toFixed(Number.isInteger(percent) ? 0 : 1)}%`;
}

function formatBoolean(value: boolean | undefined): ReactNode {
  if (value === undefined) return <Text type="secondary">Not specified</Text>;
  return <Tag color={value ? "success" : "default"}>{value ? "Yes" : "No"}</Tag>;
}

function renderTags(values: string[]): ReactNode {
  return values.length > 0
    ? <Space size={[0, 4]} wrap>{values.map((value) => <Tag key={value}>{value}</Tag>)}</Space>
    : <Text type="secondary">-</Text>;
}

function FriendlyValue({ value, depth = 0 }: { value: JsonValue; depth?: number }) {
  if (value === null) return <Text type="secondary">Not available</Text>;
  if (typeof value === "boolean") return <Tag color={value ? "success" : "default"}>{value ? "Yes" : "No"}</Tag>;
  if (typeof value === "number") return <Text>{value.toLocaleString()}</Text>;
  if (typeof value === "string") return <Text style={{ overflowWrap: "anywhere" }}>{value || "-"}</Text>;

  if (Array.isArray(value)) {
    if (value.length === 0) return <Text type="secondary">-</Text>;
    const primitiveOnly = value.every((item) => item === null || typeof item !== "object");
    if (primitiveOnly) {
      return (
        <Space size={[0, 4]} wrap>
          {value.map((item, index) => (
            <Tag key={`${String(item)}:${index}`}>
              {item === null ? "Not available" : typeof item === "boolean" ? (item ? "Yes" : "No") : String(item)}
            </Tag>
          ))}
        </Space>
      );
    }
    return (
      <div style={{ display: "grid", gap: 8 }}>
        {value.map((item, index) => (
          <div key={index} style={{ paddingLeft: 10, borderLeft: "2px solid #e8e8e8" }}>
            <Text type="secondary" style={{ display: "block", marginBottom: 4 }}>Item {index + 1}</Text>
            <FriendlyValue value={item} depth={depth + 1} />
          </div>
        ))}
      </div>
    );
  }

  const entries = Object.entries(value);
  if (entries.length === 0) return <Text type="secondary">-</Text>;
  return (
    <div style={{ display: "grid", gap: depth === 0 ? 8 : 6 }}>
      {entries.map(([key, item]) => (
        <div
          key={key}
          style={{
            display: "grid",
            gridTemplateColumns: depth === 0 ? "minmax(110px, 32%) minmax(0, 1fr)" : "minmax(96px, 30%) minmax(0, 1fr)",
            gap: 10,
            alignItems: "start",
          }}
        >
          <Text type="secondary" style={{ fontSize: 12 }}>{humanizeKey(key)}</Text>
          <FriendlyValue value={item} depth={depth + 1} />
        </div>
      ))}
    </div>
  );
}

function CompatibilityView({ value }: { value: JsonValue }) {
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <Text type="secondary">
        This campaign uses a legacy strategy shape. Its values are shown in a compatibility view.
      </Text>
      <FriendlyValue value={value} />
    </div>
  );
}

function ConditionList({ conditions }: { conditions: CampaignCondition[] }) {
  if (conditions.length === 0) return <Text type="secondary">-</Text>;
  return (
    <div style={{ display: "grid", gap: 4 }}>
      {conditions.map((condition, index) => (
        <div key={`${condition.subject}:${condition.operator}:${index}`} style={{ display: "flex", flexWrap: "wrap", gap: 4, alignItems: "center" }}>
          <Text code>{condition.subject}</Text>
          {condition.operator ? <Text>{condition.operator}</Text> : null}
          {condition.value !== undefined ? <FriendlyValue value={condition.value} /> : null}
        </div>
      ))}
    </div>
  );
}

function SectionLabel({ title, summary }: { title: string; summary: string }) {
  return (
    <div style={{ minWidth: 0, paddingRight: 8 }}>
      <Text strong style={{ display: "block" }}>{title}</Text>
      <Text type="secondary" ellipsis={{ tooltip: summary }} style={{ display: "block", fontSize: 12 }}>
        {summary}
      </Text>
    </div>
  );
}

const subSegmentColumns: TableColumnsType<CampaignSubSegment> = [
  {
    title: "Segment",
    key: "segment",
    width: 190,
    render: (_, item) => (
      <div style={{ minWidth: 0 }}>
        <Text strong style={{ display: "block" }}>{item.name}</Text>
        <Text code style={{ fontSize: 11 }}>{item.key}</Text>
      </div>
    ),
  },
  { title: "Priority", dataIndex: "priority", width: 80, render: (value?: number) => value ?? "-" },
  { title: "Criteria", dataIndex: "criteria", width: 300, render: (conditions: CampaignCondition[]) => <ConditionList conditions={conditions} /> },
  { title: "Tags", dataIndex: "tags", width: 180, render: (values: string[]) => renderTags(values) },
];

const channelColumns: TableColumnsType<CampaignChannel> = [
  {
    title: "Channel",
    key: "channel",
    width: 150,
    render: (_, item) => <div><Tag color="blue">{item.channel}</Tag><Text code style={{ fontSize: 11 }}>{item.key}</Text></div>,
  },
  { title: "Template", dataIndex: "templateKey", width: 190, render: (value?: string) => value ? <Text code>{value}</Text> : <Text type="secondary">-</Text> },
  { title: "Eligible segments", dataIndex: "eligibleSubSegmentKeys", width: 210, render: (values: string[]) => renderTags(values) },
  {
    title: "Delivery rules",
    key: "deliveryRules",
    width: 230,
    render: (_, item) => (
      <div style={{ display: "grid", gap: 3 }}>
        {item.sendWindow ? <Text>{[item.sendWindow.start && `${item.sendWindow.start}-${item.sendWindow.end || ""}`, item.sendWindow.timezone].filter(Boolean).join(" / ")}</Text> : null}
        {item.frequencyCap?.maxMessages !== undefined ? (
          <Text>{item.frequencyCap.maxMessages} messages / {item.frequencyCap.windowDays ?? "?"} days</Text>
        ) : null}
        {!item.sendWindow && item.frequencyCap?.maxMessages === undefined ? <Text type="secondary">-</Text> : null}
      </div>
    ),
  },
  {
    title: "Fallback and variables",
    key: "fallbackVariables",
    width: 220,
    render: (_, item) => (
      <div style={{ display: "grid", gap: 4 }}>
        {item.fallbackForChannelKeys.length > 0 ? <div><Text type="secondary">Fallback for: </Text>{renderTags(item.fallbackForChannelKeys)}</div> : null}
        {item.variables.length > 0 ? <div><Text type="secondary">Variables: </Text>{renderTags(item.variables)}</div> : null}
        {item.fallbackForChannelKeys.length === 0 && item.variables.length === 0 ? <Text type="secondary">-</Text> : null}
      </div>
    ),
  },
];

const offerColumns: TableColumnsType<CampaignOffer> = [
  {
    title: "Offer",
    key: "offer",
    width: 180,
    render: (_, item) => <div><Text strong style={{ display: "block" }}>{item.type || "Offer"}</Text><Text code>{item.code}</Text></div>,
  },
  {
    title: "Value",
    key: "value",
    width: 120,
    render: (_, item) => item.value === undefined ? <Text type="secondary">-</Text> : <Text>{[item.currency, item.value.toLocaleString()].filter(Boolean).join(" ")}</Text>,
  },
  { title: "Valid for", dataIndex: "validDays", width: 100, render: (value?: number) => value === undefined ? "-" : `${value} days` },
  { title: "Eligible segments", dataIndex: "eligibleSubSegmentKeys", width: 220, render: (values: string[]) => renderTags(values) },
  { title: "Customer limit", dataIndex: "perCustomerLimit", width: 120, render: (value?: number) => value ?? "-" },
];

const waveColumns: TableColumnsType<CampaignWave> = [
  {
    title: "Wave",
    key: "wave",
    width: 170,
    render: (_, item) => <div><Text strong style={{ display: "block" }}>{item.name}</Text><Text code>{item.id}</Text></div>,
  },
  { title: "Schedule", dataIndex: "scheduledAt", width: 175, render: (value?: string) => value || "-" },
  { title: "Eligible segments", dataIndex: "eligibleSubSegmentKeys", width: 220, render: (values: string[]) => renderTags(values) },
  { title: "Channels", dataIndex: "channelKeys", width: 170, render: (values: string[]) => renderTags(values) },
  { title: "Offers", dataIndex: "offerCodes", width: 190, render: (values: string[]) => renderTags(values) },
  {
    title: "Entry rule",
    key: "entryRule",
    width: 300,
    render: (_, item) => (
      <div style={{ display: "grid", gap: 4 }}>
        {item.fromWaveIds.length > 0 ? <div><Text type="secondary">From: </Text>{renderTags(item.fromWaveIds)}</div> : null}
        {item.excludeGroups.length > 0 ? <div><Text type="secondary">Exclude: </Text>{renderTags(item.excludeGroups)}</div> : null}
        <ConditionList conditions={item.conditions} />
      </div>
    ),
  },
];

const variantColumns: TableColumnsType<CampaignVariant> = [
  {
    title: "Variant",
    key: "variant",
    width: 160,
    render: (_, item) => <div><Text strong style={{ display: "block" }}>{item.name}</Text><Text code>{item.id}</Text></div>,
  },
  { title: "Traffic", dataIndex: "trafficRatio", width: 100, render: (value?: number) => value === undefined ? "-" : formatPercent(value) },
  { title: "Channel", dataIndex: "channelKey", width: 150, render: (value?: string) => value ? <Text code>{value}</Text> : "-" },
  { title: "Template", dataIndex: "templateKey", width: 190, render: (value?: string) => value ? <Text code>{value}</Text> : "-" },
  { title: "Offer", dataIndex: "offerCode", width: 170, render: (value?: string) => value ? <Text code>{value}</Text> : "-" },
];

function SegmentationDetails({ value }: { value: CampaignSegmentation }) {
  const hasMetadata = Boolean(value.version || value.audienceKey || value.source || value.assignment);
  return (
    <div style={{ display: "grid", gap: 16 }}>
      {hasMetadata ? (
        <Descriptions size="small" column={1} colon={false}>
          {value.version ? <Descriptions.Item label="Version">{value.version}</Descriptions.Item> : null}
          {value.audienceKey ? <Descriptions.Item label="Audience key"><Text code>{value.audienceKey}</Text></Descriptions.Item> : null}
          {value.source?.description ? <Descriptions.Item label="Source">{value.source.description}</Descriptions.Item> : null}
          {value.source?.segmentDataId !== undefined ? <Descriptions.Item label="Segment data"><Text code>#{value.source.segmentDataId}</Text></Descriptions.Item> : null}
          {value.source?.segmentDefinitionId !== undefined ? <Descriptions.Item label="Segment definition"><Text code>#{value.source.segmentDefinitionId}</Text></Descriptions.Item> : null}
          {value.source?.runId ? <Descriptions.Item label="Segment run"><Text code copyable>{value.source.runId}</Text></Descriptions.Item> : null}
          {value.assignment?.mode ? <Descriptions.Item label="Assignment mode">{humanizeKey(value.assignment.mode)}</Descriptions.Item> : null}
          {value.assignment?.fallbackSubSegmentKey ? <Descriptions.Item label="Fallback segment"><Text code>{value.assignment.fallbackSubSegmentKey}</Text></Descriptions.Item> : null}
        </Descriptions>
      ) : null}
      {value.subSegments.length > 0 ? (
        <section style={{ minWidth: 0, maxWidth: "100%" }}>
          <Text strong style={{ display: "block", marginBottom: 8 }}>Sub-segments</Text>
          <Table
            columns={subSegmentColumns}
            dataSource={value.subSegments}
            rowKey="key"
            pagination={false}
            size="small"
            scroll={{ x: 750 }}
            style={{ width: "100%", maxWidth: "100%" }}
          />
        </section>
      ) : null}
      {value.exclusions.length > 0 ? (
        <section>
          <Text strong style={{ display: "block", marginBottom: 8 }}>Exclusions</Text>
          <ConditionList conditions={value.exclusions} />
        </section>
      ) : null}
    </div>
  );
}

function ControlGroupDetails({ value }: { value: CampaignControlGroup }) {
  return (
    <Descriptions size="small" column={1} colon={false}>
      {value.enabled !== undefined ? <Descriptions.Item label="Enabled">{formatBoolean(value.enabled)}</Descriptions.Item> : null}
      {value.ratio !== undefined ? <Descriptions.Item label="Holdout ratio">{formatPercent(value.ratio)}</Descriptions.Item> : null}
      {value.method ? <Descriptions.Item label="Method">{humanizeKey(value.method)}</Descriptions.Item> : null}
      {value.unit ? <Descriptions.Item label="Unit">{humanizeKey(value.unit)}</Descriptions.Item> : null}
      {value.seed ? <Descriptions.Item label="Seed"><Text code>{value.seed}</Text></Descriptions.Item> : null}
      {value.stratifyBy.length > 0 ? <Descriptions.Item label="Stratified by">{renderTags(value.stratifyBy)}</Descriptions.Item> : null}
      {value.excludeFromWaves !== undefined ? <Descriptions.Item label="Excluded from waves">{formatBoolean(value.excludeFromWaves)}</Descriptions.Item> : null}
    </Descriptions>
  );
}

function ChannelDetails({ value }: { value: CampaignContentChannelStrategy }) {
  return (
    <div style={{ display: "grid", gap: 16 }}>
      {value.version || value.defaultLocale ? (
        <Descriptions size="small" column={1} colon={false}>
          {value.version ? <Descriptions.Item label="Version">{value.version}</Descriptions.Item> : null}
          {value.defaultLocale ? <Descriptions.Item label="Default locale">{value.defaultLocale}</Descriptions.Item> : null}
        </Descriptions>
      ) : null}
      {value.channels.length > 0 ? (
        <section style={{ minWidth: 0, maxWidth: "100%" }}>
          <Table
            columns={channelColumns}
            dataSource={value.channels}
            rowKey="key"
            pagination={false}
            size="small"
            scroll={{ x: 1000 }}
            style={{ width: "100%", maxWidth: "100%" }}
          />
        </section>
      ) : null}
    </div>
  );
}

function OfferDetails({ value }: { value: CampaignOfferStrategy }) {
  return (
    <div style={{ display: "grid", gap: 16 }}>
      {value.version || value.budget || value.allocation ? (
        <Descriptions size="small" column={1} colon={false}>
          {value.version ? <Descriptions.Item label="Version">{value.version}</Descriptions.Item> : null}
          {value.budget?.maxTotalCost !== undefined ? (
            <Descriptions.Item label="Budget cap">{[value.budget.currency, value.budget.maxTotalCost.toLocaleString()].filter(Boolean).join(" ")}</Descriptions.Item>
          ) : null}
          {value.allocation?.method ? <Descriptions.Item label="Allocation method">{humanizeKey(value.allocation.method)}</Descriptions.Item> : null}
          {value.allocation?.rules.length ? (
            <Descriptions.Item label="Allocation rules">
              <Space size={[0, 4]} wrap>
                {value.allocation.rules.map((rule, index) => (
                  <Tag key={`${rule.subSegmentKey}:${rule.offerCode}:${index}`}>
                    {rule.subSegmentKey || "Any segment"} to {rule.offerCode || "No offer"}
                  </Tag>
                ))}
              </Space>
            </Descriptions.Item>
          ) : null}
        </Descriptions>
      ) : null}
      {value.offers.length > 0 ? (
        <section style={{ minWidth: 0, maxWidth: "100%" }}>
          <Table
            columns={offerColumns}
            dataSource={value.offers}
            rowKey="code"
            pagination={false}
            size="small"
            scroll={{ x: 800 }}
            style={{ width: "100%", maxWidth: "100%" }}
          />
        </section>
      ) : null}
    </div>
  );
}

function WaveDetails({ value }: { value: CampaignWaveStrategy }) {
  return (
    <div style={{ display: "grid", gap: 16 }}>
      {value.enabled !== undefined || value.timezone ? (
        <Descriptions size="small" column={1} colon={false}>
          {value.enabled !== undefined ? <Descriptions.Item label="Enabled">{formatBoolean(value.enabled)}</Descriptions.Item> : null}
          {value.timezone ? <Descriptions.Item label="Timezone">{value.timezone}</Descriptions.Item> : null}
        </Descriptions>
      ) : null}
      {value.waves.length > 0 ? (
        <Table columns={waveColumns} dataSource={value.waves} rowKey="id" pagination={false} size="small" scroll={{ x: 1225 }} />
      ) : null}
    </div>
  );
}

function AbTestDetails({ value }: { value: CampaignAbTest }) {
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <Descriptions size="small" column={1} colon={false}>
        {value.enabled !== undefined ? <Descriptions.Item label="Enabled">{formatBoolean(value.enabled)}</Descriptions.Item> : null}
        {value.unit ? <Descriptions.Item label="Unit">{humanizeKey(value.unit)}</Descriptions.Item> : null}
        {value.primaryMetric ? <Descriptions.Item label="Primary metric"><Text code>{value.primaryMetric}</Text></Descriptions.Item> : null}
        {value.subSegmentKeys.length > 0 ? <Descriptions.Item label="Segments">{renderTags(value.subSegmentKeys)}</Descriptions.Item> : null}
        {value.waveIds.length > 0 ? <Descriptions.Item label="Waves">{renderTags(value.waveIds)}</Descriptions.Item> : null}
        {value.winnerPolicy?.method ? <Descriptions.Item label="Winner policy">{humanizeKey(value.winnerPolicy.method)}</Descriptions.Item> : null}
        {value.winnerPolicy?.minSampleSizePerVariant !== undefined ? <Descriptions.Item label="Minimum sample / variant">{value.winnerPolicy.minSampleSizePerVariant.toLocaleString()}</Descriptions.Item> : null}
        {value.winnerPolicy?.confidence !== undefined ? <Descriptions.Item label="Confidence">{formatPercent(value.winnerPolicy.confidence)}</Descriptions.Item> : null}
      </Descriptions>
      {value.variants.length > 0 ? (
        <Table columns={variantColumns} dataSource={value.variants} rowKey="id" pagination={false} size="small" scroll={{ x: 770 }} />
      ) : null}
    </div>
  );
}

function segmentationSummary(value?: CampaignSegmentation): string {
  if (!value) return "Legacy strategy format";
  const parts = [
    value.subSegments.length > 0 ? `${value.subSegments.length} sub-segments` : undefined,
    value.exclusions.length > 0 ? `${value.exclusions.length} exclusions` : undefined,
    value.source?.description,
  ].filter((item): item is string => Boolean(item));
  return parts.join(" / ") || "Configured";
}

function controlSummary(value?: CampaignControlGroup): string {
  if (!value) return "Legacy strategy format";
  const parts = [
    value.enabled === undefined ? undefined : value.enabled ? "Enabled" : "Disabled",
    value.ratio === undefined ? undefined : `${formatPercent(value.ratio)} holdout`,
    value.method ? humanizeKey(value.method) : undefined,
  ].filter((item): item is string => Boolean(item));
  return parts.join(" / ") || "Configured";
}

function channelSummary(value?: CampaignContentChannelStrategy): string {
  if (!value) return "Legacy strategy format";
  const channels = Array.from(new Set(value.channels.map((item) => item.channel)));
  return channels.length > 0 ? channels.join(", ") : "Configured";
}

function offerSummary(value?: CampaignOfferStrategy): string {
  if (!value) return "Legacy strategy format";
  const parts = [
    value.offers.length > 0 ? `${value.offers.length} offers` : undefined,
    value.budget?.maxTotalCost === undefined
      ? undefined
      : `${value.budget.currency || ""} ${value.budget.maxTotalCost.toLocaleString()} budget`.trim(),
  ].filter((item): item is string => Boolean(item));
  return parts.join(" / ") || "Configured";
}

function waveSummary(value?: CampaignWaveStrategy): string {
  if (!value) return "Legacy strategy format";
  const count = value.waves.length;
  return count > 0 ? `${count} wave${count === 1 ? "" : "s"}${value.timezone ? ` / ${value.timezone}` : ""}` : "Configured";
}

function abTestSummary(value?: CampaignAbTest): string {
  if (!value) return "Legacy strategy format";
  const parts = [
    value.enabled === undefined ? undefined : value.enabled ? "Enabled" : "Disabled",
    value.variants.length > 0 ? `${value.variants.length} variants` : undefined,
    value.primaryMetric,
  ].filter((item): item is string => Boolean(item));
  return parts.join(" / ") || "Configured";
}

export function CampaignStrategyTab({
  campaign,
  presentation,
}: {
  campaign: MarketingCampaignVO;
  presentation: CampaignPresentation;
}) {
  const items: CollapseProps["items"] = [];

  if (hasJsonContent(campaign.segmentationStrategy)) {
    items.push({
      key: "segmentation",
      label: <SectionLabel title="Segmentation" summary={segmentationSummary(presentation.segmentation)} />,
      children: presentation.segmentation
        ? <SegmentationDetails value={presentation.segmentation} />
        : <CompatibilityView value={campaign.segmentationStrategy} />,
    });
  }
  if (hasJsonContent(campaign.controlGroupStrategy)) {
    items.push({
      key: "control-group",
      label: <SectionLabel title="Control Group" summary={controlSummary(presentation.controlGroup)} />,
      children: presentation.controlGroup
        ? <ControlGroupDetails value={presentation.controlGroup} />
        : <CompatibilityView value={campaign.controlGroupStrategy} />,
    });
  }
  if (hasJsonContent(campaign.contentChannelStrategy)) {
    items.push({
      key: "channels",
      label: <SectionLabel title="Channels" summary={channelSummary(presentation.contentChannel)} />,
      children: presentation.contentChannel
        ? <ChannelDetails value={presentation.contentChannel} />
        : <CompatibilityView value={campaign.contentChannelStrategy} />,
    });
  }
  if (hasJsonContent(campaign.offerStrategy)) {
    items.push({
      key: "offers",
      label: <SectionLabel title="Offers" summary={offerSummary(presentation.offer)} />,
      children: presentation.offer
        ? <OfferDetails value={presentation.offer} />
        : <CompatibilityView value={campaign.offerStrategy} />,
    });
  }
  if (hasJsonContent(campaign.waveStrategy)) {
    items.push({
      key: "waves",
      label: <SectionLabel title="Waves" summary={waveSummary(presentation.wave)} />,
      children: presentation.wave
        ? <WaveDetails value={presentation.wave} />
        : <CompatibilityView value={campaign.waveStrategy} />,
    });
  }
  if (hasJsonContent(campaign.abTestStrategy)) {
    items.push({
      key: "ab-test",
      label: <SectionLabel title="A/B Test" summary={abTestSummary(presentation.abTest)} />,
      children: presentation.abTest
        ? <AbTestDetails value={presentation.abTest} />
        : <CompatibilityView value={campaign.abTestStrategy} />,
    });
  }

  if (items.length === 0) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No strategy configured yet." />;
  }

  return (
    <Collapse
      items={items}
      defaultActiveKey={items.some((item) => item?.key === "segmentation") ? ["segmentation"] : []}
      expandIconPosition="end"
      style={{ background: "transparent", minWidth: 0 }}
    />
  );
}
