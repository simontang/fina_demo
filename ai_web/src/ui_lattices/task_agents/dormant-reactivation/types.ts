export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

interface CampaignAudienceStatistics {
  targetCount?: number;
  controlCount?: number;
  treatmentCount?: number;
}

interface CampaignDeliveryStatistics {
  sent?: number;
  delivered?: number;
  opened?: number;
  clicked?: number;
}

interface CampaignConversionStatistics {
  converted?: number;
  conversionRate?: number | null;
}

export interface CampaignStatistics {
  audience?: CampaignAudienceStatistics;
  delivery?: CampaignDeliveryStatistics;
  conversion?: CampaignConversionStatistics;
}

export interface MarketingCampaignVO {
  id: number;
  tenantId: string;
  name: string;
  description: string;
  type: string;
  status: string;
  goal: string;
  startTime: string;
  endTime: string;
  mainSegmentDataId: number | null;
  segmentationStrategy?: JsonValue;
  controlGroupStrategy?: JsonValue;
  contentChannelStrategy?: JsonValue;
  offerStrategy?: JsonValue;
  waveStrategy?: JsonValue;
  abTestStrategy?: JsonValue;
  statistics?: CampaignStatistics;
  actualStartedAt?: string;
  actualStoppedAt?: string;
  createdAt: string;
  updatedAt: string;
}

export interface MarketingCampaignPage {
  items: MarketingCampaignVO[];
  total: number;
  page: number;
  pageSize: number;
}

export const campaignStatusConfig: Record<string, { color: string; label: string }> = {
  draft: { color: "default", label: "Draft" },
  scheduled: { color: "blue", label: "Scheduled" },
  running: { color: "processing", label: "Running" },
  stopped: { color: "warning", label: "Stopped" },
  completed: { color: "success", label: "Completed" },
};
