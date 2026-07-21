export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export interface MarketingCampaignVO {
  id: number;
  tenantId: string;
  threadId?: string;
  name: string;
  description?: string | null;
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
  statistics?: JsonValue;
  actualStartedAt?: string | null;
  actualStoppedAt?: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface MarketingCampaignPage {
  items: MarketingCampaignVO[];
  total: number;
  page: number;
  pageSize: number;
}

export interface CampaignCondition {
  subject: string;
  operator?: string;
  value?: JsonValue;
}

export interface CampaignSegmentSource {
  segmentDataId?: number;
  segmentDefinitionId?: number;
  runId?: string;
  description?: string;
}

export interface CampaignSubSegment {
  key: string;
  name: string;
  priority?: number;
  criteria: CampaignCondition[];
  tags: string[];
}

export interface CampaignSegmentation {
  version?: string;
  audienceKey?: string;
  source?: CampaignSegmentSource;
  subSegments: CampaignSubSegment[];
  assignment?: {
    mode?: string;
    fallbackSubSegmentKey?: string;
  };
  exclusions: CampaignCondition[];
}

export interface CampaignControlGroup {
  enabled?: boolean;
  method?: string;
  unit?: string;
  ratio?: number;
  seed?: string;
  stratifyBy: string[];
  excludeFromWaves?: boolean;
}

export interface CampaignChannel {
  key: string;
  channel: string;
  templateKey?: string;
  eligibleSubSegmentKeys: string[];
  fallbackForChannelKeys: string[];
  sendWindow?: { timezone?: string; start?: string; end?: string };
  frequencyCap?: { maxMessages?: number; windowDays?: number };
  variables: string[];
}

export interface CampaignContentChannelStrategy {
  version?: string;
  defaultLocale?: string;
  channels: CampaignChannel[];
}

export interface CampaignOffer {
  code: string;
  type?: string;
  value?: number;
  currency?: string;
  validDays?: number;
  eligibleSubSegmentKeys: string[];
  perCustomerLimit?: number;
}

export interface CampaignOfferStrategy {
  version?: string;
  budget?: { currency?: string; maxTotalCost?: number };
  offers: CampaignOffer[];
  allocation?: {
    method?: string;
    rules: Array<{ subSegmentKey?: string; offerCode?: string }>;
  };
}

export interface CampaignWave {
  id: string;
  name: string;
  scheduledAt?: string;
  eligibleSubSegmentKeys: string[];
  channelKeys: string[];
  offerCodes: string[];
  fromWaveIds: string[];
  excludeGroups: string[];
  conditions: CampaignCondition[];
}

export interface CampaignWaveStrategy {
  enabled?: boolean;
  timezone?: string;
  waves: CampaignWave[];
}

export interface CampaignVariant {
  id: string;
  name: string;
  trafficRatio?: number;
  channelKey?: string;
  templateKey?: string;
  offerCode?: string;
}

export interface CampaignAbTest {
  enabled?: boolean;
  unit?: string;
  primaryMetric?: string;
  waveIds: string[];
  subSegmentKeys: string[];
  variants: CampaignVariant[];
  winnerPolicy?: {
    method?: string;
    minSampleSizePerVariant?: number;
    confidence?: number;
  };
}

export interface CampaignStatistics {
  audience?: { targetCount?: number; controlCount?: number; treatmentCount?: number };
  delivery?: { sent?: number; delivered?: number; opened?: number; clicked?: number; failed?: number };
  conversion?: { converted?: number; conversionRate?: number | null; incrementalLift?: number | null };
  revenue?: {
    currency?: string;
    grossRevenue?: number;
    incrementalRevenue?: number | null;
    offerCost?: number;
    grossMargin?: number | null;
  };
}

export interface CampaignPresentation {
  segmentation?: CampaignSegmentation;
  controlGroup?: CampaignControlGroup;
  contentChannel?: CampaignContentChannelStrategy;
  offer?: CampaignOfferStrategy;
  wave?: CampaignWaveStrategy;
  abTest?: CampaignAbTest;
  statistics?: CampaignStatistics;
}

export const campaignStatusConfig: Record<string, { color: string; label: string }> = {
  draft: { color: "default", label: "Draft" },
  scheduled: { color: "blue", label: "Scheduled" },
  running: { color: "processing", label: "Running" },
  stopped: { color: "warning", label: "Stopped" },
  completed: { color: "success", label: "Completed" },
};
