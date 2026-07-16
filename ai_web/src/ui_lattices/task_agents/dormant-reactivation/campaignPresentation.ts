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
  CampaignStatistics,
  CampaignSubSegment,
  CampaignVariant,
  CampaignWave,
  CampaignWaveStrategy,
  JsonValue,
  MarketingCampaignVO,
} from "./types";

type JsonObject = { [key: string]: JsonValue };

function asObject(value: unknown): JsonObject | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject
    : undefined;
}

function asObjects(value: unknown): JsonObject[] {
  return Array.isArray(value)
    ? value.map(asObject).filter((item): item is JsonObject => Boolean(item))
    : [];
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function asStrings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map(asString).filter((item): item is string => Boolean(item))
    : [];
}

function parseCondition(value: JsonObject): CampaignCondition | undefined {
  const subject = asString(value.field) || asString(value.metric);
  return subject ? { subject, operator: asString(value.operator), value: value.value } : undefined;
}

function parseConditions(value: unknown): CampaignCondition[] {
  return asObjects(value)
    .map(parseCondition)
    .filter((item): item is CampaignCondition => Boolean(item));
}

function parseSubSegments(value: unknown): CampaignSubSegment[] {
  return asObjects(value).flatMap((item) => {
    const key = asString(item.subSegmentKey);
    if (!key) return [];
    return [{
      key,
      name: asString(item.name) || key,
      priority: asNumber(item.priority),
      criteria: parseConditions(item.criteria),
      tags: asStrings(item.tags),
    }];
  });
}

function parseSegmentation(value: unknown): CampaignSegmentation | undefined {
  const strategy = asObject(value);
  if (!strategy) return undefined;

  const sourceObject = asObject(strategy.source);
  const source = sourceObject ? {
    segmentDataId: asNumber(sourceObject.segmentDataId),
    segmentDefinitionId: asNumber(sourceObject.segmentDefinitionId),
    runId: asString(sourceObject.runId),
    description: asString(sourceObject.description),
  } : undefined;
  const sourceHasValue = source && Object.values(source).some((item) => item !== undefined);

  const assignmentObject = asObject(strategy.assignment);
  const assignment = assignmentObject ? {
    mode: asString(assignmentObject.mode),
    fallbackSubSegmentKey: asString(assignmentObject.fallbackSubSegmentKey),
  } : undefined;
  const assignmentHasValue = assignment && Object.values(assignment).some((item) => item !== undefined);

  const version = asString(strategy.version);
  const audienceKey = asString(strategy.audienceKey);
  const subSegments = parseSubSegments(strategy.subSegments);
  const exclusions = parseConditions(strategy.exclusions);
  if (!version && !audienceKey && !sourceHasValue && !assignmentHasValue && subSegments.length === 0 && exclusions.length === 0) {
    return undefined;
  }

  return {
    version,
    audienceKey,
    source: sourceHasValue ? source : undefined,
    subSegments,
    assignment: assignmentHasValue ? assignment : undefined,
    exclusions,
  };
}

function parseControlGroup(value: unknown): CampaignControlGroup | undefined {
  const strategy = asObject(value);
  if (!strategy) return undefined;

  const controlGroup: CampaignControlGroup = {
    enabled: asBoolean(strategy.enabled),
    method: asString(strategy.method),
    unit: asString(strategy.unit),
    ratio: asNumber(strategy.ratio),
    seed: asString(strategy.seed),
    stratifyBy: asStrings(strategy.stratifyBy),
    excludeFromWaves: asBoolean(strategy.excludeFromWaves),
  };
  const hasValue = Object.entries(controlGroup)
    .some(([, item]) => Array.isArray(item) ? item.length > 0 : item !== undefined);
  return hasValue ? controlGroup : undefined;
}

function parseChannels(value: unknown): CampaignChannel[] {
  return asObjects(value).flatMap((item) => {
    const key = asString(item.channelKey);
    const channel = asString(item.channel);
    if (!key || !channel) return [];
    const sendWindow = asObject(item.sendWindow);
    const frequencyCap = asObject(item.frequencyCap);
    return [{
      key,
      channel,
      templateKey: asString(item.templateKey),
      eligibleSubSegmentKeys: asStrings(item.eligibleSubSegmentKeys),
      fallbackForChannelKeys: asStrings(item.fallbackForChannelKeys),
      sendWindow: sendWindow ? {
        timezone: asString(sendWindow.timezone),
        start: asString(sendWindow.start),
        end: asString(sendWindow.end),
      } : undefined,
      frequencyCap: frequencyCap ? {
        maxMessages: asNumber(frequencyCap.maxMessages),
        windowDays: asNumber(frequencyCap.windowDays),
      } : undefined,
      variables: asStrings(item.variables),
    }];
  });
}

function parseContentChannel(value: unknown): CampaignContentChannelStrategy | undefined {
  const strategy = asObject(value);
  if (!strategy) return undefined;
  const version = asString(strategy.version);
  const defaultLocale = asString(strategy.defaultLocale);
  const channels = parseChannels(strategy.channels);
  return version || defaultLocale || channels.length > 0
    ? { version, defaultLocale, channels }
    : undefined;
}

function parseOffers(value: unknown): CampaignOffer[] {
  return asObjects(value).flatMap((item) => {
    const code = asString(item.offerCode);
    if (!code) return [];
    return [{
      code,
      type: asString(item.type),
      value: asNumber(item.value),
      currency: asString(item.currency),
      validDays: asNumber(item.validDays),
      eligibleSubSegmentKeys: asStrings(item.eligibleSubSegmentKeys),
      perCustomerLimit: asNumber(item.perCustomerLimit),
    }];
  });
}

function parseOffer(value: unknown): CampaignOfferStrategy | undefined {
  const strategy = asObject(value);
  if (!strategy) return undefined;
  const version = asString(strategy.version);
  const budgetObject = asObject(strategy.budget);
  const budget = budgetObject ? {
    currency: asString(budgetObject.currency),
    maxTotalCost: asNumber(budgetObject.maxTotalCost),
  } : undefined;
  const budgetHasValue = budget && Object.values(budget).some((item) => item !== undefined);
  const offers = parseOffers(strategy.offers);
  const allocationObject = asObject(strategy.allocation);
  const allocationRules = asObjects(allocationObject?.rules).flatMap((item) => {
    const rule = {
      subSegmentKey: asString(item.subSegmentKey),
      offerCode: asString(item.offerCode),
    };
    return rule.subSegmentKey || rule.offerCode ? [rule] : [];
  });
  const allocationMethod = asString(allocationObject?.method);
  const allocation = allocationMethod || allocationRules.length > 0
    ? { method: allocationMethod, rules: allocationRules }
    : undefined;

  return version || budgetHasValue || offers.length > 0 || allocation
    ? { version, budget: budgetHasValue ? budget : undefined, offers, allocation }
    : undefined;
}

function parseWaves(value: unknown): CampaignWave[] {
  return asObjects(value).flatMap((item) => {
    const id = asString(item.waveId);
    if (!id) return [];
    const entryRule = asObject(item.entryRule);
    return [{
      id,
      name: asString(item.name) || id,
      scheduledAt: asString(item.scheduledAt),
      eligibleSubSegmentKeys: asStrings(item.eligibleSubSegmentKeys),
      channelKeys: asStrings(item.channelKeys),
      offerCodes: asStrings(item.offerCodes),
      fromWaveIds: asStrings(entryRule?.fromWaveIds),
      excludeGroups: asStrings(entryRule?.excludeGroups),
      conditions: parseConditions(entryRule?.includeIf),
    }];
  });
}

function parseWave(value: unknown): CampaignWaveStrategy | undefined {
  const strategy = asObject(value);
  if (!strategy) return undefined;
  const enabled = asBoolean(strategy.enabled);
  const timezone = asString(strategy.timezone);
  const waves = parseWaves(strategy.waves);
  return enabled !== undefined || timezone || waves.length > 0
    ? { enabled, timezone, waves }
    : undefined;
}

function parseAbTest(value: unknown): CampaignAbTest | undefined {
  const strategy = asObject(value);
  if (!strategy) return undefined;
  const scope = asObject(strategy.scope);
  const winnerPolicyObject = asObject(strategy.winnerPolicy);
  const winnerPolicy = winnerPolicyObject ? {
    method: asString(winnerPolicyObject.method),
    minSampleSizePerVariant: asNumber(winnerPolicyObject.minSampleSizePerVariant),
    confidence: asNumber(winnerPolicyObject.confidence),
  } : undefined;
  const winnerPolicyHasValue = winnerPolicy && Object.values(winnerPolicy).some((item) => item !== undefined);
  const variants: CampaignVariant[] = asObjects(strategy.variants).flatMap((item) => {
    const id = asString(item.variantId);
    if (!id) return [];
    return [{
      id,
      name: asString(item.name) || id,
      trafficRatio: asNumber(item.trafficRatio),
      channelKey: asString(item.channelKey),
      templateKey: asString(item.templateKey),
      offerCode: asString(item.offerCode),
    }];
  });
  const abTest: CampaignAbTest = {
    enabled: asBoolean(strategy.enabled),
    unit: asString(strategy.unit),
    primaryMetric: asString(strategy.primaryMetric),
    waveIds: asStrings(scope?.waveIds),
    subSegmentKeys: asStrings(scope?.subSegmentKeys),
    variants,
    winnerPolicy: winnerPolicyHasValue ? winnerPolicy : undefined,
  };
  const hasValue = Object.entries(abTest)
    .some(([, item]) => Array.isArray(item) ? item.length > 0 : item !== undefined);
  return hasValue ? abTest : undefined;
}

function parseStatistics(value: unknown): CampaignStatistics | undefined {
  const statistics = asObject(value);
  if (!statistics) return undefined;
  const audience = asObject(statistics.audience);
  const delivery = asObject(statistics.delivery);
  const conversion = asObject(statistics.conversion);
  const revenue = asObject(statistics.revenue);
  if (!audience && !delivery && !conversion && !revenue) return undefined;
  return {
    audience: audience ? {
      targetCount: asNumber(audience.targetCount),
      controlCount: asNumber(audience.controlCount),
      treatmentCount: asNumber(audience.treatmentCount),
    } : undefined,
    delivery: delivery ? {
      sent: asNumber(delivery.sent),
      delivered: asNumber(delivery.delivered),
      opened: asNumber(delivery.opened),
      clicked: asNumber(delivery.clicked),
      failed: asNumber(delivery.failed),
    } : undefined,
    conversion: conversion ? {
      converted: asNumber(conversion.converted),
      conversionRate: conversion.conversionRate === null ? null : asNumber(conversion.conversionRate),
      incrementalLift: conversion.incrementalLift === null ? null : asNumber(conversion.incrementalLift),
    } : undefined,
    revenue: revenue ? {
      currency: asString(revenue.currency),
      grossRevenue: asNumber(revenue.grossRevenue),
      incrementalRevenue: revenue.incrementalRevenue === null ? null : asNumber(revenue.incrementalRevenue),
      offerCost: asNumber(revenue.offerCost),
      grossMargin: revenue.grossMargin === null ? null : asNumber(revenue.grossMargin),
    } : undefined,
  };
}

export function buildCampaignPresentation(campaign: MarketingCampaignVO): CampaignPresentation {
  return {
    segmentation: parseSegmentation(campaign.segmentationStrategy),
    controlGroup: parseControlGroup(campaign.controlGroupStrategy),
    contentChannel: parseContentChannel(campaign.contentChannelStrategy),
    offer: parseOffer(campaign.offerStrategy),
    wave: parseWave(campaign.waveStrategy),
    abTest: parseAbTest(campaign.abTestStrategy),
    statistics: parseStatistics(campaign.statistics),
  };
}
