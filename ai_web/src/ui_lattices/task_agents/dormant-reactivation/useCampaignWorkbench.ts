import { useCallback, useEffect, useRef, useState } from "react";
import { message } from "antd";
import { useApi, useConversationContext } from "@axiom-lattice/react-sdk";
import { CDP_API_BASE, getErrorMessage, unwrapCdpResponse } from "../shared/cdp";
import type { CdpApiResponse } from "../shared/cdp";
import type { MarketingCampaignPage, MarketingCampaignVO } from "./types";

const ARTIFACT_PAGE_SIZE = 20;
const SEARCH_DELAY_MS = 300;

interface RetryRequest {
  page: number;
  append: boolean;
}

function normalizeInitialId(initialKey: unknown): number | null {
  const value = Number(initialKey);
  return Number.isFinite(value) && value > 0 ? value : null;
}

function mergeCampaigns(
  current: MarketingCampaignVO[],
  incoming: MarketingCampaignVO[],
): MarketingCampaignVO[] {
  const merged = new Map(current.map((campaign) => [campaign.id, campaign]));
  incoming.forEach((campaign) => merged.set(campaign.id, campaign));
  return Array.from(merged.values());
}

function buildCampaignPageUrl(query: string, page: number, pageSize = ARTIFACT_PAGE_SIZE): string {
  const params = new URLSearchParams({
    type: "reactivation",
    page: String(page),
    pageSize: String(pageSize),
  });
  if (query) params.set("keyword", query);
  return `${CDP_API_BASE}/marketing-campaigns?${params.toString()}`;
}

export function useCampaignSummary(): MarketingCampaignVO[] {
  const [campaigns, setCampaigns] = useState<MarketingCampaignVO[]>([]);
  const { get } = useApi();

  useEffect(() => {
    let active = true;

    void get<CdpApiResponse<MarketingCampaignPage>>(buildCampaignPageUrl("", 1, 5))
      .then((response) => {
        const page = unwrapCdpResponse(response);
        if (active) setCampaigns(Array.isArray(page.items) ? page.items : []);
      })
      .catch(() => {});

    return () => {
      active = false;
    };
  }, [get]);

  return campaigns;
}

export function useCampaignWorkbench(initialKey: unknown) {
  const { get, post, del } = useApi();
  const { selectThread } = useConversationContext();
  const initialIdRef = useRef(normalizeInitialId(initialKey));
  const initialSelectionAppliedRef = useRef(false);
  const queryRef = useRef("");
  const retryRequestRef = useRef<RetryRequest>({ page: 1, append: false });
  const selectedIdRef = useRef<number | null>(null);
  const campaignListRequestId = useRef(0);
  const campaignDetailRequestId = useRef(0);
  const campaignActionRequestId = useRef(0);

  const [campaigns, setCampaigns] = useState<MarketingCampaignVO[]>([]);
  const [searchValue, setSearchValue] = useState("");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [overallTotal, setOverallTotal] = useState(0);
  const [matchedTotal, setMatchedTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedOption, setSelectedOption] = useState<MarketingCampaignVO | null>(null);
  const [selectedCampaign, setSelectedCampaign] = useState<MarketingCampaignVO | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [actionLoadingId, setActionLoadingId] = useState<number | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const normalizedQuery = searchValue.trim();
      queryRef.current = normalizedQuery;
      setQuery(normalizedQuery);
    }, SEARCH_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [searchValue]);

  const clearSearch = useCallback(() => {
    queryRef.current = "";
    setSearchValue("");
    setQuery("");
  }, []);

  const loadCampaignDetail = useCallback(async (campaignId: number) => {
    if (selectedIdRef.current !== campaignId) return;

    const requestId = ++campaignDetailRequestId.current;
    setDetailLoading(true);
    setDetailError(null);
    try {
      const response = await get<CdpApiResponse<MarketingCampaignVO>>(
        `${CDP_API_BASE}/marketing-campaigns/${campaignId}`,
      );
      if (requestId !== campaignDetailRequestId.current || selectedIdRef.current !== campaignId) return;
      const campaign = unwrapCdpResponse(response);
      setSelectedCampaign(campaign);
      setSelectedOption(campaign);
    } catch (requestError: unknown) {
      if (requestId === campaignDetailRequestId.current && selectedIdRef.current === campaignId) {
        setDetailError(getErrorMessage(requestError, "Failed to load campaign detail"));
      }
    } finally {
      if (requestId === campaignDetailRequestId.current && selectedIdRef.current === campaignId) {
        setDetailLoading(false);
      }
    }
  }, [get]);

  const selectCampaign = useCallback((campaign: MarketingCampaignVO, resetSearch = true) => {
    const changed = selectedIdRef.current !== campaign.id;
    if (changed) {
      campaignDetailRequestId.current += 1;
      setSelectedCampaign(null);
      setDetailError(null);
    }
    selectedIdRef.current = campaign.id;
    setSelectedOption(campaign);
    if (resetSearch) clearSearch();
    if (campaign.threadId) selectThread(campaign.threadId);
    if (changed) void loadCampaignDetail(campaign.id);
  }, [clearSearch, loadCampaignDetail, selectThread]);

  const clearSelection = useCallback(() => {
    campaignDetailRequestId.current += 1;
    selectedIdRef.current = null;
    setSelectedOption(null);
    setSelectedCampaign(null);
    setDetailLoading(false);
    setDetailError(null);
  }, []);

  const loadCampaignOptions = useCallback(async (
    requestedQuery: string,
    requestedPage: number,
    append: boolean,
    preferredId: number | null = null,
    autoSelect = false,
  ): Promise<MarketingCampaignPage | null> => {
    const requestId = ++campaignListRequestId.current;
    retryRequestRef.current = { page: requestedPage, append };
    setError(null);
    if (append) {
      setLoadingMore(true);
    } else {
      setLoading(true);
      setLoadingMore(false);
    }

    try {
      const pageRequest = get<CdpApiResponse<MarketingCampaignPage>>(
        buildCampaignPageUrl(requestedQuery, requestedPage),
      );
      const preferredRequest: Promise<MarketingCampaignVO | null> = preferredId
        ? get<CdpApiResponse<MarketingCampaignVO>>(
            `${CDP_API_BASE}/marketing-campaigns/${preferredId}`,
          )
            .then(unwrapCdpResponse)
            .then((campaign) => campaign.type === "reactivation" ? campaign : null)
            .catch(() => null)
        : Promise.resolve(null);
      const [pageResult, preferredResult] = await Promise.allSettled([pageRequest, preferredRequest]);
      if (requestId !== campaignListRequestId.current || queryRef.current !== requestedQuery) return null;
      const preferredCampaign = preferredResult.status === "fulfilled" ? preferredResult.value : null;
      if (pageResult.status === "rejected") {
        if (autoSelect && preferredCampaign) {
          initialSelectionAppliedRef.current = true;
          selectCampaign(preferredCampaign, requestedQuery === "");
        }
        throw pageResult.reason;
      }
      const pageResponse = pageResult.value;
      const campaignPage = unwrapCdpResponse(pageResponse);
      const items = Array.isArray(campaignPage.items) ? campaignPage.items : [];
      setCampaigns((current) => append ? mergeCampaigns(current, items) : items);
      setPage(campaignPage.page);
      setMatchedTotal(campaignPage.total);
      if (!requestedQuery) setOverallTotal(campaignPage.total);

      if (autoSelect) {
        const nextSelection = preferredCampaign ?? items[0] ?? null;
        initialSelectionAppliedRef.current = true;
        if (nextSelection) selectCampaign(nextSelection, requestedQuery === "");
        else clearSelection();
      }
      return campaignPage;
    } catch (requestError: unknown) {
      if (requestId !== campaignListRequestId.current || queryRef.current !== requestedQuery) return null;
      setError(getErrorMessage(requestError, "Failed to load campaigns"));
      return null;
    } finally {
      if (requestId === campaignListRequestId.current) {
        setLoading(false);
        setLoadingMore(false);
      }
    }
  }, [clearSelection, get, selectCampaign]);

  useEffect(() => {
    const needsInitialSelection = !initialSelectionAppliedRef.current;
    const preferredId = needsInitialSelection && query === "" ? initialIdRef.current : null;
    void loadCampaignOptions(
      query,
      1,
      false,
      preferredId,
      needsInitialSelection || selectedIdRef.current === null,
    );
  }, [loadCampaignOptions, query]);

  useEffect(() => () => {
    campaignListRequestId.current += 1;
    campaignDetailRequestId.current += 1;
    campaignActionRequestId.current += 1;
  }, []);

  const loadMore = useCallback(() => {
    if (loading || loadingMore || error || campaigns.length >= matchedTotal) return;
    void loadCampaignOptions(queryRef.current, page + 1, true);
  }, [campaigns.length, error, loadCampaignOptions, loading, loadingMore, matchedTotal, page]);

  const retryList = useCallback(() => {
    const retry = retryRequestRef.current;
    const preferredId = !initialSelectionAppliedRef.current && queryRef.current === ""
      ? initialIdRef.current
      : null;
    void loadCampaignOptions(
      queryRef.current,
      retry.page,
      retry.append,
      preferredId,
      selectedIdRef.current === null,
    );
  }, [loadCampaignOptions]);

  const retryCampaignDetail = useCallback(() => {
    const selectedId = selectedIdRef.current;
    if (selectedId) void loadCampaignDetail(selectedId);
  }, [loadCampaignDetail]);

  const deleteSelectedCampaign = useCallback(async () => {
    const targetId = selectedIdRef.current;
    if (!targetId) return;

    try {
      const response = await del<CdpApiResponse<null>>(
        `${CDP_API_BASE}/marketing-campaigns/${targetId}`,
      );
      unwrapCdpResponse(response);
      message.success("Campaign deleted");
      clearSelection();
      setCampaigns((current) => current.filter((campaign) => campaign.id !== targetId));

      setOverallTotal((current) => Math.max(0, current - 1));

      const activeQuery = queryRef.current;
      const refreshedPage = await loadCampaignOptions(activeQuery, 1, false, null, true);
      if (refreshedPage && activeQuery && refreshedPage.total === 0) clearSearch();
    } catch (requestError: unknown) {
      message.error(getErrorMessage(requestError, "Delete failed"));
    }
  }, [clearSearch, clearSelection, del, loadCampaignOptions]);

  const runStatusAction = useCallback(async (action: "start" | "stop") => {
    const targetId = selectedIdRef.current;
    if (!targetId) return;

    const requestId = ++campaignActionRequestId.current;
    setActionLoadingId(targetId);
    try {
      const response = await post<CdpApiResponse<MarketingCampaignVO>>(
        `${CDP_API_BASE}/marketing-campaigns/${targetId}/${action}`,
        {},
      );
      const updatedCampaign = unwrapCdpResponse(response);
      message.success(action === "start" ? "Campaign started" : "Campaign stopped");
      if (selectedIdRef.current === targetId) {
        setSelectedCampaign(updatedCampaign);
        setSelectedOption(updatedCampaign);
      }
      setCampaigns((current) => current.map((campaign) => (
        campaign.id === targetId ? updatedCampaign : campaign
      )));
    } catch (requestError: unknown) {
      message.error(getErrorMessage(requestError, action === "start" ? "Start failed" : "Stop failed"));
    } finally {
      if (requestId === campaignActionRequestId.current) {
        setActionLoadingId((currentId) => currentId === targetId ? null : currentId);
      }
    }
  }, [post]);

  const startSelectedCampaign = useCallback(
    () => runStatusAction("start"),
    [runStatusAction],
  );
  const stopSelectedCampaign = useCallback(
    () => runStatusAction("stop"),
    [runStatusAction],
  );

  return {
    campaigns,
    searchValue,
    query,
    overallTotal,
    matchedTotal,
    loading,
    loadingMore,
    error,
    selectedId: selectedOption?.id ?? null,
    selectedOption,
    selectedCampaign,
    detailLoading,
    detailError,
    actionLoading: actionLoadingId !== null && actionLoadingId === selectedOption?.id,
    setSearchValue,
    selectCampaign,
    loadMore,
    retryList,
    retryCampaignDetail,
    deleteSelectedCampaign,
    startSelectedCampaign,
    stopSelectedCampaign,
  };
}
