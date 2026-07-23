import { useCallback, useEffect, useRef, useState } from "react";
import { message } from "antd";
import { useApi, useConversationContext } from "@axiom-lattice/react-sdk";
import { CDP_API_BASE, getErrorMessage, unwrapCdpResponse } from "../shared/cdp";
import type { CdpApiResponse } from "../shared/cdp";
import type { MarketingCampaignPage, MarketingCampaignVO } from "./types";

export function useCampaignSummary(): MarketingCampaignVO[] {
  const [campaigns, setCampaigns] = useState<MarketingCampaignVO[]>([]);
  const { get } = useApi();

  useEffect(() => {
    let active = true;

    void get<CdpApiResponse<MarketingCampaignPage>>(
      `${CDP_API_BASE}/marketing-campaigns?type=reactivation&pageSize=5`,
    )
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

function normalizeInitialId(initialKey: unknown): number | null {
  const value = Number(initialKey);
  return Number.isFinite(value) && value > 0 ? value : null;
}

export function useCampaignWorkbench(initialKey: unknown) {
  const { get, post, del } = useApi();
  const { selectThread } = useConversationContext();
  const initialIdRef = useRef(normalizeInitialId(initialKey));
  const initialSelectionApplied = useRef(false);
  const [campaigns, setCampaigns] = useState<MarketingCampaignVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(initialIdRef.current);
  const [selectedCampaign, setSelectedCampaign] = useState<MarketingCampaignVO | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const campaignListRequestId = useRef(0);
  const campaignDetailRequestId = useRef(0);
  const selectedIdRef = useRef<number | null>(initialIdRef.current);

  const loadCampaigns = useCallback(async () => {
    const requestId = ++campaignListRequestId.current;
    setLoading(true);
    setError(null);
    try {
      const response = await get<CdpApiResponse<MarketingCampaignPage>>(
        `${CDP_API_BASE}/marketing-campaigns?type=reactivation`,
      );
      const page = unwrapCdpResponse(response);
      if (requestId !== campaignListRequestId.current) return;
      setCampaigns(Array.isArray(page.items) ? page.items : []);
    } catch (requestError: unknown) {
      if (requestId !== campaignListRequestId.current) return;
      setError(getErrorMessage(requestError, "Failed to load campaigns"));
    } finally {
      if (requestId === campaignListRequestId.current) setLoading(false);
    }
  }, [get]);

  useEffect(() => {
    void loadCampaigns();
    return () => {
      campaignListRequestId.current += 1;
    };
  }, [loadCampaigns]);

  const loadCampaignBundle = useCallback(async (campaign: MarketingCampaignVO) => {
    const requestId = ++campaignDetailRequestId.current;
    setDetailLoading(true);
    setDetailError(null);
    setSelectedCampaign(null);

    try {
      const response = await get<CdpApiResponse<MarketingCampaignVO>>(
        `${CDP_API_BASE}/marketing-campaigns/${campaign.id}`,
      );
      if (requestId !== campaignDetailRequestId.current) return;
      setSelectedCampaign(unwrapCdpResponse(response));
    } catch (requestError: unknown) {
      if (requestId !== campaignDetailRequestId.current) return;
      setDetailError(getErrorMessage(requestError, "Failed to load campaign detail"));
    } finally {
      if (requestId === campaignDetailRequestId.current) setDetailLoading(false);
    }
  }, [get]);

  const selectCampaign = useCallback((campaign: MarketingCampaignVO) => {
    initialSelectionApplied.current = true;
    selectedIdRef.current = campaign.id;
    setSelectedId(campaign.id);
    void loadCampaignBundle(campaign);
    if (campaign.threadId) {
      selectThread(campaign.threadId);
    }
  }, [loadCampaignBundle, selectThread]);

  const clearSelection = useCallback(() => {
    campaignDetailRequestId.current += 1;
    selectedIdRef.current = null;
    setSelectedId(null);
    setSelectedCampaign(null);
    setDetailLoading(false);
    setDetailError(null);
  }, []);

  useEffect(() => {
    const initialId = initialIdRef.current;
    if (initialSelectionApplied.current || initialId == null || campaigns.length === 0) return;
    initialSelectionApplied.current = true;
    const initialCampaign = campaigns.find((campaign) => campaign.id === initialId);
    if (initialCampaign) void loadCampaignBundle(initialCampaign);
  }, [campaigns, loadCampaignBundle]);

  useEffect(() => () => {
    campaignDetailRequestId.current += 1;
  }, []);

  const reloadCampaignDetail = useCallback(async (id: number) => {
    try {
      const response = await get<CdpApiResponse<MarketingCampaignVO>>(
        `${CDP_API_BASE}/marketing-campaigns/${id}`,
      );
      if (selectedIdRef.current === id) setSelectedCampaign(unwrapCdpResponse(response));
    } catch (requestError: unknown) {
      if (selectedIdRef.current === id) {
        setDetailError(getErrorMessage(requestError, "Failed to refresh campaign detail"));
      }
    }
  }, [get]);

  const deleteSelectedCampaign = useCallback(async () => {
    if (!selectedId) return;
    try {
      const response = await del<CdpApiResponse<null>>(
        `${CDP_API_BASE}/marketing-campaigns/${selectedId}`,
      );
      unwrapCdpResponse(response);
      message.success("Campaign deleted");
      campaignDetailRequestId.current += 1;
      selectedIdRef.current = null;
      setSelectedId(null);
      setSelectedCampaign(null);
      void loadCampaigns();
    } catch (requestError: unknown) {
      message.error(getErrorMessage(requestError, "Delete failed"));
    }
  }, [del, loadCampaigns, selectedId]);

  const runStatusAction = useCallback(async (action: "start" | "stop") => {
    if (!selectedId) return;
    setActionLoading(true);
    try {
      const response = await post<CdpApiResponse<MarketingCampaignVO>>(
        `${CDP_API_BASE}/marketing-campaigns/${selectedId}/${action}`,
        {},
      );
      unwrapCdpResponse(response);
      message.success(action === "start" ? "Campaign started" : "Campaign stopped");
      void reloadCampaignDetail(selectedId);
      void loadCampaigns();
    } catch (requestError: unknown) {
      message.error(getErrorMessage(requestError, action === "start" ? "Start failed" : "Stop failed"));
    } finally {
      setActionLoading(false);
    }
  }, [loadCampaigns, post, reloadCampaignDetail, selectedId]);

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
    loading,
    error,
    selectedId,
    selectedCampaign,
    detailLoading,
    detailError,
    actionLoading,
    selectCampaign,
    clearSelection,
    deleteSelectedCampaign,
    startSelectedCampaign,
    stopSelectedCampaign,
  };
}
