import { useCallback, useEffect, useRef, useState } from "react";
import { message } from "antd";
import { useApi } from "@axiom-lattice/react-sdk";
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

export function useCampaignWorkbench(initialKey: unknown) {
  const { get, post, del } = useApi();
  const [campaigns, setCampaigns] = useState<MarketingCampaignVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(
    initialKey ? Number(initialKey) : null,
  );
  const [selectedCampaign, setSelectedCampaign] = useState<MarketingCampaignVO | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const campaignListRequestId = useRef(0);
  const campaignDetailRequestId = useRef(0);

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

  const loadCampaignDetail = useCallback(async (id: number) => {
    const requestId = ++campaignDetailRequestId.current;
    setDetailLoading(true);
    try {
      const response = await get<CdpApiResponse<MarketingCampaignVO>>(
        `${CDP_API_BASE}/marketing-campaigns/${id}`,
      );
      const detail = unwrapCdpResponse(response);
      if (requestId !== campaignDetailRequestId.current) return;
      setSelectedCampaign(detail);
    } catch {
      if (requestId === campaignDetailRequestId.current) setSelectedCampaign(null);
    } finally {
      if (requestId === campaignDetailRequestId.current) setDetailLoading(false);
    }
  }, [get]);

  useEffect(() => {
    if (selectedId) {
      void loadCampaignDetail(selectedId);
    } else {
      campaignDetailRequestId.current += 1;
      setSelectedCampaign(null);
    }
    return () => {
      campaignDetailRequestId.current += 1;
    };
  }, [selectedId, loadCampaignDetail]);

  const deleteSelectedCampaign = useCallback(async () => {
    if (!selectedId) return;
    try {
      const response = await del<CdpApiResponse<null>>(
        `${CDP_API_BASE}/marketing-campaigns/${selectedId}`,
      );
      unwrapCdpResponse(response);
      message.success("Campaign deleted");
      setSelectedId(null);
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
      void loadCampaignDetail(selectedId);
      void loadCampaigns();
    } catch (requestError: unknown) {
      message.error(getErrorMessage(requestError, action === "start" ? "Start failed" : "Stop failed"));
    } finally {
      setActionLoading(false);
    }
  }, [loadCampaignDetail, loadCampaigns, post, selectedId]);

  const startSelectedCampaign = useCallback(
    () => runStatusAction("start"),
    [runStatusAction],
  );
  const stopSelectedCampaign = useCallback(
    () => runStatusAction("stop"),
    [runStatusAction],
  );
  const selectCampaign = useCallback((id: number) => {
    setSelectedId(id);
  }, []);

  return {
    campaigns,
    loading,
    error,
    selectedId,
    selectedCampaign,
    detailLoading,
    actionLoading,
    selectCampaign,
    deleteSelectedCampaign,
    startSelectedCampaign,
    stopSelectedCampaign,
  };
}
