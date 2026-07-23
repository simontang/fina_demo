import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { message } from "antd";
import { useApi, useConversationContext } from "@axiom-lattice/react-sdk";
import { CDP_API_BASE, getErrorMessage, unwrapCdpResponse } from "../shared/cdp";
import type { CdpApiResponse } from "../shared/cdp";
import type { SegmentDataPage, SegmentDataRow, SegmentDataVO, SegmentDefinitionVO } from "./types";

function parseSegmentRows(dataJson: string): SegmentDataRow[] {
  try {
    const parsed: unknown = JSON.parse(dataJson);
    return Array.isArray(parsed) ? parsed as SegmentDataRow[] : [];
  } catch {
    return [];
  }
}

export function useSegmentSummary(): SegmentDefinitionVO[] {
  const [segments, setSegments] = useState<SegmentDefinitionVO[]>([]);
  const { get } = useApi();

  useEffect(() => {
    let active = true;

    void get<CdpApiResponse<SegmentDefinitionVO[]>>(`${CDP_API_BASE}/segment-definitions`)
      .then((response) => {
        const list = unwrapCdpResponse(response);
        if (active) setSegments(Array.isArray(list) ? list : []);
      })
      .catch(() => {});

    return () => {
      active = false;
    };
  }, [get]);

  return segments;
}

export function useSegmentWorkbench(initialKey: unknown) {
  const { get, post, del } = useApi();
  const { selectThread } = useConversationContext();
  const [segments, setSegments] = useState<SegmentDefinitionVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(
    initialKey ? Number(initialKey) : null,
  );
  const [segmentData, setSegmentData] = useState<SegmentDataVO | null>(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const segmentListRequestId = useRef(0);
  const segmentDataRequestId = useRef(0);

  const loadSegments = useCallback(async () => {
    const requestId = ++segmentListRequestId.current;
    setLoading(true);
    setError(null);
    try {
      const response = await get<CdpApiResponse<SegmentDefinitionVO[]>>(
        `${CDP_API_BASE}/segment-definitions`,
      );
      const list = unwrapCdpResponse(response);
      if (requestId !== segmentListRequestId.current) return;
      setSegments(Array.isArray(list) ? list : []);
    } catch (requestError: unknown) {
      if (requestId !== segmentListRequestId.current) return;
      setError(getErrorMessage(requestError, "Failed to load segments"));
    } finally {
      if (requestId === segmentListRequestId.current) setLoading(false);
    }
  }, [get]);

  useEffect(() => {
    void loadSegments();
    return () => {
      segmentListRequestId.current += 1;
    };
  }, [loadSegments]);

  const loadSegmentData = useCallback(async (definitionId: number) => {
    const requestId = ++segmentDataRequestId.current;
    setDataLoading(true);
    try {
      const response = await get<CdpApiResponse<SegmentDataPage>>(
        `${CDP_API_BASE}/segment-data?definitionId=${definitionId}&pageSize=1`,
      );
      const { items } = unwrapCdpResponse(response);
      if (requestId !== segmentDataRequestId.current) return;
      setSegmentData(Array.isArray(items) ? items[0] ?? null : null);
    } catch {
      if (requestId === segmentDataRequestId.current) setSegmentData(null);
    } finally {
      if (requestId === segmentDataRequestId.current) setDataLoading(false);
    }
  }, [get]);

  useEffect(() => {
    if (selectedId) {
      void loadSegmentData(selectedId);
    } else {
      segmentDataRequestId.current += 1;
      setSegmentData(null);
    }
    return () => {
      segmentDataRequestId.current += 1;
    };
  }, [selectedId, loadSegmentData]);

  const selectSegment = useCallback((segment: SegmentDefinitionVO) => {
    setSelectedId(segment.id);
    if (segment.threadId) {
      selectThread(segment.threadId);
    }
  }, [selectThread]);

  const clearSelection = useCallback(() => {
    segmentDataRequestId.current += 1;
    setSelectedId(null);
    setSegmentData(null);
    setDataLoading(false);
  }, []);

  const deleteSelectedSegment = useCallback(async () => {
    if (!selectedId) return;
    try {
      const response = await del<CdpApiResponse<null>>(
        `${CDP_API_BASE}/segment-definitions/${selectedId}`,
      );
      unwrapCdpResponse(response);
      message.success("Segment deleted");
      setSelectedId(null);
      void loadSegments();
    } catch (requestError: unknown) {
      message.error(getErrorMessage(requestError, "Delete failed"));
    }
  }, [del, loadSegments, selectedId]);

  const processSelectedSegment = useCallback(async () => {
    if (!selectedId) return;
    setProcessing(true);
    try {
      const response = await post<CdpApiResponse<SegmentDataVO>>(
        `${CDP_API_BASE}/segment-definitions/${selectedId}/process`,
        {},
      );
      unwrapCdpResponse(response);
      message.success("Process completed");
      void loadSegmentData(selectedId);
    } catch (requestError: unknown) {
      message.error(getErrorMessage(requestError, "Process failed"));
    } finally {
      setProcessing(false);
    }
  }, [loadSegmentData, post, selectedId]);

  const selectedSegment = segments.find((segment) => segment.id === selectedId);
  const rows = useMemo(
    () => segmentData ? parseSegmentRows(segmentData.dataJson) : [],
    [segmentData],
  );

  return {
    segments,
    loading,
    error,
    selectedId,
    selectedSegment,
    segmentData,
    dataLoading,
    processing,
    rows,
    selectSegment,
    clearSelection,
    deleteSelectedSegment,
    processSelectedSegment,
  };
}
