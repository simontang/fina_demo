import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { message } from "antd";
import { useApi, useConversationContext } from "@axiom-lattice/react-sdk";
import { CDP_API_BASE, getErrorMessage, unwrapCdpResponse } from "../shared/cdp";
import type { CdpApiResponse } from "../shared/cdp";
import type {
  SegmentDataPage,
  SegmentDataRow,
  SegmentDataVO,
  SegmentDefinitionPage,
  SegmentDefinitionVO,
} from "./types";

const ARTIFACT_PAGE_SIZE = 20;
const SEARCH_DELAY_MS = 300;

interface RetryRequest {
  page: number;
  append: boolean;
}

function parseSegmentRows(dataJson: string): SegmentDataRow[] {
  try {
    const parsed: unknown = JSON.parse(dataJson);
    return Array.isArray(parsed) ? parsed as SegmentDataRow[] : [];
  } catch {
    return [];
  }
}

function normalizeInitialId(initialKey: unknown): number | null {
  const value = Number(initialKey);
  return Number.isFinite(value) && value > 0 ? value : null;
}

function mergeSegments(
  current: SegmentDefinitionVO[],
  incoming: SegmentDefinitionVO[],
): SegmentDefinitionVO[] {
  const merged = new Map(current.map((segment) => [segment.id, segment]));
  incoming.forEach((segment) => merged.set(segment.id, segment));
  return Array.from(merged.values());
}

function buildSegmentPageUrl(query: string, page: number): string {
  const params = new URLSearchParams({
    page: String(page),
    pageSize: String(ARTIFACT_PAGE_SIZE),
  });
  if (query) params.set("keyword", query);
  return `${CDP_API_BASE}/segment-definitions/page?${params.toString()}`;
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
  const initialIdRef = useRef(normalizeInitialId(initialKey));
  const initialSelectionAppliedRef = useRef(false);
  const queryRef = useRef("");
  const retryRequestRef = useRef<RetryRequest>({ page: 1, append: false });
  const selectedIdRef = useRef<number | null>(null);
  const segmentListRequestId = useRef(0);
  const segmentDataRequestId = useRef(0);
  const segmentProcessRequestId = useRef(0);

  const [segments, setSegments] = useState<SegmentDefinitionVO[]>([]);
  const [searchValue, setSearchValue] = useState("");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [overallTotal, setOverallTotal] = useState(0);
  const [matchedTotal, setMatchedTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSegment, setSelectedSegment] = useState<SegmentDefinitionVO | null>(null);
  const [segmentData, setSegmentData] = useState<SegmentDataVO | null>(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [dataError, setDataError] = useState<string | null>(null);
  const [processingId, setProcessingId] = useState<number | null>(null);

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

  const loadSegmentData = useCallback(async (definitionId: number) => {
    if (selectedIdRef.current !== definitionId) return;

    const requestId = ++segmentDataRequestId.current;
    setDataLoading(true);
    setDataError(null);
    setSegmentData(null);
    try {
      const params = new URLSearchParams({ definitionId: String(definitionId), pageSize: "1" });
      const response = await get<CdpApiResponse<SegmentDataPage>>(
        `${CDP_API_BASE}/segment-data?${params.toString()}`,
      );
      const { items } = unwrapCdpResponse(response);
      if (requestId !== segmentDataRequestId.current || selectedIdRef.current !== definitionId) return;
      setSegmentData(Array.isArray(items) ? items[0] ?? null : null);
    } catch (requestError: unknown) {
      if (requestId === segmentDataRequestId.current && selectedIdRef.current === definitionId) {
        setSegmentData(null);
        setDataError(getErrorMessage(requestError, "Failed to load segment data"));
      }
    } finally {
      if (requestId === segmentDataRequestId.current && selectedIdRef.current === definitionId) {
        setDataLoading(false);
      }
    }
  }, [get]);

  const selectSegment = useCallback((segment: SegmentDefinitionVO, resetSearch = true) => {
    const changed = selectedIdRef.current !== segment.id;
    if (changed) {
      segmentDataRequestId.current += 1;
      setSegmentData(null);
      setDataError(null);
    }
    selectedIdRef.current = segment.id;
    setSelectedSegment(segment);
    if (resetSearch) clearSearch();
    if (segment.threadId) selectThread(segment.threadId);
    if (changed) void loadSegmentData(segment.id);
  }, [clearSearch, loadSegmentData, selectThread]);

  const clearSelection = useCallback(() => {
    segmentDataRequestId.current += 1;
    selectedIdRef.current = null;
    setSelectedSegment(null);
    setSegmentData(null);
    setDataLoading(false);
    setDataError(null);
  }, []);

  const loadSegmentOptions = useCallback(async (
    requestedQuery: string,
    requestedPage: number,
    append: boolean,
    preferredId: number | null = null,
    autoSelect = false,
  ): Promise<SegmentDefinitionPage | null> => {
    const requestId = ++segmentListRequestId.current;
    retryRequestRef.current = { page: requestedPage, append };
    setError(null);
    if (append) {
      setLoadingMore(true);
    } else {
      setLoading(true);
      setLoadingMore(false);
    }

    try {
      const pageRequest = get<CdpApiResponse<SegmentDefinitionPage>>(
        buildSegmentPageUrl(requestedQuery, requestedPage),
      );
      const preferredRequest: Promise<SegmentDefinitionVO | null> = preferredId
        ? get<CdpApiResponse<SegmentDefinitionVO>>(
            `${CDP_API_BASE}/segment-definitions/${preferredId}`,
          ).then(unwrapCdpResponse).catch(() => null)
        : Promise.resolve(null);
      const [pageResult, preferredResult] = await Promise.allSettled([pageRequest, preferredRequest]);
      if (requestId !== segmentListRequestId.current || queryRef.current !== requestedQuery) return null;
      const preferredSegment = preferredResult.status === "fulfilled" ? preferredResult.value : null;
      if (pageResult.status === "rejected") {
        if (autoSelect && preferredSegment) {
          initialSelectionAppliedRef.current = true;
          selectSegment(preferredSegment, requestedQuery === "");
        }
        throw pageResult.reason;
      }
      const pageResponse = pageResult.value;
      const segmentPage = unwrapCdpResponse(pageResponse);
      const items = Array.isArray(segmentPage.items) ? segmentPage.items : [];
      setSegments((current) => append ? mergeSegments(current, items) : items);
      setPage(segmentPage.page);
      setMatchedTotal(segmentPage.total);
      if (!requestedQuery) setOverallTotal(segmentPage.total);

      if (autoSelect) {
        const nextSelection = preferredSegment ?? items[0] ?? null;
        initialSelectionAppliedRef.current = true;
        if (nextSelection) selectSegment(nextSelection, requestedQuery === "");
        else clearSelection();
      }
      return segmentPage;
    } catch (requestError: unknown) {
      if (requestId !== segmentListRequestId.current || queryRef.current !== requestedQuery) return null;
      setError(getErrorMessage(requestError, "Failed to load segments"));
      return null;
    } finally {
      if (requestId === segmentListRequestId.current) {
        setLoading(false);
        setLoadingMore(false);
      }
    }
  }, [clearSelection, get, selectSegment]);

  useEffect(() => {
    const needsInitialSelection = !initialSelectionAppliedRef.current;
    const preferredId = needsInitialSelection && query === "" ? initialIdRef.current : null;
    void loadSegmentOptions(
      query,
      1,
      false,
      preferredId,
      needsInitialSelection || selectedIdRef.current === null,
    );
  }, [loadSegmentOptions, query]);

  useEffect(() => () => {
    segmentListRequestId.current += 1;
    segmentDataRequestId.current += 1;
    segmentProcessRequestId.current += 1;
  }, []);

  const loadMore = useCallback(() => {
    if (loading || loadingMore || error || segments.length >= matchedTotal) return;
    void loadSegmentOptions(queryRef.current, page + 1, true);
  }, [error, loadSegmentOptions, loading, loadingMore, matchedTotal, page, segments.length]);

  const retryList = useCallback(() => {
    const retry = retryRequestRef.current;
    const preferredId = !initialSelectionAppliedRef.current && queryRef.current === ""
      ? initialIdRef.current
      : null;
    void loadSegmentOptions(
      queryRef.current,
      retry.page,
      retry.append,
      preferredId,
      selectedIdRef.current === null,
    );
  }, [loadSegmentOptions]);

  const retrySegmentData = useCallback(() => {
    const selectedId = selectedIdRef.current;
    if (selectedId) void loadSegmentData(selectedId);
  }, [loadSegmentData]);

  const deleteSelectedSegment = useCallback(async () => {
    const targetId = selectedIdRef.current;
    if (!targetId) return;

    try {
      const response = await del<CdpApiResponse<null>>(
        `${CDP_API_BASE}/segment-definitions/${targetId}`,
      );
      unwrapCdpResponse(response);
      message.success("Segment deleted");
      clearSelection();
      setSegments((current) => current.filter((segment) => segment.id !== targetId));

      setOverallTotal((current) => Math.max(0, current - 1));

      const activeQuery = queryRef.current;
      const refreshedPage = await loadSegmentOptions(activeQuery, 1, false, null, true);
      if (refreshedPage && activeQuery && refreshedPage.total === 0) clearSearch();
    } catch (requestError: unknown) {
      message.error(getErrorMessage(requestError, "Delete failed"));
    }
  }, [clearSearch, clearSelection, del, loadSegmentOptions]);

  const processSelectedSegment = useCallback(async () => {
    const targetId = selectedIdRef.current;
    if (!targetId) return;

    const requestId = ++segmentProcessRequestId.current;
    setProcessingId(targetId);
    try {
      const response = await post<CdpApiResponse<SegmentDataVO>>(
        `${CDP_API_BASE}/segment-definitions/${targetId}/process`,
        {},
      );
      unwrapCdpResponse(response);
      message.success("Process completed");
      if (selectedIdRef.current === targetId) void loadSegmentData(targetId);
    } catch (requestError: unknown) {
      message.error(getErrorMessage(requestError, "Process failed"));
    } finally {
      if (requestId === segmentProcessRequestId.current) {
        setProcessingId((currentId) => currentId === targetId ? null : currentId);
      }
    }
  }, [loadSegmentData, post]);

  const rows = useMemo(
    () => segmentData ? parseSegmentRows(segmentData.dataJson) : [],
    [segmentData],
  );

  return {
    segments,
    searchValue,
    query,
    overallTotal,
    matchedTotal,
    loading,
    loadingMore,
    error,
    selectedId: selectedSegment?.id ?? null,
    selectedSegment,
    segmentData,
    dataLoading,
    dataError,
    processing: processingId !== null && processingId === selectedSegment?.id,
    rows,
    setSearchValue,
    selectSegment,
    loadMore,
    retryList,
    retrySegmentData,
    deleteSelectedSegment,
    processSelectedSegment,
  };
}
