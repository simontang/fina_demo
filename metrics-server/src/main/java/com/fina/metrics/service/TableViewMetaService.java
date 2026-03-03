package com.fina.metrics.service;

import com.fina.metrics.dto.TableViewDetailResponse;
import com.fina.metrics.dto.TableViewIndexItem;

import java.util.List;

/**
 * Provides read-only access to Table/View meta from meta/view-*.json and
 * meta/MTC_VW_AI_*.csv. Loaded once at startup and cached.
 */
public interface TableViewMetaService {

    /** Lightweight index of all known tables/views (view-*.json first, then CSV-only). */
    List<TableViewIndexItem> getTableViewsIndex();

    /** Full detail per table/view (same order as index). */
    List<TableViewDetailResponse> getTableViewsDetails();
}
