package com.fina.b1s.service;

import com.fina.b1s.dto.TableViewDetailResponse;
import com.fina.b1s.dto.TableViewIndexItem;

import java.util.List;

/**
 * Provides read-only access to Table/View meta from meta/view-*.json and
 * meta/VW_*.csv. Loaded once at startup and cached.
 */
public interface TableViewMetaService {

    /** Lightweight index of all known tables/views (view-*.json first, then CSV-only). */
    List<TableViewIndexItem> getTableViewsIndex();

    /** Full detail per table/view (same order as index). */
    List<TableViewDetailResponse> getTableViewsDetails();
}
