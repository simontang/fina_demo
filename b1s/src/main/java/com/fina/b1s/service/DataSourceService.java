package com.fina.b1s.service;

import com.fina.b1s.dto.DataSourceRequest;
import com.fina.b1s.dto.DataSourceUpdateRequest;
import com.fina.b1s.dto.DataSourceVO;

import java.util.List;
import java.util.Map;

public interface DataSourceService {

    /** List all non-deleted datasources */
    List<DataSourceVO> listAll();

    /** List only active (status=1) datasources */
    List<DataSourceVO> listActive();

    /** Get a single datasource by ID */
    DataSourceVO getById(Long id);

    /**
     * Create a new SAP B1 SQL Server datasource and register its connection pool.
     * Password in the request is plain-text and will be AES-encrypted before storing.
     */
    DataSourceVO create(DataSourceRequest request);

    /**
     * Update an existing datasource and refresh its connection pool.
     * If {@code request.getPassword()} is blank, the existing encrypted password is kept.
     */
    DataSourceVO update(Long id, DataSourceUpdateRequest request);

    /** Soft-delete a datasource and close its pool */
    void delete(Long id);

    /**
     * Set datasource status to 1 (active) and register its connection pool.
     */
    DataSourceVO enable(Long id);

    /**
     * Set datasource status to 0 (inactive) and close its connection pool.
     */
    DataSourceVO disable(Long id);

    /**
     * Set arbitrary status value (0 or 1) for a datasource.
     * Pool is registered/removed accordingly.
     */
    DataSourceVO setStatus(Long id, Integer status);

    /**
     * Test connectivity using the provided plain-text credentials.
     * Does NOT persist anything.
     */
    boolean testConnection(DataSourceRequest request);

    /**
     * Test connectivity of an already-saved datasource using its stored credentials.
     * Returns a detailed result map with {@code connected} and {@code message}.
     */
    Map<String, Object> testConnectionById(Long id);

    /**
     * Reload a datasource pool from its current DB config.
     * Useful after external credential changes.
     */
    void reload(Long id);

    /**
     * Return HikariCP pool metrics for a registered datasource.
     * Returns {@code registered=false} when the pool is not active.
     */
    Map<String, Object> getPoolStatus(Long id);
}
