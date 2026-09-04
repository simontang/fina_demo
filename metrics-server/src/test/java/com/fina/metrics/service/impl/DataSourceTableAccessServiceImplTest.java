package com.fina.metrics.service.impl;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.fina.metrics.config.DynamicDataSourceManager;
import com.fina.metrics.dto.DataSourceTableGrantRequest;
import com.fina.metrics.dto.DataSourceTableVO;
import com.fina.metrics.dto.MetricsQueryData;
import com.fina.metrics.dto.SqlProbeRequest;
import com.fina.metrics.entity.DataSourceConfig;
import com.fina.metrics.entity.DataSourceTableGrant;
import com.fina.metrics.exception.ForbiddenException;
import com.fina.metrics.mapper.DataSourceConfigMapper;
import com.fina.metrics.mapper.DataSourceTableGrantMapper;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.ResultSetExtractor;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatNoException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class DataSourceTableAccessServiceImplTest {

    private static final long DATASOURCE_ID = 15L;

    private DataSourceTableGrantMapper grantMapper;
    private DataSourceConfigMapper datasourceMapper;
    private DynamicDataSourceManager dsManager;
    private DataSourceTableAccessServiceImpl service;

    @BeforeAll
    static void initializeMybatisMetadata() {
        TableInfoHelper.initTableInfo(
                new MapperBuilderAssistant(new MybatisConfiguration(), "test"),
                DataSourceTableGrant.class);
        TableInfoHelper.initTableInfo(
                new MapperBuilderAssistant(new MybatisConfiguration(), "test"),
                DataSourceConfig.class);
    }

    @BeforeEach
    void setUp() {
        grantMapper = mock(DataSourceTableGrantMapper.class);
        datasourceMapper = mock(DataSourceConfigMapper.class);
        dsManager = mock(DynamicDataSourceManager.class);
        service = new DataSourceTableAccessServiceImpl(grantMapper, datasourceMapper, dsManager);
        when(datasourceMapper.selectOne(any())).thenReturn(datasource());
        when(grantMapper.selectList(any())).thenReturn(List.of(prefixGrant("hankel_")));
    }

    @Test
    void hankelPrefixGrantOnlyAllowsHankelTables() {
        assertThat(service.isTableAuthorized("hankel", DATASOURCE_ID, null, "hankel_sales")).isTrue();
        assertThat(service.isTableAuthorized("hankel", DATASOURCE_ID, null, "retailcdp_customers")).isFalse();
    }

    @Test
    void sqlAuthorizationRejectsTablesOutsideGrant() {
        assertThatNoException().isThrownBy(() ->
                service.assertSqlAuthorized("hankel", DATASOURCE_ID, "SELECT * FROM public.hankel_sales"));
        assertThatNoException().isThrownBy(() ->
                service.assertSqlAuthorized("hankel", DATASOURCE_ID, "SELECT TOP 10 * FROM [public].[hankel_sales]"));

        assertThatThrownBy(() ->
                service.assertSqlAuthorized("hankel", DATASOURCE_ID, "SELECT * FROM t_datasource_config"))
                .isInstanceOf(ForbiddenException.class)
                .hasMessageContaining("unauthorized table");
    }

    @Test
    void readChecksFallBackToDatasourceGrantsWhenTenantHasNone() {
        when(grantMapper.selectList(any()))
                .thenReturn(List.of(), List.of(prefixGrant("hankel_")));

        assertThat(service.isTableAuthorized("default", DATASOURCE_ID, null, "hankel_sales")).isTrue();

        verify(grantMapper, times(2)).selectList(any());
    }

    @Test
    void tenantGrantTakesPrecedenceOverDatasourceFallback() {
        when(grantMapper.selectList(any()))
                .thenReturn(List.of(prefixGrant("tenant_only_")));

        assertThat(service.isTableAuthorized("default", DATASOURCE_ID, null, "hankel_sales")).isFalse();
        assertThat(service.isTableAuthorized("default", DATASOURCE_ID, null, "tenant_only_sales")).isTrue();

        verify(grantMapper, times(2)).selectList(any());
    }

    @Test
    void updateGrantDoesNotFallBackAcrossTenants() {
        when(grantMapper.selectOne(any())).thenReturn(null);
        DataSourceTableGrantRequest request = new DataSourceTableGrantRequest();
        request.setSchemaName("public");
        request.setTablePattern("hankel_");
        request.setPatternType("PREFIX");

        assertThatThrownBy(() -> service.updateGrant("default", DATASOURCE_ID, 1L, request))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("DataSource table grant not found");

        verify(grantMapper, never()).selectList(any());
    }

    @Test
    void listAuthorizedTablesFiltersMetadataRowsByGrant() throws Exception {
        DataSource dataSource = mock(DataSource.class);
        Connection connection = mock(Connection.class);
        DatabaseMetaData metaData = mock(DatabaseMetaData.class);
        ResultSet rs = mock(ResultSet.class);
        when(dsManager.getNamedJdbcTemplate(DATASOURCE_ID)).thenReturn(new NamedParameterJdbcTemplate(dataSource));
        when(dataSource.getConnection()).thenReturn(connection);
        when(connection.getMetaData()).thenReturn(metaData);
        when(connection.getCatalog()).thenReturn("postgres");
        when(metaData.getSearchStringEscape()).thenReturn("\\");
        when(metaData.getTables(any(), any(), any(), any())).thenReturn(rs);
        when(rs.next()).thenReturn(true, true, false);
        when(rs.getString("TABLE_SCHEM")).thenReturn("public", "public");
        when(rs.getString("TABLE_NAME")).thenReturn("hankel_sales", "retailcdp_customers");
        when(rs.getString("TABLE_TYPE")).thenReturn("TABLE", "TABLE");
        when(rs.getString("REMARKS")).thenReturn((String) null, (String) null);

        List<DataSourceTableVO> tables = service.listAuthorizedTables("hankel", DATASOURCE_ID);

        assertThat(tables).extracting(DataSourceTableVO::getTableName)
                .containsExactly("hankel_sales");
    }

    @Test
    void listPhysicalTablesDoesNotFilterByGrant() throws Exception {
        DataSource dataSource = mock(DataSource.class);
        Connection connection = mock(Connection.class);
        DatabaseMetaData metaData = mock(DatabaseMetaData.class);
        ResultSet rs = mock(ResultSet.class);
        when(dsManager.getNamedJdbcTemplate(DATASOURCE_ID)).thenReturn(new NamedParameterJdbcTemplate(dataSource));
        when(dataSource.getConnection()).thenReturn(connection);
        when(connection.getMetaData()).thenReturn(metaData);
        when(connection.getCatalog()).thenReturn("postgres");
        when(metaData.getSearchStringEscape()).thenReturn("\\");
        when(metaData.getTables(any(), any(), any(), any())).thenReturn(rs);
        when(rs.next()).thenReturn(true, true, false);
        when(rs.getString("TABLE_SCHEM")).thenReturn("public", "public");
        when(rs.getString("TABLE_NAME")).thenReturn("hankel_sales", "t_datasource_config");
        when(rs.getString("TABLE_TYPE")).thenReturn("TABLE", "TABLE");
        when(rs.getString("REMARKS")).thenReturn((String) null, (String) null);

        List<DataSourceTableVO> tables = service.listPhysicalTables(DATASOURCE_ID, "public");

        assertThat(tables).extracting(DataSourceTableVO::getTableName)
                .containsExactly("hankel_sales", "t_datasource_config");
    }

    @Test
    void listAuthorizedColumnsRejectsTableOutsideGrant() {
        assertThatThrownBy(() ->
                service.listAuthorizedColumns("hankel", DATASOURCE_ID, "public", "t_datasource_config"))
                .isInstanceOf(ForbiddenException.class)
                .hasMessageContaining("not authorized");
    }

    @Test
    void probeSqlUsesJdbcMaxRowsWithoutAppendingDialectLimit() {
        NamedParameterJdbcTemplate named = mock(NamedParameterJdbcTemplate.class);
        JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
        when(dsManager.getNamedJdbcTemplate(DATASOURCE_ID)).thenReturn(named);
        when(named.getJdbcTemplate()).thenReturn(jdbcTemplate);
        when(jdbcTemplate.getMaxRows()).thenReturn(0);
        when(named.query(eq("SELECT * FROM hankel_sales"), any(MapSqlParameterSource.class), any(ResultSetExtractor.class)))
                .thenReturn(List.of());
        SqlProbeRequest request = new SqlProbeRequest();
        request.setSql("SELECT * FROM hankel_sales");
        request.setMaxRows(25);

        MetricsQueryData result = service.probeSql("hankel", DATASOURCE_ID, request);

        verify(jdbcTemplate).setMaxRows(25);
        verify(jdbcTemplate).setMaxRows(0);
        verify(named).query(eq("SELECT * FROM hankel_sales"), any(MapSqlParameterSource.class), any(ResultSetExtractor.class));
        assertThat(result.getRowCount()).isZero();
    }

    @Test
    void queryDatasourceAllowsAnyReadOnlyTableWithoutGrantFiltering() {
        when(grantMapper.selectList(any())).thenReturn(List.of());
        NamedParameterJdbcTemplate named = mock(NamedParameterJdbcTemplate.class);
        JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
        when(dsManager.getNamedJdbcTemplate(DATASOURCE_ID)).thenReturn(named);
        when(named.getJdbcTemplate()).thenReturn(jdbcTemplate);
        when(jdbcTemplate.getMaxRows()).thenReturn(0);
        when(named.query(eq("SELECT * FROM t_datasource_config"), any(MapSqlParameterSource.class), any(ResultSetExtractor.class)))
                .thenReturn(List.of());
        SqlProbeRequest request = new SqlProbeRequest();
        request.setSql("SELECT * FROM t_datasource_config");
        request.setMaxRows(10);

        MetricsQueryData result = service.queryDatasource(DATASOURCE_ID, request);

        assertThat(result.getSemanticModel()).isEqualTo("datasource_query");
        assertThat(result.getRowCount()).isZero();
        verify(jdbcTemplate).setMaxRows(10);
        verify(jdbcTemplate).setMaxRows(0);
    }

    private DataSourceConfig datasource() {
        DataSourceConfig datasource = new DataSourceConfig();
        datasource.setId(DATASOURCE_ID);
        datasource.setSchemaName("public");
        datasource.setSourceType("cdp_postgres");
        datasource.setDeleted(0);
        return datasource;
    }

    private DataSourceTableGrant prefixGrant(String prefix) {
        DataSourceTableGrant grant = new DataSourceTableGrant();
        grant.setId(1L);
        grant.setTenantId("hankel");
        grant.setDatasourceId(DATASOURCE_ID);
        grant.setSchemaName("public");
        grant.setTablePattern(prefix);
        grant.setPatternType("PREFIX");
        grant.setCaseSensitive(false);
        grant.setStatus(1);
        grant.setDeleted(0);
        return grant;
    }
}
