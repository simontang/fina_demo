# Hankel Distributor Performance Semantic Assets

## Scope

These assets provide governed Metrics Runtime access to Hankel Sell-in,
Sell-out, and Inventory data on datasource `15`.

Builder/Admin exploration can still inspect granted raw tables through:

```http
POST /api/v1/datasources/15/query
X-Tenant-Id: hankel
```

Metrics Runtime exposes only the published semantic views below. The raw
tables are deliberately not published through `meta/tables`.

## Semantic Views

| View | Grain | Runtime use |
|---|---|---|
| `hankel_view_distr_sell_in` | One imported Sell-in allocation row | Flow metrics with a real `posting_date` |
| `hankel_view_distr_sell_out` | One customer-product-month-sales-team allocation row | Quality-filtered Sell-out metrics |
| `hankel_view_distr_inventory_monthly` | One customer-product-month-sales-team snapshot row | Historical stock investigation |
| `hankel_view_distr_inventory_current` | One customer-product-sales-team row at the latest snapshot | Current inventory metrics |

Sell-in and Sell-out are flow datasets. When a query has no date filter, the
result covers all loaded periods and the metric Meta states that explicitly.

Inventory is a stock dataset. Existing `hankel_inventory_*` metrics query only
the latest available snapshot and aggregate `territory_inventory_*` allocation
values. They do not sum repeated raw inventory values or multiple months.

## Published Metrics

Sell-in:

- `hankel_sell_in_nes`
- `hankel_sell_in_quantity`
- `hankel_sell_in_gross_margin`
- `hankel_sell_in_gross_margin_rate`
- `hankel_sell_in_contribution`

Sell-out:

- `hankel_sell_out_value`
- `hankel_sell_out_quantity`
- `hankel_sell_out_excluded_value`
- `hankel_sell_out_quality_issue_count`

Inventory:

- `hankel_inventory_value`
- `hankel_inventory_quantity`

All metric details use `supported_dimensions`, a schema-qualified source view,
SQL-free calculation metadata, and an explicit `default_time_context`.

## Apply

Create or update the views:

```bash
psql -v ON_ERROR_STOP=1 \
  -f metrics-server/scripts/hankel/distributor-sell-out-view.sql \
  -f metrics-server/scripts/hankel/distributor-sell-in-inventory-views.sql
```

Publish Meta:

```bash
bash metrics-server/scripts/hankel/publish-distributor-sell-out-meta.sh
bash metrics-server/scripts/hankel/publish-distributor-sell-in-inventory-meta.sh
```

The second publisher removes Runtime Meta for these raw tables after all
replacement views and metrics have been published:

- `hankel_distr_sell_in`
- `hankel_distr_sell_out`
- `hankel_distr_inventory`

It does not drop source tables or remove Builder/Admin datasource access.

## Runtime Rules

- Semantic `groupBy` and filters must use a declared `supported_dimensions`
  key.
- `orderBy.field` must be one of the selected dimensions or metrics.
- Runtime `customSql` requires active table grants and every referenced table
  must also be present in published table Meta.
- Builder/Admin `/datasources/{dsId}/query` remains the unrestricted read-only
  exploration API inside the datasource boundary.
- Non-additive score metrics may declare `query_constraints.required_group_by`;
  the Runtime rejects queries that omit those dimensions.
