---
name: bjy-metrics-definition
description: 白居易公寓指标口径与数据集市定义。包含所有业务指标的计算公径、数据来源表、维度和业务约定，供指标智能体加载并解读用户查询意图。
metadata:
  category: metrics
---

---

## I. Conventions

### 1.1 Time Granularity

- **Day**: `date` (calendar day)
- **Month / Period**: closed interval `[period_start, period_end]` (inclusive), e.g. 2024-01-01..2024-01-31
- SQL: `date between period_start and period_end`

### 1.2 Room Scope (Denominator)

- All denominators (occupancy rate, total rooms, available room-nights, etc.) use the **same room scope**
- Room master data: `bjy_apartment` (or equivalent dim), providing shop / door number / room type / area / pricing
- **Out-of-scope rooms** (e.g. Novotel-occupied rooms, Yishan V1008, etc.) must be excluded from denominators
- **v1 scope**: use full `bjy_apartment` as "rentable"; shop scope unified via `bjy_apartment.shop`

### 1.3 Occupancy Rules

- **Occupancy segment fact table**: `bjy_dw_room_occupancy`, granularity = one room occupancy segment
- **Occupied day**: **checkout date counts as occupied**, i.e. `date between actual_start_date and actual_end_date` (inclusive)
- **Room-night fact table**: `bjy_mart_room_night`, granularity = `(date, apartment_id)`, expanded from occupancy segments to daily granularity; preferred for occupancy rate / room-night / vacancy distribution calculations

### 1.4 Rent & Pricing (No Finance Tables)

- **Use tenant table only**, do NOT use finance tables
- Monthly rent: `rent` (CNY/month)
- Daily rent: `rent_day = rent / 30.0`
- Period total rent: `total_rent = Σ(rent_day * is_occupied)` (aggregated by room-day)
- List price: `list_price = coalesce(bjy_apartment.adjusted_price, bjy_apartment.standard_price)`
- **ADR**: `total_rent / occupied_room_nights`
- **RevPAR**: `total_rent / total_rooms` (v1: `total_rooms` uses **period-end room count**, i.e. room inventory on period_end)

### 1.5 Other Conventions

- **Low-rent filter**: `rent < 500` excluded at ODS→STG stage (dbt var: `min_valid_rent_amount=500`)
- **Checkout scope (v1)**: only `正常退房 / 违约退房` count as churn; room change / sublease / contract amendment do NOT count
- **Lease term buckets**: short ≤90 days, medium 91~180 days, long 181~365 days, extra-long >365 days
- **QFT (换签)**: includes room change + sublease (used for switch rate, etc.)

---

## II. Metric Definitions

### 2.1 Occupancy & Room-Nights

| API Name                     | Chinese Name   | Definition                          | Formula                                                              | Primary Source                                  |
| ---------------------------- | -------------- | ----------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------- |
| `occupied_room_nights`       | 间夜量         | Total occupied room-days            | `Σ is_occupied`                                                      | `bjy_mart_room_night`                           |
| `available_room_nights`      | 可出租间夜     | Total available room-days           | `Σ is_available`                                                     | `bjy_mart_room_night` + `bjy_apartment`         |
| `occupancy_rate`             | 实际出租率     | Room time utilization               | `occupied_room_nights / available_room_nights`                       | Same as above                                   |
| `occupancy_achievement_rate` | 间夜达成率     | Actual vs target occupancy          | `occupancy_rate / target_occupancy_rate`                             | `fact_occupancy_target_month` + above           |
| `full_occupancy_rate`        | 满租率         | Occupied room ratio on a given date | `occupied_rooms_on_date / total_rooms_on_date`                       | `bjy_mart_room_night` + `bjy_apartment`         |
| `adr`                        | ADR (日均房价) | Revenue per room-night              | `total_rent / occupied_room_nights` (total_rent = rent/30 converted) | `bjy_mart_room_night` + `bjy_dw_room_occupancy` |
| `revpar`                     | RevPAR         | Revenue per available room          | `total_rent / total_rooms` (v1: period-end room count)               | Same as above                                   |
| `vacancy_bucket`             | 空置分布       | Vacant rooms by vacancy duration    | Bucketed by `vacancy_streak_days`: <7 / 7-15 / 15-25 / >25           | `bjy_mart_room_night`                           |

### 2.2 Pricing & Discount

| API Name                  | Chinese Name       | Definition                      | Formula                                                    | Primary Source                            |
| ------------------------- | ------------------ | ------------------------------- | ---------------------------------------------------------- | ----------------------------------------- |
| `discount_rate_contract`  | 单房折扣率         | Single-contract discount level  | `rent / list_price`                                        | `bjy_dw_room_occupancy` + `bjy_apartment` |
| `discount_rate_overall`   | 综合折扣率         | Overall price discount          | `Σrent / Σlist_price` (contract-weighted)                  | Same as above                             |
| `discount_amount`         | 折扣额             | Amount lost to discounts        | `Σ(list_price - rent)`                                     | Same as above                             |
| `avg_rent_active`         | 在租均价           | Current average deal price      | `Σrent_active / active_contracts`                          | `bjy_dw_room_occupancy`                   |
| `avg_rent_new_sign`       | 新签均价           | New-sign average rent in period | `Σrent_new_sign / new_sign_contracts`                      | Same (filter: start_origin_std=NEW_SIGN)  |
| `avg_rent_renewal`        | 续签均价           | Renewal average rent in period  | `Σrent_renewal / renewal_contracts`                        | Same (filter: RENEWAL)                    |
| `rent_variance_by_layout` | 价格方差（按户型） | Rent dispersion by room type    | `VAR(rent)` grouped by room type                           | `bjy_apartment` + occupancy               |
| `rent_per_sqm`            | 坪效               | Revenue per area                | `rent / actual_area`                                       | Same as above                             |
| `price_break_rate`        | 破价率             | Below-floor-price lease ratio   | `count(rent < floor_price) / active_contracts`             | `dim_pricing` + occupancy                 |
| `avg_price_break_depth`   | 平均破价幅度       | Average depth of price break    | `avg(1 - rent / floor_price)` (only price-break contracts) | Same as above                             |

### 2.3 Contracts & Turnover Efficiency

| API Name                                    | Chinese Name      | Definition                          | Formula                                                                                                           | Primary Source                  |
| ------------------------------------------- | ----------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `occupied_rooms`                            | 在租房源数        | Occupied rooms on a given date      | `count(distinct apartment_id) where is_occupied=1`                                                                | `bjy_mart_room_night`           |
| `new_sign_contracts`                        | 新签房源数/段数   | New-sign segments in period         | `count(*) where start_origin_std=NEW_SIGN and actual_start_date in period`                                        | `bjy_dw_room_occupancy`         |
| `renewal_contracts`                         | 续签段数          | Renewal segments in period          | `start_origin_std=RENEWAL`                                                                                        | Same as above                   |
| `switch_contracts`                          | 换租段数          | Room change + sublease              | `ROOM_CHANGE + SUBLEASE`                                                                                          | Same as above                   |
| `lease_term_days` / `avg_lease_term_days`   | 租期时长/平均租期 | Contract duration in days           | `contract_end_date - contract_start_date`                                                                         | `bjy_dw_room_occupancy`         |
| `remaining_days`                            | 剩余租期          | Days until expiry                   | `contract_end_date - as_of_date` (active contracts)                                                               | Same as above                   |
| `renewal_rate`                              | 续租率            | Tenant retention rate               | `renewals / expiring_contracts` (tenant continuity: next segment RENEWAL/ROOM_CHANGE and next_start ≤ prev_end+1) | `bjy_dw_room_occupancy` + rules |
| `room_change_rate`                          | 换租率            | Room change / sublease ratio        | `(room_change + sublease) / churn_count`                                                                          | Same as above                   |
| `vacancy_gap_days` / `avg_vacancy_gap_days` | 空置期/平均空置期 | Gap between checkout and next lease | `next_actual_start - prev_actual_end - 1` (per room, adjacent segments)                                           | `bjy_dw_room_occupancy`         |
| `contracts_per_day`                         | 日均签约量        | Signing pace                        | `period_contracts / period_days`                                                                                  | contract DW / occupancy         |

### 2.4 Retention & Risk (Churn)

| API Name                 | Chinese Name | Definition                        | Formula                                                                   | Primary Source                     |
| ------------------------ | ------------ | --------------------------------- | ------------------------------------------------------------------------- | ---------------------------------- |
| `churn_contracts`        | 退租段数     | Normal + breach checkout segments | `end_nature_std in ('正常退房','违约退房')` and actual_end_date in period | `bjy_dw_room_occupancy`            |
| `normal_churn_contracts` | 正常退房段数 | Normal checkout                   | `end_nature_std='正常退房'`                                               | Same as above                      |
| `breach_churn_contracts` | 违约退房段数 | Breach checkout                   | `end_nature_std='违约退房'`                                               | Same as above                      |
| `churn_rate`             | 退租率       | Room churn level                  | `churn_contracts / occupied_rooms_start` (monthly)                        | Same as above                      |
| `breach_churn_rate`      | 违约退租率   | Abnormal termination ratio        | `breach_churn_contracts / churn_contracts`                                | Same (QFT breakdown more reliable) |
| `net_new_contracts`      | 净增单量     | Net change in leased rooms        | `new_sign + renewal - churn` (in period)                                  | occupancy / contract               |

### 2.5 Structure & Channel

| API Name             | Chinese Name | Definition                     | Formula                                                  | Primary Source                            |
| -------------------- | ------------ | ------------------------------ | -------------------------------------------------------- | ----------------------------------------- |
| `channel_share`      | 渠道占比     | Customer acquisition structure | `channel_contracts / total_contracts` (by tenant_source) | `bjy_dw_room_occupancy.tenant_source_std` |
| `layout_share`       | 户型占比     | Occupied room structure        | `layout_occupied / total_occupied`                       | `bjy_mart_room_night` (by room_type)      |
| `avg_rent_by_layout` | 户型均价     | Room type premium              | `Σrent / occupied_count` grouped by room type            | `bjy_apartment` + occupancy               |
| `long_lease_share`   | 长租占比     | Lease structure (long)         | `long_lease_contracts / total_active` (term ≥ 181)       | Lease term threshold                      |
| `short_lease_share`  | 短租占比     | Lease structure (short)        | `short_lease_contracts / total_active` (term ≤ 90)       | Same as above                             |

---

## III. Data Mart Dictionary

### 3.1 Table Inventory (v1)

**Required Tables**

| Table                                           | Layer | Purpose                                                                           |
| ----------------------------------------------- | ----- | --------------------------------------------------------------------------------- |
| `bjy_dw_room_occupancy`                         | DW    | Room occupancy segment fact (interval granularity), unified KE/QFT                |
| `bjy_mart_room_night`                           | MART  | Room calendar fact (day x room), room-nights / occupancy / ADR / RevPAR / vacancy |
| `bjy_dm_shop_day_room_metrics`                  | DM    | Shop x room_type x day KPI aggregation (thin metric layer)                        |
| `bjy_dm_shop_day_sales_source_contract_metrics` | DM    | Shop x sales x source x room_type x day lease-start metrics                       |
| `bjy_dm_shop_day_sales_source_churn_metrics`    | DM    | Shop x sales x source x room_type x day churn metrics                             |
| `bjy_apartment`                                 | Dim   | Room master: shop / door_number / room_type / area / pricing                      |

**Optional (v1 not hard-dependent)**

| Table                                                                | Purpose                                                    |
| -------------------------------------------------------------------- | ---------------------------------------------------------- |
| `fact_occupancy_target_month`                                        | Monthly x shop target occupancy rate, for achievement rate |
| `bjy_dw_room_occupancy_audit` / `bjy_dw_room_occupancy_null_profile` | Quality audit                                              |

### 3.2 Table Schema Details

#### 3.2.1 `bjy_dw_room_occupancy` (Interval Fact)

- **Purpose**: Unified KE/QFT room occupancy segments; lease term, vacancy gap, churn/renewal/switch contract chain metrics; source for `bjy_mart_room_night` daily occupancy.
- **Granularity**: one row = one occupancy segment `(source_system, occupancy_key)`.

| Column                                      | Description                                                    |
| ------------------------------------------- | -------------------------------------------------------------- |
| `source_system`                             | KE / QFT                                                       |
| `occupancy_key`                             | Unique occupancy segment key                                   |
| `snapshot_date`                             | Calibration snapshot date                                      |
| `apartment_id`                              | Maps to bjy_apartment.id                                       |
| `shop`                                      | Shop (unified via bjy_apartment.shop)                          |
| `door_number`, `room_type`, `sub_room_type` | Door number, room type                                         |
| `tenant_name`, `tenant_phone`               | Tenant info                                                    |
| `contract_start_date` / `contract_end_date` | Contract start/end                                             |
| `actual_start_date` / `actual_end_date`     | Actual start/end (**checkout date inclusive**)                 |
| `start_origin_std`                          | NEW_SIGN / RENEWAL / ROOM_CHANGE / SUBLEASE / ...              |
| `end_origin_std`                            | CHECKOUT / ROOM_CHANGE / SUBLEASE / CONTRACT_END / ...         |
| `checkout_nature_std` / `end_nature_std`    | Checkout nature: 正常退房 / 违约退房 / 换房 / 转租 / 改签 etc. |
| `rent`                                      | Monthly rent (filtered: >= 500)                                |
| `tenant_source_std`, `salesperson`          | Channel, salesperson                                           |

#### 3.2.2 `bjy_mart_room_night` (Daily Fact: Room Status)

- **Purpose**: Daily room occupancy/vacancy; room-nights, available room-nights, occupancy rate, full occupancy rate, ADR, RevPAR, vacancy distribution.
- **Granularity**: one row = `date x apartment_id`.

| Column                                              | Description                                                   |
| --------------------------------------------------- | ------------------------------------------------------------- |
| `snapshot_date`, `load_time`                        | Snapshot date, load time                                      |
| `date`                                              | Calendar day                                                  |
| `apartment_id`, `apartment_unique_id`               | Room ID                                                       |
| `shop`, `door_number`, `room_type`, `sub_room_type` | Denormalized dimensions                                       |
| `is_available`                                      | v1: always true (rentable)                                    |
| `is_occupied`                                       | Whether occupied on this day (checkout date inclusive)        |
| `occupied_segments`                                 | Number of occupancy segments covering this day (normally 0/1) |
| `rent_day_amount`                                   | Daily rent = rent/30 (on occupied days)                       |
| `list_price_day_amount`                             | List price / 30 (from bjy_apartment)                          |
| `vacancy_streak_days`                               | Consecutive vacant days as of this date                       |

**Generation**: rooms from `bjy_apartment` (excluding out-of-scope), calendar via `generate_series(period_start, period_end)`, occupancy expanded from `actual_start_date..actual_end_date`.

#### 3.2.3 `bjy_dm_shop_day_room_metrics` (DM: Shop x RoomType x Day KPI)

- **Purpose**: Thin metric layer dataset, pre-computed common KPIs, avoids full-scan of `bjy_mart_room_night`.
- **Granularity**: `snapshot_date x date x shop x room_type x sub_room_type`.

| Column                               | Description                                                                             |
| ------------------------------------ | --------------------------------------------------------------------------------------- |
| `snapshot_date`, `date`              | Snapshot date, calendar day                                                             |
| `shop`, `room_type`, `sub_room_type` | Dimensions                                                                              |
| `available_room_nights`              | Available room-nights (summable)                                                        |
| `occupied_room_nights`               | Occupied room-nights (summable)                                                         |
| `total_rent_amount`                  | Contract rent total (rent/30 accumulated daily)                                         |
| `total_list_price_amount`            | List price total (list_price/30 accumulated daily)                                      |
| `occupancy_rate_day`                 | Daily occupancy rate (for validation only; period rate = occupied/available aggregated) |
| `adr_day`                            | Daily ADR (same caveat)                                                                 |

#### 3.2.4 `bjy_dm_shop_day_sales_source_contract_metrics` (DM: Lease-Start Metrics)

- **Purpose**: Sales/channel analysis, daily granularity by lease start; thin metric layer dataset.
- **Granularity**: `snapshot_date x date x shop x room_type x sub_room_type x salesperson x tenant_source`.
- **Time**: `date` = `actual_start_date` (lease start date).

| Column               | Description                                   |
| -------------------- | --------------------------------------------- |
| `new_sign_contracts` | New-sign segments (start_origin_std=NEW_SIGN) |
| `renewal_contracts`  | Renewal segments (RENEWAL)                    |
| `switch_contracts`   | Switch segments (ROOM_CHANGE + SUBLEASE)      |

#### 3.2.5 `bjy_dm_shop_day_sales_source_churn_metrics` (DM: Churn Metrics)

- **Purpose**: Churn/breach/attrition analysis, daily granularity by churn date; thin metric layer dataset.
- **Granularity**: same as above, `date` = `actual_end_date` (churn date).
- **Churn scope**: `end_nature_std in ('正常退房','违约退房')`.

| Column                   | Description              |
| ------------------------ | ------------------------ |
| `churn_contracts`        | Churn segments           |
| `normal_churn_contracts` | Normal checkout segments |
| `breach_churn_contracts` | Breach checkout segments |

#### 3.2.6 `bjy_apartment` (Room Dimension)

- **Purpose**: Room count denominator, shop/room type/area dimensions, pricing denominator.
- **Key columns**: `id`/`unique_id`, `shop`/`door_number`, `room_type`/`sub_room_type`, `actual_area`, `standard_price`/`adjusted_price`.

#### 3.2.7 `fact_occupancy_target_month` (Monthly Target)

- **Purpose**: Occupancy achievement rate = actual_occupancy_rate / target_occupancy_rate.
- **Granularity**: `month_key x shop`.
- **Columns**: `month_key`, `shop`, `target_occupancy_rate` (0~1), `target_occupancy_rate_pct`, etc.

---

## IV. BI Query & Semantic Layer

- **Metric query API**: `POST /bi_api/api/metrics/query` (see `docs/metrics_query_examples.md`).
- **Semantic model & metric metadata**: `apps/bi/src/main/resources/metrics/metricflow_meta.json` (semantic_models + metrics).
- **Time granularity**: `groupBy` supports `date`, `date__month`, `month_key`, `end_date__month`, etc.; when `snapshot_date` is omitted, the latest snapshot is used by default.

---

## V. Table Selection Guide

When choosing which table to query, follow this priority:

1. **DM tables** (`bjy_dm_shop_day_room_metrics`, `bjy_dm_shop_day_sales_source_*_metrics`): Use for aggregated KPIs by shop/room_type/day. Fastest, pre-computed.
2. **MART table** (`bjy_mart_room_night`): Use for daily room-level granularity, vacancy streaks, custom aggregations.
3. **DW table** (`bjy_dw_room_occupancy`): Use for contract-level analysis, lease terms, churn reasons, tenant info, rent details.
4. **Dim table** (`bjy_apartment`): Use for room master data, pricing, area, room type.

**Cross-table joins**:

- `bjy_mart_room_night.apartment_id = bjy_apartment.id` (room details)
- `bjy_dw_room_occupancy.apartment_id = bjy_apartment.id` (room details for contracts)
- DM tables already denormalize shop/room_type, usually no join needed

---
