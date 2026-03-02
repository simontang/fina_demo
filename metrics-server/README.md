# metrics-server

Spring Boot service that exposes a unified metrics API backed by one or more SAP HANA databases.
Designed to be called by AI agents and other services in the Fina Demo stack.

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Spring Boot 3.2 (Java 17) |
| ORM | MyBatis Plus 3.5 |
| HANA driver | SAP ngdbc 2.19 |
| Connection pool | HikariCP |
| Port | **5704** |

## Architecture

```
┌─────────────┐     REST      ┌──────────────────────────┐
│  AI Agent   │──────────────▶│  metrics-server :5704    │
└─────────────┘               │                          │
                              │  ┌────────────────────┐  │
                              │  │ Master HANA (meta) │  │  ← T_DATASOURCE_CONFIG
                              │  │  MyBatis Plus      │  │  ← T_METRICS_META
                              │  └────────────────────┘  │
                              │                          │
                              │  ┌────────────────────┐  │
                              │  │ Dynamic DS Pool    │  │  ← HANA-1
                              │  │ HikariCP per DS    │  │  ← HANA-2 …
                              │  └────────────────────┘  │
                              └──────────────────────────┘
```

- **Master HANA** (`MASTER_HANA_*` env) stores datasource configs and metric definitions.
- On startup, all active datasource configs are loaded and a HikariCP pool is created for each.
- When a datasource is added/updated via API, the pool is registered/replaced at runtime without restart.

## Quick Start

### 1. Prerequisites

- Java 17+
- Maven 3.9+
- Access to a SAP HANA instance for the master schema

### 2. Create master schema tables

Connect to your HANA instance and run:
```sql
-- Run as the user specified in MASTER_HANA_USER
-- (switch to your schema first if needed)
-- See src/main/resources/sql/init.sql
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your HANA connection details
```

### 4. Run locally

```bash
# Export env vars (or use .env tooling)
export MASTER_HANA_URL="jdbc:sap://your-hana:39015?currentSchema=METRICS"
export MASTER_HANA_USER="SYSTEM"
export MASTER_HANA_PASSWORD="yourpassword"

./gradlew bootRun
```

### 5. Docker Compose

```bash
# From fina_demo root:
docker compose up metrics_server
```

## API Reference

### Datasource Management

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/datasources` | List all datasources |
| GET | `/api/v1/datasources/active` | List active datasources only |
| GET | `/api/v1/datasources/{id}` | Get datasource by ID |
| POST | `/api/v1/datasources` | Create datasource |
| PUT | `/api/v1/datasources/{id}` | Update datasource |
| DELETE | `/api/v1/datasources/{id}` | Soft-delete datasource |
| POST | `/api/v1/datasources/test` | Test connection (no persist) |
| POST | `/api/v1/datasources/{id}/reload` | Reload pool from DB config |

### Metric Definitions

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/datasources/{dsId}/metrics` | List metrics for a datasource |
| GET | `/api/v1/datasources/{dsId}/metrics/{code}` | Get metric definition |
| POST | `/api/v1/datasources/{dsId}/metrics` | Create metric definition |
| PUT | `/api/v1/metrics/{id}` | Update metric definition |
| DELETE | `/api/v1/metrics/{id}` | Delete metric definition |

### Query Execution

```
POST /api/v1/metrics/query
```

**Mode 1 — by registered metric code:**
```json
{
  "datasourceId": 1,
  "metricCode": "SALES_BY_DATE",
  "params": {
    "startDate": "2024-01-01",
    "endDate": "2024-12-31"
  },
  "limit": 500
}
```

**Mode 2 — ad-hoc SQL (for agent tool calls):**
```json
{
  "datasourceId": 1,
  "customSql": "SELECT TOP 10 * FROM SALES WHERE REGION = :region",
  "params": { "region": "APAC" }
}
```

**Response:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "datasourceId": 1,
    "metricCode": "SALES_BY_DATE",
    "columns": ["SALE_DATE", "TOTAL_AMOUNT"],
    "rows": [
      { "SALE_DATE": "2024-01-01", "TOTAL_AMOUNT": 125000.00 }
    ],
    "rowCount": 1,
    "executionTimeMs": 42,
    "executedSql": "SELECT ... LIMIT 500"
  }
}
```

## Metric Parameter Schema

The `parameters` field in a metric definition is a JSON array:

```json
[
  {
    "name": "startDate",
    "type": "STRING",
    "required": true,
    "description": "Start date in yyyy-MM-dd format"
  },
  {
    "name": "region",
    "type": "STRING",
    "required": false,
    "description": "Filter by region code"
  }
]
```

SQL uses `:paramName` syntax (Spring NamedParameterJdbcTemplate):
```sql
SELECT * FROM SALES
WHERE SALE_DATE BETWEEN :startDate AND :endDate
  AND REGION = :region
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MASTER_HANA_URL` | Yes | — | Master HANA JDBC URL |
| `MASTER_HANA_USER` | Yes | — | Master HANA username |
| `MASTER_HANA_PASSWORD` | Yes | — | Master HANA password |
| `ENCRYPT_KEY` | No | `fina-metrics-2024!` | AES key for stored passwords |
| `SERVER_PORT` | No | `5704` | HTTP server port |
| `MASTER_POOL_MAX` | No | `10` | Max HikariCP connections (master) |
| `MASTER_POOL_MIN` | No | `2` | Min idle connections (master) |
| `SKIP_INIT` | No | `false` | Skip loading dynamic DS on startup |
| `LOG_LEVEL` | No | `INFO` | Logging level for `com.fina.metrics` |

## Security Notes

- Passwords stored in `T_DATASOURCE_CONFIG` are AES-256/CBC encrypted using `ENCRYPT_KEY`.
- Passwords are **never** returned in API responses (`DataSourceVO` omits the `password` field).
- Change `ENCRYPT_KEY` from the default in production.
