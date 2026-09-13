# Agentic CDP Draft

This directory captures the current draft for an agentic customer data platform demo. It turns the three source folders from `/Users/cid/Documents` into a versioned repo artifact:

- `spec/`: Bitable-style specification CSVs for agents, tools, topology, skills, scenarios, metrics, schema, and quality checks.
- `delta/`: incremental harness rows that add task/utility agents.
- `../../../raw_data/agentic_cdp/`: synthetic retail CDP data registered with the existing dataset API.

Use-case exports:

- [Dormant Reactivation Use Case](./dormant-reactivation-use-case.html)

## Current Shape

The draft is split into two cooperating agent platforms:

| Platform | Role | Boundary |
| --- | --- | --- |
| Data Agent Platform | Evidence-backed analysis, SQL/RFM/propensity utilities, metrics queries, widgets, semantic-layer style access | Produces insight and structured artifacts; does not send campaigns or create CRM tasks directly. |
| Process Agent Platform | Scheduling, approvals, content rendering, campaign/task execution, audit log, connector-facing actions | Turns verified insight into controlled action; must pass consent, budget, frequency, and approval gates. |

The current Bitable spec defines:

| Area | Count | Source |
| --- | ---: | --- |
| Agent harness rows | 33 | `spec/Agent_Harness.csv` |
| Task/utility delta rows | 29 | `delta/Agent_Harness_Task_Utility.csv` |
| Topology edges | 62 | `spec/Topology.csv` |
| Skills | 12 | `spec/Skills.csv` |
| Tools | 15 | `spec/Tools.csv` |
| Metrics | 10 | `spec/Metrics.csv` |
| Business scenarios | 4 | `spec/Scenarios.csv` |
| Data tables | 13 | `spec/Data_Dictionary.csv` |

## Registered Data Assets

The business CSVs are copied into `raw_data/agentic_cdp/` and registered in `prediction_app/config/datasets.json` with `agentic_cdp_*` dataset IDs.

| Dataset ID | CSV | Primary use |
| --- | --- | --- |
| `agentic_cdp_customers` | `customers.csv` | Customer 360 profile, lifecycle, consent snapshot, RFM fields, churn/replenishment/VIP scores |
| `agentic_cdp_consents` | `consents.csv` | Channel consent and policy checks |
| `agentic_cdp_loyalty_accounts` | `loyalty_accounts.csv` | Tier, points, anniversary, and next-tier-gap workflows |
| `agentic_cdp_transactions` | `transactions.csv` | RFM, reactivation, attribution, and purchase history |
| `agentic_cdp_transaction_items` | `transaction_items.csv` | Basket, category, product, and replenishment analysis |
| `agentic_cdp_behavior_events` | `behavior_events.csv` | Intent signals, journey triggers, and audience discovery |
| `agentic_cdp_products` | `products.csv` | Product catalog, replenishment cycles, hero SKU flags |
| `agentic_cdp_inventory` | `inventory.csv` | Stock guardrails for next-best-action recommendations |
| `agentic_cdp_stores` | `stores.csv` | Store, region, city-tier, and manager context |
| `agentic_cdp_campaigns` | `campaigns.csv` | Scenario-linked campaign definitions |
| `agentic_cdp_campaign_interactions` | `campaign_interactions.csv` | Campaign response and revenue attribution |
| `agentic_cdp_service_tickets` | `service_tickets.csv` | Service recovery, sentiment, SLA, and churn-risk signals |
| `agentic_cdp_agent_runs` | `agent_runs.csv` | Agent run telemetry, quality, latency, cost, and approvals |

`agentic_cdp_transactions` includes an RFM config using:

- user: `customer_id`
- order: `transaction_id`
- date: `transaction_date`
- monetary amount: `total_amount`

That makes the existing RFM workflow usable before building a dedicated CDP orchestrator.

## Scenario Model

The four first-pass scenarios are:

| Scenario | Business goal | Primary orchestrator | Core data |
| --- | --- | --- | --- |
| Dormant member reactivation | Find reachable members inactive for 120+ days, generate segment, offer, content, approval packet, and measurement report | `ORCH_DORMANT_REACTIVATION` | `customers`, `transactions`, `behavior_events`, `consents`, `campaign_interactions`, `products` |
| Replenishment and NBA | Recommend the next product, benefit, channel, and timing using replenishment cycles, behavior, and stock | `ORCH_REPLENISHMENT_NBA` | `customers`, `transactions`, `transaction_items`, `behavior_events`, `products`, `inventory`, `consents` |
| VIP clienteling | Generate customer cards, product bundles, appointment suggestions, and sales-associate follow-up tasks | `ORCH_VIP_CLIENTELING` | `customers`, `loyalty_accounts`, `transactions`, `products`, `stores`, `inventory`, `behavior_events` |
| Service recovery retention | Prioritize complaints and recovery actions based on sentiment, SLA, value, churn risk, and consent | `ORCH_SERVICE_RECOVERY` | `customers`, `service_tickets`, `transactions`, `consents`, `campaign_interactions` |

## Implementation Path

1. Data asset layer: keep CSVs registered as first-class datasets and verify preview, stats, and RFM behavior.
2. Semantic/query layer: convert `spec/SQL_Schema.csv` and `spec/Metrics.csv` into a CDP schema prompt or lightweight semantic manifest.
3. Agent layer: add an `agentic_cdp_agent` that reuses the existing Data Agent SQL/RFM patterns but swaps in CDP-specific metrics, scenarios, tools, and guardrails.
4. Process layer: start with dry-run artifacts only: segment spec, NBA list, campaign brief, approval packet, clienteling task, and measurement report.
5. Governance layer: wire consent/frequency/budget checks before any simulated execution artifact is marked executable.

## Verification Checklist

- `prediction_app/config/datasets.json` parses as JSON.
- `/api/v1/datasets` should list the `agentic_cdp_*` datasets when the Python API is running.
- `/api/v1/datasets/agentic_cdp_transactions/preview` should show transaction rows.
- `/api/v1/datasets/agentic_cdp_transactions/rfm` should run with the default RFM config.
