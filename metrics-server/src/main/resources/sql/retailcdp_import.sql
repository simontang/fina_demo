-- Retail CDP CSV import for metrics-server PostgreSQL datasource.
-- Run with psql from a machine that can read /Users/cid/Documents/csv/*.csv.
\set ON_ERROR_STOP on

DROP VIEW IF EXISTS retailcdp_v_customer_360;
DROP VIEW IF EXISTS retailcdp_v_transaction_wide;
DROP VIEW IF EXISTS retailcdp_v_campaign_performance;
DROP VIEW IF EXISTS retailcdp_v_inventory_health;
DROP VIEW IF EXISTS retailcdp_v_agent_run_summary;

DROP TABLE IF EXISTS retailcdp_transactions CASCADE;
DROP TABLE IF EXISTS retailcdp_transaction_items CASCADE;
DROP TABLE IF EXISTS retailcdp_stores CASCADE;
DROP TABLE IF EXISTS retailcdp_service_tickets CASCADE;
DROP TABLE IF EXISTS retailcdp_products CASCADE;
DROP TABLE IF EXISTS retailcdp_loyalty_accounts CASCADE;
DROP TABLE IF EXISTS retailcdp_inventory CASCADE;
DROP TABLE IF EXISTS retailcdp_customers CASCADE;
DROP TABLE IF EXISTS retailcdp_consents CASCADE;
DROP TABLE IF EXISTS retailcdp_campaigns CASCADE;
DROP TABLE IF EXISTS retailcdp_campaign_interactions CASCADE;
DROP TABLE IF EXISTS retailcdp_behavior_events CASCADE;
DROP TABLE IF EXISTS retailcdp_agent_runs CASCADE;

CREATE TABLE retailcdp_agent_runs (
    run_id VARCHAR(16),
    timestamp TIMESTAMP,
    agent_id VARCHAR(64),
    scenario_id VARCHAR(64),
    status VARCHAR(32),
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    tool_calls INTEGER,
    estimated_cost_usd NUMERIC(12,4),
    eval_score NUMERIC(8,4),
    human_approval_required CHAR(1),
    PRIMARY KEY (run_id)
);

CREATE TABLE retailcdp_behavior_events (
    event_id VARCHAR(16),
    customer_id VARCHAR(16),
    event_timestamp TIMESTAMP,
    event_type VARCHAR(64),
    channel VARCHAR(32),
    product_id VARCHAR(16),
    session_id VARCHAR(32),
    value NUMERIC(12,4),
    PRIMARY KEY (event_id)
);

CREATE TABLE retailcdp_campaign_interactions (
    interaction_id VARCHAR(16),
    campaign_id VARCHAR(16),
    customer_id VARCHAR(16),
    sent_date DATE,
    channel VARCHAR(32),
    event_status VARCHAR(32),
    revenue_attributed NUMERIC(12,2),
    suppression_reason VARCHAR(128),
    PRIMARY KEY (interaction_id)
);

CREATE TABLE retailcdp_campaigns (
    campaign_id VARCHAR(16),
    scenario_id VARCHAR(64),
    campaign_name VARCHAR(128),
    objective TEXT,
    channel_mix VARCHAR(128),
    start_date DATE,
    end_date DATE,
    budget NUMERIC(14,2),
    status VARCHAR(32),
    PRIMARY KEY (campaign_id)
);

CREATE TABLE retailcdp_consents (
    customer_id VARCHAR(16),
    marketing_consent CHAR(1),
    sms_consent CHAR(1),
    wechat_consent CHAR(1),
    email_consent CHAR(1),
    consent_source VARCHAR(64),
    updated_at TIMESTAMP,
    PRIMARY KEY (customer_id)
);

CREATE TABLE retailcdp_customers (
    customer_id VARCHAR(16),
    name VARCHAR(64),
    gender VARCHAR(16),
    age INTEGER,
    city VARCHAR(64),
    province VARCHAR(64),
    region VARCHAR(64),
    city_tier VARCHAR(32),
    join_date DATE,
    member_level VARCHAR(32),
    lifecycle_stage VARCHAR(64),
    preferred_channel VARCHAR(32),
    home_store_id VARCHAR(16),
    category_affinity VARCHAR(64),
    size_profile VARCHAR(16),
    price_sensitivity VARCHAR(32),
    discount_affinity VARCHAR(32),
    private_domain_status VARCHAR(64),
    acquisition_channel VARCHAR(64),
    mobile_hash VARCHAR(64),
    email_hash VARCHAR(64),
    wechat_openid_hash VARCHAR(64),
    consent_marketing CHAR(1),
    consent_sms CHAR(1),
    consent_wechat CHAR(1),
    consent_email CHAR(1),
    last_purchase_date DATE,
    days_since_purchase INTEGER,
    orders_365 INTEGER,
    total_spend_365 NUMERIC(14,2),
    avg_order_value NUMERIC(14,2),
    rfm_segment VARCHAR(64),
    churn_risk_score NUMERIC(8,4),
    replenishment_due_score NUMERIC(8,4),
    vip_score NUMERIC(8,4),
    return_rate NUMERIC(8,4),
    complaint_count INTEGER,
    created_at TIMESTAMP,
    PRIMARY KEY (customer_id)
);

CREATE TABLE retailcdp_inventory (
    inventory_id VARCHAR(16),
    store_id VARCHAR(16),
    product_id VARCHAR(16),
    stock_on_hand INTEGER,
    reserved_qty INTEGER,
    available_qty INTEGER,
    reorder_point INTEGER,
    last_stocktake_date DATE,
    PRIMARY KEY (inventory_id)
);

CREATE TABLE retailcdp_loyalty_accounts (
    customer_id VARCHAR(16),
    loyalty_id VARCHAR(16),
    tier VARCHAR(32),
    points_balance INTEGER,
    points_expiring_90d INTEGER,
    anniversary_month INTEGER,
    next_tier_gap_amount NUMERIC(14,2),
    PRIMARY KEY (customer_id)
);

CREATE TABLE retailcdp_products (
    product_id VARCHAR(16),
    sku VARCHAR(32),
    product_name VARCHAR(128),
    category VARCHAR(64),
    subcategory VARCHAR(64),
    brand_line VARCHAR(64),
    gender VARCHAR(16),
    season VARCHAR(16),
    unit_price NUMERIC(14,2),
    replenishment_cycle_days INTEGER,
    launch_date DATE,
    is_hero_sku CHAR(1),
    PRIMARY KEY (product_id)
);

CREATE TABLE retailcdp_service_tickets (
    ticket_id VARCHAR(16),
    customer_id VARCHAR(16),
    created_date DATE,
    channel VARCHAR(32),
    issue_type VARCHAR(64),
    sentiment VARCHAR(32),
    sla_hours INTEGER,
    resolution_status VARCHAR(32),
    compensation_amount NUMERIC(12,2),
    PRIMARY KEY (ticket_id)
);

CREATE TABLE retailcdp_stores (
    store_id VARCHAR(16),
    store_name VARCHAR(128),
    city VARCHAR(64),
    province VARCHAR(64),
    region VARCHAR(64),
    city_tier VARCHAR(32),
    store_type VARCHAR(32),
    manager VARCHAR(64),
    opening_date DATE,
    PRIMARY KEY (store_id)
);

CREATE TABLE retailcdp_transaction_items (
    item_id VARCHAR(16),
    transaction_id VARCHAR(16),
    product_id VARCHAR(16),
    quantity INTEGER,
    unit_price NUMERIC(14,2),
    discount_rate NUMERIC(8,4),
    net_amount NUMERIC(14,2),
    PRIMARY KEY (item_id)
);

CREATE TABLE retailcdp_transactions (
    transaction_id VARCHAR(16),
    customer_id VARCHAR(16),
    transaction_date DATE,
    channel VARCHAR(32),
    store_id VARCHAR(16),
    total_amount NUMERIC(14,2),
    discount_amount NUMERIC(14,2),
    payment_method VARCHAR(32),
    order_status VARCHAR(32),
    campaign_id VARCHAR(16),
    PRIMARY KEY (transaction_id)
);

\copy retailcdp_agent_runs (run_id, timestamp, agent_id, scenario_id, status, latency_ms, input_tokens, output_tokens, tool_calls, estimated_cost_usd, eval_score, human_approval_required) FROM '/Users/cid/Documents/csv/agent_runs.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_behavior_events (event_id, customer_id, event_timestamp, event_type, channel, product_id, session_id, value) FROM '/Users/cid/Documents/csv/behavior_events.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_campaign_interactions (interaction_id, campaign_id, customer_id, sent_date, channel, event_status, revenue_attributed, suppression_reason) FROM '/Users/cid/Documents/csv/campaign_interactions.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_campaigns (campaign_id, scenario_id, campaign_name, objective, channel_mix, start_date, end_date, budget, status) FROM '/Users/cid/Documents/csv/campaigns.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_consents (customer_id, marketing_consent, sms_consent, wechat_consent, email_consent, consent_source, updated_at) FROM '/Users/cid/Documents/csv/consents.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_customers (customer_id, name, gender, age, city, province, region, city_tier, join_date, member_level, lifecycle_stage, preferred_channel, home_store_id, category_affinity, size_profile, price_sensitivity, discount_affinity, private_domain_status, acquisition_channel, mobile_hash, email_hash, wechat_openid_hash, consent_marketing, consent_sms, consent_wechat, consent_email, last_purchase_date, days_since_purchase, orders_365, total_spend_365, avg_order_value, rfm_segment, churn_risk_score, replenishment_due_score, vip_score, return_rate, complaint_count, created_at) FROM '/Users/cid/Documents/csv/customers.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_inventory (inventory_id, store_id, product_id, stock_on_hand, reserved_qty, available_qty, reorder_point, last_stocktake_date) FROM '/Users/cid/Documents/csv/inventory.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_loyalty_accounts (customer_id, loyalty_id, tier, points_balance, points_expiring_90d, anniversary_month, next_tier_gap_amount) FROM '/Users/cid/Documents/csv/loyalty_accounts.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_products (product_id, sku, product_name, category, subcategory, brand_line, gender, season, unit_price, replenishment_cycle_days, launch_date, is_hero_sku) FROM '/Users/cid/Documents/csv/products.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_service_tickets (ticket_id, customer_id, created_date, channel, issue_type, sentiment, sla_hours, resolution_status, compensation_amount) FROM '/Users/cid/Documents/csv/service_tickets.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_stores (store_id, store_name, city, province, region, city_tier, store_type, manager, opening_date) FROM '/Users/cid/Documents/csv/stores.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_transaction_items (item_id, transaction_id, product_id, quantity, unit_price, discount_rate, net_amount) FROM '/Users/cid/Documents/csv/transaction_items.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');
\copy retailcdp_transactions (transaction_id, customer_id, transaction_date, channel, store_id, total_amount, discount_amount, payment_method, order_status, campaign_id) FROM '/Users/cid/Documents/csv/transactions.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');

CREATE INDEX idx_retailcdp_transactions_customer ON retailcdp_transactions (customer_id);
CREATE INDEX idx_retailcdp_transactions_date ON retailcdp_transactions (transaction_date);
CREATE INDEX idx_retailcdp_transactions_campaign ON retailcdp_transactions (campaign_id);
CREATE INDEX idx_retailcdp_transaction_items_txn ON retailcdp_transaction_items (transaction_id);
CREATE INDEX idx_retailcdp_transaction_items_product ON retailcdp_transaction_items (product_id);
CREATE INDEX idx_retailcdp_behavior_events_customer ON retailcdp_behavior_events (customer_id);
CREATE INDEX idx_retailcdp_behavior_events_time ON retailcdp_behavior_events (event_timestamp);
CREATE INDEX idx_retailcdp_campaign_interactions_campaign ON retailcdp_campaign_interactions (campaign_id);
CREATE INDEX idx_retailcdp_campaign_interactions_customer ON retailcdp_campaign_interactions (customer_id);
CREATE INDEX idx_retailcdp_service_tickets_customer ON retailcdp_service_tickets (customer_id);

CREATE OR REPLACE VIEW retailcdp_v_customer_360 AS
SELECT
    c.customer_id,
    c.name,
    c.gender,
    c.age,
    c.city,
    c.province,
    c.region,
    c.city_tier,
    c.member_level,
    c.lifecycle_stage,
    c.preferred_channel,
    c.category_affinity,
    c.price_sensitivity,
    c.discount_affinity,
    c.private_domain_status,
    c.acquisition_channel,
    c.last_purchase_date,
    c.days_since_purchase,
    c.orders_365,
    c.total_spend_365,
    c.avg_order_value,
    c.rfm_segment,
    c.churn_risk_score,
    c.replenishment_due_score,
    c.vip_score,
    c.return_rate,
    c.complaint_count,
    s.store_name AS home_store_name,
    s.region AS home_store_region,
    l.tier AS loyalty_tier,
    l.points_balance,
    l.points_expiring_90d,
    co.marketing_consent,
    co.sms_consent,
    co.wechat_consent,
    co.email_consent,
    COALESCE(tx.txn_count, 0) AS transaction_count,
    COALESCE(tx.revenue, 0) AS transaction_revenue,
    COALESCE(tx.discount_amount, 0) AS transaction_discount_amount,
    tx.first_transaction_date,
    tx.last_transaction_date,
    COALESCE(be.behavior_event_count, 0) AS behavior_event_count,
    be.last_behavior_at,
    COALESCE(ci.campaign_touch_count, 0) AS campaign_touch_count,
    COALESCE(ci.campaign_revenue_attributed, 0) AS campaign_revenue_attributed,
    COALESCE(st.service_ticket_count, 0) AS service_ticket_count,
    COALESCE(st.open_or_escalated_ticket_count, 0) AS open_or_escalated_ticket_count,
    COALESCE(st.compensation_amount, 0) AS compensation_amount
FROM retailcdp_customers c
LEFT JOIN retailcdp_stores s ON s.store_id = c.home_store_id
LEFT JOIN retailcdp_loyalty_accounts l ON l.customer_id = c.customer_id
LEFT JOIN retailcdp_consents co ON co.customer_id = c.customer_id
LEFT JOIN (
    SELECT customer_id, COUNT(*) AS txn_count, SUM(total_amount) AS revenue,
           SUM(discount_amount) AS discount_amount, MIN(transaction_date) AS first_transaction_date,
           MAX(transaction_date) AS last_transaction_date
    FROM retailcdp_transactions
    GROUP BY customer_id
) tx ON tx.customer_id = c.customer_id
LEFT JOIN (
    SELECT customer_id, COUNT(*) AS behavior_event_count, MAX(event_timestamp) AS last_behavior_at
    FROM retailcdp_behavior_events
    GROUP BY customer_id
) be ON be.customer_id = c.customer_id
LEFT JOIN (
    SELECT customer_id, COUNT(*) AS campaign_touch_count, SUM(revenue_attributed) AS campaign_revenue_attributed
    FROM retailcdp_campaign_interactions
    GROUP BY customer_id
) ci ON ci.customer_id = c.customer_id
LEFT JOIN (
    SELECT customer_id, COUNT(*) AS service_ticket_count,
           COUNT(*) FILTER (WHERE resolution_status IN ('Open', 'Escalated')) AS open_or_escalated_ticket_count,
           SUM(compensation_amount) AS compensation_amount
    FROM retailcdp_service_tickets
    GROUP BY customer_id
) st ON st.customer_id = c.customer_id;

CREATE OR REPLACE VIEW retailcdp_v_transaction_wide AS
SELECT
    t.transaction_id,
    t.transaction_date,
    t.customer_id,
    c.name AS customer_name,
    c.region AS customer_region,
    c.city AS customer_city,
    c.member_level,
    c.lifecycle_stage,
    t.channel,
    t.store_id,
    s.store_name,
    s.region AS store_region,
    t.payment_method,
    t.order_status,
    t.campaign_id,
    ca.campaign_name,
    ti.item_id,
    ti.product_id,
    p.sku,
    p.product_name,
    p.category,
    p.subcategory,
    p.brand_line,
    p.season,
    ti.quantity,
    ti.unit_price,
    ti.discount_rate,
    ti.net_amount,
    t.total_amount,
    t.discount_amount
FROM retailcdp_transactions t
LEFT JOIN retailcdp_transaction_items ti ON ti.transaction_id = t.transaction_id
LEFT JOIN retailcdp_products p ON p.product_id = ti.product_id
LEFT JOIN retailcdp_customers c ON c.customer_id = t.customer_id
LEFT JOIN retailcdp_stores s ON s.store_id = t.store_id
LEFT JOIN retailcdp_campaigns ca ON ca.campaign_id = t.campaign_id;

CREATE OR REPLACE VIEW retailcdp_v_campaign_performance AS
SELECT
    c.campaign_id,
    c.scenario_id,
    c.campaign_name,
    c.objective,
    c.channel_mix,
    c.start_date,
    c.end_date,
    c.budget,
    c.status,
    COUNT(i.interaction_id) AS touch_count,
    COUNT(DISTINCT i.customer_id) AS reached_customers,
    COUNT(*) FILTER (WHERE i.event_status = 'opened') AS opened_count,
    COUNT(*) FILTER (WHERE i.event_status = 'clicked') AS clicked_count,
    COUNT(*) FILTER (WHERE i.event_status = 'converted') AS converted_count,
    SUM(i.revenue_attributed) AS revenue_attributed,
    CASE WHEN c.budget > 0 THEN SUM(i.revenue_attributed) / c.budget ELSE NULL END AS revenue_to_budget_ratio
FROM retailcdp_campaigns c
LEFT JOIN retailcdp_campaign_interactions i ON i.campaign_id = c.campaign_id
GROUP BY c.campaign_id, c.scenario_id, c.campaign_name, c.objective, c.channel_mix,
         c.start_date, c.end_date, c.budget, c.status;

CREATE OR REPLACE VIEW retailcdp_v_inventory_health AS
SELECT
    i.inventory_id,
    i.store_id,
    s.store_name,
    s.city,
    s.region,
    i.product_id,
    p.sku,
    p.product_name,
    p.category,
    p.subcategory,
    p.brand_line,
    i.stock_on_hand,
    i.reserved_qty,
    i.available_qty,
    i.reorder_point,
    (i.available_qty <= i.reorder_point) AS below_reorder_point,
    i.last_stocktake_date
FROM retailcdp_inventory i
LEFT JOIN retailcdp_stores s ON s.store_id = i.store_id
LEFT JOIN retailcdp_products p ON p.product_id = i.product_id;

CREATE OR REPLACE VIEW retailcdp_v_agent_run_summary AS
SELECT
    date_trunc('day', timestamp)::date AS run_date,
    agent_id,
    scenario_id,
    status,
    COUNT(*) AS run_count,
    AVG(latency_ms) AS avg_latency_ms,
    SUM(input_tokens) AS input_tokens,
    SUM(output_tokens) AS output_tokens,
    SUM(tool_calls) AS tool_calls,
    SUM(estimated_cost_usd) AS estimated_cost_usd,
    AVG(eval_score) AS avg_eval_score,
    COUNT(*) FILTER (WHERE human_approval_required = 'Y') AS human_approval_required_count
FROM retailcdp_agent_runs
GROUP BY date_trunc('day', timestamp)::date, agent_id, scenario_id, status;
