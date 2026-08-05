-- Caterpillar-prefixed schema generated from kater_database_schema_postgresql.sql.
-- All business tables, named constraints, indexes, triggers, and functions are isolated by prefix.
-- 卡特营销线索与客户运营系统数据库结构
-- Database: PostgreSQL 15+
-- Encoding: UTF-8

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================
-- 1. 基础组织、渠道及用户权限
-- ============================================================

CREATE TABLE caterpillar_agency (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agency_code           VARCHAR(50) NOT NULL UNIQUE,
    agency_name           VARCHAR(200) NOT NULL,
    contact_name          VARCHAR(100),
    contact_mobile        VARCHAR(32),
    region_scope          JSONB NOT NULL DEFAULT '[]'::jsonb,
    product_scope         JSONB NOT NULL DEFAULT '[]'::jsonb,
    daily_capacity        INTEGER CHECK (daily_capacity IS NULL OR daily_capacity >= 0),
    distribution_weight   NUMERIC(8,4) NOT NULL DEFAULT 1 CHECK (distribution_weight >= 0),
    callback_url          VARCHAR(1000),
    status                VARCHAR(20) NOT NULL DEFAULT 'ENABLED'
                          CHECK (status IN ('ENABLED','DISABLED')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE caterpillar_system_user (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username              VARCHAR(100) NOT NULL UNIQUE,
    display_name          VARCHAR(100) NOT NULL,
    mobile                VARCHAR(32),
    email                 VARCHAR(200),
    agency_id             UUID REFERENCES caterpillar_agency(id) ON DELETE SET NULL,
    department            VARCHAR(100),
    status                VARCHAR(20) NOT NULL DEFAULT 'ENABLED'
                          CHECK (status IN ('ENABLED','DISABLED','LOCKED')),
    last_login_at         TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE caterpillar_role (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_code             VARCHAR(50) NOT NULL UNIQUE,
    role_name             VARCHAR(100) NOT NULL,
    data_scope            VARCHAR(30) NOT NULL DEFAULT 'SELF'
                          CHECK (data_scope IN ('ALL','DEPARTMENT','AGENCY','SELF')),
    permission_json       JSONB NOT NULL DEFAULT '{}'::jsonb,
    status                VARCHAR(20) NOT NULL DEFAULT 'ENABLED'
                          CHECK (status IN ('ENABLED','DISABLED')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE caterpillar_user_role (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               UUID NOT NULL REFERENCES caterpillar_system_user(id) ON DELETE CASCADE,
    role_id               UUID NOT NULL REFERENCES caterpillar_role(id) ON DELETE CASCADE,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT caterpillar_uk_user_role UNIQUE (user_id, role_id)
);

CREATE TABLE caterpillar_channel (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_code          VARCHAR(50) NOT NULL UNIQUE,
    channel_name          VARCHAR(100) NOT NULL,
    channel_type          VARCHAR(30) NOT NULL,
    parent_id             UUID REFERENCES caterpillar_channel(id) ON DELETE SET NULL,
    media_platform        VARCHAR(50),
    external_channel_id   VARCHAR(100),
    status                VARCHAR(20) NOT NULL DEFAULT 'ENABLED'
                          CHECK (status IN ('ENABLED','DISABLED')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE caterpillar_campaign (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_code         VARCHAR(50) NOT NULL UNIQUE,
    campaign_name         VARCHAR(200) NOT NULL,
    campaign_type         VARCHAR(30),
    channel_id            UUID REFERENCES caterpillar_channel(id) ON DELETE SET NULL,
    start_at              TIMESTAMPTZ,
    end_at                TIMESTAMPTZ,
    budget                NUMERIC(18,2) CHECK (budget IS NULL OR budget >= 0),
    utm_source            VARCHAR(100),
    utm_medium            VARCHAR(100),
    utm_campaign          VARCHAR(100),
    owner_user_id         UUID REFERENCES caterpillar_system_user(id) ON DELETE SET NULL,
    status                VARCHAR(20) NOT NULL DEFAULT 'DRAFT'
                          CHECK (status IN ('DRAFT','ACTIVE','FINISHED','CANCELLED')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT caterpillar_ck_campaign_time CHECK (end_at IS NULL OR start_at IS NULL OR end_at >= start_at)
);

-- ============================================================
-- 2. 客户中心
-- ============================================================

CREATE TABLE caterpillar_customer (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_no           VARCHAR(50) NOT NULL UNIQUE,
    name                  VARCHAR(100),
    mobile                VARCHAR(512),
    mobile_hash           VARCHAR(64),
    gender                VARCHAR(20),
    birthday              DATE,
    province              VARCHAR(50),
    city                  VARCHAR(50),
    district              VARCHAR(50),
    address               VARCHAR(1000),
    customer_level        VARCHAR(30),
    lifecycle_stage       VARCHAR(30),
    first_channel_id      UUID REFERENCES caterpillar_channel(id) ON DELETE SET NULL,
    first_campaign_id     UUID REFERENCES caterpillar_campaign(id) ON DELETE SET NULL,
    owner_user_id         UUID REFERENCES caterpillar_system_user(id) ON DELETE SET NULL,
    consent_status        VARCHAR(20) NOT NULL DEFAULT 'UNKNOWN'
                          CHECK (consent_status IN ('UNKNOWN','GRANTED','REFUSED','WITHDRAWN')),
    consent_time          TIMESTAMPTZ,
    status                VARCHAR(20) NOT NULL DEFAULT 'ACTIVE'
                          CHECK (status IN ('ACTIVE','MERGED','DISABLED')),
    merged_to_id          UUID REFERENCES caterpillar_customer(id) ON DELETE SET NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT caterpillar_ck_customer_not_merge_self CHECK (merged_to_id IS NULL OR merged_to_id <> id)
);

CREATE TABLE caterpillar_customer_identity (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id           UUID NOT NULL REFERENCES caterpillar_customer(id) ON DELETE CASCADE,
    identity_type         VARCHAR(30) NOT NULL,
    identity_value        TEXT NOT NULL,
    identity_hash         VARCHAR(64) NOT NULL,
    platform              VARCHAR(50) NOT NULL DEFAULT '',
    is_primary            BOOLEAN NOT NULL DEFAULT FALSE,
    verified              BOOLEAN NOT NULL DEFAULT FALSE,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT caterpillar_uk_customer_identity UNIQUE (identity_type, identity_hash, platform)
);

CREATE TABLE caterpillar_customer_tag (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tag_code              VARCHAR(50) NOT NULL UNIQUE,
    tag_name              VARCHAR(100) NOT NULL,
    tag_category          VARCHAR(50),
    description           VARCHAR(500),
    rule_json             JSONB,
    status                VARCHAR(20) NOT NULL DEFAULT 'ENABLED'
                          CHECK (status IN ('ENABLED','DISABLED')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE caterpillar_customer_tag_relation (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id           UUID NOT NULL REFERENCES caterpillar_customer(id) ON DELETE CASCADE,
    tag_id                UUID NOT NULL REFERENCES caterpillar_customer_tag(id) ON DELETE CASCADE,
    source_type           VARCHAR(30) NOT NULL,
    source_id             UUID,
    tagged_at             TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expire_at             TIMESTAMPTZ,
    created_by            UUID REFERENCES caterpillar_system_user(id) ON DELETE SET NULL,
    CONSTRAINT caterpillar_uk_customer_tag UNIQUE (customer_id, tag_id),
    CONSTRAINT caterpillar_ck_tag_expiry CHECK (expire_at IS NULL OR expire_at >= tagged_at)
);

-- ============================================================
-- 3. 触点与埋点
-- ============================================================

CREATE TABLE caterpillar_touchpoint (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    touchpoint_code       VARCHAR(50) NOT NULL UNIQUE,
    touchpoint_name       VARCHAR(200) NOT NULL,
    touchpoint_type       VARCHAR(30) NOT NULL
                          CHECK (touchpoint_type IN ('FORM','WEB','MINIPROGRAM','QR','SDK','OFFLINE','OTHER')),
    channel_id            UUID REFERENCES caterpillar_channel(id) ON DELETE SET NULL,
    campaign_id           UUID REFERENCES caterpillar_campaign(id) ON DELETE SET NULL,
    external_app_id       VARCHAR(100),
    page_url              VARCHAR(2000),
    config_json           JSONB NOT NULL DEFAULT '{}'::jsonb,
    status                VARCHAR(20) NOT NULL DEFAULT 'ENABLED'
                          CHECK (status IN ('ENABLED','DISABLED')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE caterpillar_behavior_event (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id              VARCHAR(64) NOT NULL UNIQUE,
    event_name            VARCHAR(100) NOT NULL,
    event_type            VARCHAR(50) NOT NULL,
    customer_id           UUID REFERENCES caterpillar_customer(id) ON DELETE SET NULL,
    anonymous_id          VARCHAR(100),
    session_id            VARCHAR(100),
    touchpoint_id         UUID REFERENCES caterpillar_touchpoint(id) ON DELETE SET NULL,
    channel_id            UUID REFERENCES caterpillar_channel(id) ON DELETE SET NULL,
    campaign_id           UUID REFERENCES caterpillar_campaign(id) ON DELETE SET NULL,
    page_url              VARCHAR(2000),
    page_title            VARCHAR(500),
    referrer              VARCHAR(2000),
    device_type           VARCHAR(30),
    device_id             VARCHAR(100),
    ip_address            VARCHAR(64),
    event_properties      JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at           TIMESTAMPTZ NOT NULL,
    received_at           TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT caterpillar_ck_event_identity CHECK (customer_id IS NOT NULL OR anonymous_id IS NOT NULL)
);

-- ============================================================
-- 4. 线索管理
-- ============================================================

CREATE TABLE caterpillar_lead (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_no               VARCHAR(50) NOT NULL UNIQUE,
    customer_id           UUID REFERENCES caterpillar_customer(id) ON DELETE SET NULL,
    name                  VARCHAR(100),
    mobile                VARCHAR(512),
    mobile_hash           VARCHAR(64),
    channel_id            UUID REFERENCES caterpillar_channel(id) ON DELETE SET NULL,
    campaign_id           UUID REFERENCES caterpillar_campaign(id) ON DELETE SET NULL,
    touchpoint_id         UUID REFERENCES caterpillar_touchpoint(id) ON DELETE SET NULL,
    product_interest      VARCHAR(200),
    province              VARCHAR(50),
    city                  VARCHAR(50),
    lead_score            NUMERIC(8,2),
    quality_level         VARCHAR(20),
    status                VARCHAR(30) NOT NULL DEFAULT 'NEW'
                          CHECK (status IN ('NEW','PENDING_CLEAN','PENDING_CALL','CALLING','VALID','INVALID','ASSIGNED','CONVERTED','CLOSED')),
    invalid_reason        VARCHAR(200),
    duplicate_of_id       UUID REFERENCES caterpillar_lead(id) ON DELETE SET NULL,
    cleaning_result       JSONB NOT NULL DEFAULT '{}'::jsonb,
    received_at           TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT caterpillar_ck_lead_not_duplicate_self CHECK (duplicate_of_id IS NULL OR duplicate_of_id <> id)
);

CREATE TABLE caterpillar_lead_source_record (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id               UUID REFERENCES caterpillar_lead(id) ON DELETE SET NULL,
    touchpoint_id         UUID NOT NULL REFERENCES caterpillar_touchpoint(id) ON DELETE RESTRICT,
    external_record_id    VARCHAR(100),
    request_id            VARCHAR(100),
    raw_data              JSONB NOT NULL,
    validation_status     VARCHAR(20) NOT NULL DEFAULT 'PENDING'
                          CHECK (validation_status IN ('PENDING','PASSED','FAILED')),
    validation_message    TEXT,
    received_at           TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at          TIMESTAMPTZ
);

CREATE TABLE caterpillar_lead_status_history (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id               UUID NOT NULL REFERENCES caterpillar_lead(id) ON DELETE CASCADE,
    from_status           VARCHAR(30),
    to_status             VARCHAR(30) NOT NULL,
    change_reason         VARCHAR(500),
    operator_type         VARCHAR(20) NOT NULL
                          CHECK (operator_type IN ('USER','SYSTEM','AGENCY','CALL_PROVIDER')),
    operator_id           UUID,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 5. 外呼与代理商分配
-- ============================================================

CREATE TABLE caterpillar_call_task (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_no               VARCHAR(50) NOT NULL UNIQUE,
    task_name             VARCHAR(200) NOT NULL,
    campaign_id           UUID REFERENCES caterpillar_campaign(id) ON DELETE SET NULL,
    provider_code         VARCHAR(50),
    call_strategy         JSONB NOT NULL DEFAULT '{}'::jsonb,
    total_count           INTEGER NOT NULL DEFAULT 0 CHECK (total_count >= 0),
    completed_count       INTEGER NOT NULL DEFAULT 0 CHECK (completed_count >= 0),
    status                VARCHAR(20) NOT NULL DEFAULT 'DRAFT'
                          CHECK (status IN ('DRAFT','PENDING','RUNNING','COMPLETED','CANCELLED')),
    start_at              TIMESTAMPTZ,
    end_at                TIMESTAMPTZ,
    created_by            UUID NOT NULL REFERENCES caterpillar_system_user(id) ON DELETE RESTRICT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT caterpillar_ck_call_task_count CHECK (completed_count <= total_count),
    CONSTRAINT caterpillar_ck_call_task_time CHECK (end_at IS NULL OR start_at IS NULL OR end_at >= start_at)
);

CREATE TABLE caterpillar_call_record (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_task_id          UUID NOT NULL REFERENCES caterpillar_call_task(id) ON DELETE CASCADE,
    lead_id               UUID NOT NULL REFERENCES caterpillar_lead(id) ON DELETE RESTRICT,
    external_call_id      VARCHAR(100),
    call_sequence         INTEGER NOT NULL DEFAULT 1 CHECK (call_sequence > 0),
    call_status           VARCHAR(30) NOT NULL
                          CHECK (call_status IN ('PENDING','DIALING','ANSWERED','NO_ANSWER','REJECTED','BUSY','INVALID_NUMBER','FAILED')),
    business_result       VARCHAR(30),
    intent_level          VARCHAR(20),
    agent_name            VARCHAR(100),
    agent_external_id     VARCHAR(100),
    started_at            TIMESTAMPTZ,
    answered_at           TIMESTAMPTZ,
    ended_at              TIMESTAMPTZ,
    duration_seconds      INTEGER CHECK (duration_seconds IS NULL OR duration_seconds >= 0),
    recording_url         VARCHAR(1000),
    remark                TEXT,
    callback_raw_data     JSONB,
    callback_at           TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT caterpillar_uk_call_attempt UNIQUE (call_task_id, lead_id, call_sequence),
    CONSTRAINT caterpillar_uk_external_call UNIQUE (external_call_id)
);

CREATE TABLE caterpillar_lead_assignment (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id               UUID NOT NULL REFERENCES caterpillar_lead(id) ON DELETE RESTRICT,
    agency_id             UUID NOT NULL REFERENCES caterpillar_agency(id) ON DELETE RESTRICT,
    assignment_batch_no   VARCHAR(50),
    assignment_rule       VARCHAR(100),
    assignment_reason     VARCHAR(500),
    assigned_at           TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    delivery_status       VARCHAR(30) NOT NULL DEFAULT 'PENDING'
                          CHECK (delivery_status IN ('PENDING','SENT','ACCEPTED','FAILED','RETURNED')),
    external_lead_id      VARCHAR(100),
    response_code         VARCHAR(50),
    response_message      TEXT,
    followup_status       VARCHAR(30),
    conversion_status     VARCHAR(30),
    converted_at          TIMESTAMPTZ,
    is_current            BOOLEAN NOT NULL DEFAULT TRUE,
    created_by            UUID REFERENCES caterpillar_system_user(id) ON DELETE SET NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 6. 订单与问卷
-- ============================================================

CREATE TABLE caterpillar_order_record (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_no              VARCHAR(100) NOT NULL UNIQUE,
    customer_id           UUID NOT NULL REFERENCES caterpillar_customer(id) ON DELETE RESTRICT,
    lead_id               UUID REFERENCES caterpillar_lead(id) ON DELETE SET NULL,
    agency_id             UUID REFERENCES caterpillar_agency(id) ON DELETE SET NULL,
    campaign_id           UUID REFERENCES caterpillar_campaign(id) ON DELETE SET NULL,
    product_code          VARCHAR(100),
    product_name          VARCHAR(200) NOT NULL,
    quantity              INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
    order_amount          NUMERIC(18,2) NOT NULL CHECK (order_amount >= 0),
    paid_amount           NUMERIC(18,2) NOT NULL DEFAULT 0 CHECK (paid_amount >= 0),
    order_status          VARCHAR(30) NOT NULL
                          CHECK (order_status IN ('PENDING_PAYMENT','PAID','COMPLETED','CANCELLED','REFUNDED')),
    order_at              TIMESTAMPTZ NOT NULL,
    paid_at               TIMESTAMPTZ,
    completed_at          TIMESTAMPTZ,
    external_order_id     VARCHAR(100),
    extra_data            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE caterpillar_survey_response (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    survey_code           VARCHAR(50) NOT NULL,
    survey_version        VARCHAR(20),
    customer_id           UUID REFERENCES caterpillar_customer(id) ON DELETE SET NULL,
    lead_id               UUID REFERENCES caterpillar_lead(id) ON DELETE SET NULL,
    touchpoint_id         UUID REFERENCES caterpillar_touchpoint(id) ON DELETE SET NULL,
    campaign_id           UUID REFERENCES caterpillar_campaign(id) ON DELETE SET NULL,
    anonymous_id          VARCHAR(100),
    status                VARCHAR(20) NOT NULL DEFAULT 'IN_PROGRESS'
                          CHECK (status IN ('IN_PROGRESS','SUBMITTED','INVALID')),
    started_at            TIMESTAMPTZ,
    submitted_at          TIMESTAMPTZ,
    total_score           NUMERIC(10,2),
    extra_data            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT caterpillar_ck_survey_identity CHECK (customer_id IS NOT NULL OR anonymous_id IS NOT NULL)
);

CREATE TABLE caterpillar_survey_answer (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id           UUID NOT NULL REFERENCES caterpillar_survey_response(id) ON DELETE CASCADE,
    question_code         VARCHAR(50) NOT NULL,
    question_text         VARCHAR(1000),
    answer_type           VARCHAR(30) NOT NULL,
    answer_text           TEXT,
    answer_number         NUMERIC(18,4),
    answer_json           JSONB,
    score                 NUMERIC(10,2),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT caterpillar_uk_survey_question UNIQUE (response_id, question_code)
);

-- ============================================================
-- 7. 自定义模块
-- ============================================================

CREATE TABLE caterpillar_custom_module (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module_code           VARCHAR(50) NOT NULL UNIQUE,
    module_name           VARCHAR(100) NOT NULL,
    description           VARCHAR(500),
    primary_object_type   VARCHAR(30) NOT NULL CHECK (primary_object_type IN ('CUSTOMER','LEAD','NONE')),
    display_field_code    VARCHAR(50),
    enable_workflow       BOOLEAN NOT NULL DEFAULT FALSE,
    status                VARCHAR(20) NOT NULL DEFAULT 'ENABLED'
                          CHECK (status IN ('ENABLED','DISABLED')),
    created_by            UUID NOT NULL REFERENCES caterpillar_system_user(id) ON DELETE RESTRICT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE caterpillar_custom_field (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module_id             UUID NOT NULL REFERENCES caterpillar_custom_module(id) ON DELETE CASCADE,
    field_code            VARCHAR(50) NOT NULL,
    field_name            VARCHAR(100) NOT NULL,
    field_type            VARCHAR(30) NOT NULL
                          CHECK (field_type IN ('TEXT','LONG_TEXT','INTEGER','DECIMAL','DATE','DATETIME','BOOLEAN','SINGLE_SELECT','MULTI_SELECT','REFERENCE','JSON')),
    is_required           BOOLEAN NOT NULL DEFAULT FALSE,
    is_unique             BOOLEAN NOT NULL DEFAULT FALSE,
    is_searchable         BOOLEAN NOT NULL DEFAULT FALSE,
    is_sensitive          BOOLEAN NOT NULL DEFAULT FALSE,
    option_json           JSONB,
    validation_json       JSONB,
    sort_order            INTEGER NOT NULL DEFAULT 0,
    status                VARCHAR(20) NOT NULL DEFAULT 'ENABLED'
                          CHECK (status IN ('ENABLED','DISABLED')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT caterpillar_uk_module_field UNIQUE (module_id, field_code)
);

CREATE TABLE caterpillar_custom_record (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module_id             UUID NOT NULL REFERENCES caterpillar_custom_module(id) ON DELETE RESTRICT,
    record_no             VARCHAR(50) NOT NULL,
    customer_id           UUID REFERENCES caterpillar_customer(id) ON DELETE SET NULL,
    lead_id               UUID REFERENCES caterpillar_lead(id) ON DELETE SET NULL,
    record_data           JSONB NOT NULL DEFAULT '{}'::jsonb,
    record_status         VARCHAR(30),
    external_record_id    VARCHAR(100),
    owner_user_id         UUID REFERENCES caterpillar_system_user(id) ON DELETE SET NULL,
    created_by            UUID NOT NULL REFERENCES caterpillar_system_user(id) ON DELETE RESTRICT,
    updated_by            UUID NOT NULL REFERENCES caterpillar_system_user(id) ON DELETE RESTRICT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at            TIMESTAMPTZ,
    CONSTRAINT caterpillar_uk_custom_record_no UNIQUE (module_id, record_no)
);

-- ============================================================
-- 8. 接口与审计日志
-- ============================================================

CREATE TABLE caterpillar_integration_log (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    system_code           VARCHAR(50) NOT NULL,
    interface_code        VARCHAR(100) NOT NULL,
    direction             VARCHAR(10) NOT NULL CHECK (direction IN ('IN','OUT')),
    request_id            VARCHAR(100),
    business_type         VARCHAR(50),
    business_id           UUID,
    request_data          JSONB,
    response_data         JSONB,
    http_status           INTEGER,
    process_status        VARCHAR(20) NOT NULL
                          CHECK (process_status IN ('PROCESSING','SUCCEEDED','FAILED')),
    error_message         TEXT,
    retry_count           INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    started_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at          TIMESTAMPTZ
);

CREATE TABLE caterpillar_audit_log (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               UUID REFERENCES caterpillar_system_user(id) ON DELETE SET NULL,
    action                VARCHAR(50) NOT NULL,
    object_type           VARCHAR(50) NOT NULL,
    object_id             UUID,
    before_data           JSONB,
    after_data            JSONB,
    ip_address            VARCHAR(64),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 9. 核心索引
-- ============================================================

CREATE INDEX caterpillar_idx_system_user_agency ON caterpillar_system_user(agency_id);
CREATE INDEX caterpillar_idx_channel_parent ON caterpillar_channel(parent_id);
CREATE INDEX caterpillar_idx_campaign_channel_status ON caterpillar_campaign(channel_id, status);

CREATE INDEX caterpillar_idx_customer_mobile_hash ON caterpillar_customer(mobile_hash);
CREATE INDEX caterpillar_idx_customer_owner_status ON caterpillar_customer(owner_user_id, status);
CREATE INDEX caterpillar_idx_customer_source_created ON caterpillar_customer(first_channel_id, created_at DESC);
CREATE INDEX caterpillar_idx_customer_identity_customer ON caterpillar_customer_identity(customer_id);
CREATE UNIQUE INDEX caterpillar_uk_customer_primary_identity
    ON caterpillar_customer_identity(customer_id, identity_type)
    WHERE is_primary = TRUE;
CREATE INDEX caterpillar_idx_customer_tag_relation_tag ON caterpillar_customer_tag_relation(tag_id, tagged_at DESC);

CREATE INDEX caterpillar_idx_touchpoint_campaign ON caterpillar_touchpoint(campaign_id, status);
CREATE INDEX caterpillar_idx_behavior_customer_time ON caterpillar_behavior_event(customer_id, occurred_at DESC);
CREATE INDEX caterpillar_idx_behavior_anonymous_time ON caterpillar_behavior_event(anonymous_id, occurred_at DESC);
CREATE INDEX caterpillar_idx_behavior_campaign_event_time ON caterpillar_behavior_event(campaign_id, event_name, occurred_at DESC);
CREATE INDEX caterpillar_idx_behavior_properties_gin ON caterpillar_behavior_event USING GIN(event_properties);

CREATE INDEX caterpillar_idx_lead_customer ON caterpillar_lead(customer_id, created_at DESC);
CREATE INDEX caterpillar_idx_lead_mobile_campaign ON caterpillar_lead(mobile_hash, campaign_id, received_at DESC);
CREATE INDEX caterpillar_idx_lead_status_created ON caterpillar_lead(status, created_at DESC);
CREATE INDEX caterpillar_idx_lead_source_record_lead ON caterpillar_lead_source_record(lead_id);
CREATE INDEX caterpillar_idx_lead_source_request ON caterpillar_lead_source_record(request_id);
CREATE UNIQUE INDEX caterpillar_uk_lead_source_external
    ON caterpillar_lead_source_record(touchpoint_id, external_record_id)
    WHERE external_record_id IS NOT NULL;
CREATE INDEX caterpillar_idx_lead_status_history_lead_time ON caterpillar_lead_status_history(lead_id, created_at DESC);

CREATE INDEX caterpillar_idx_call_record_lead_time ON caterpillar_call_record(lead_id, created_at DESC);
CREATE INDEX caterpillar_idx_call_record_task_status ON caterpillar_call_record(call_task_id, call_status);
CREATE INDEX caterpillar_idx_assignment_lead_time ON caterpillar_lead_assignment(lead_id, assigned_at DESC);
CREATE INDEX caterpillar_idx_assignment_agency_status ON caterpillar_lead_assignment(agency_id, delivery_status, assigned_at DESC);
CREATE UNIQUE INDEX caterpillar_uk_lead_current_assignment
    ON caterpillar_lead_assignment(lead_id)
    WHERE is_current = TRUE;

CREATE INDEX caterpillar_idx_order_customer_time ON caterpillar_order_record(customer_id, order_at DESC);
CREATE INDEX caterpillar_idx_order_lead ON caterpillar_order_record(lead_id);
CREATE INDEX caterpillar_idx_order_campaign_status ON caterpillar_order_record(campaign_id, order_status, order_at DESC);
CREATE UNIQUE INDEX caterpillar_uk_order_external
    ON caterpillar_order_record(external_order_id)
    WHERE external_order_id IS NOT NULL;

CREATE INDEX caterpillar_idx_survey_customer_time ON caterpillar_survey_response(customer_id, submitted_at DESC);
CREATE INDEX caterpillar_idx_survey_campaign_status ON caterpillar_survey_response(campaign_id, status);
CREATE INDEX caterpillar_idx_survey_answer_response ON caterpillar_survey_answer(response_id);

CREATE INDEX caterpillar_idx_custom_record_customer ON caterpillar_custom_record(customer_id, created_at DESC);
CREATE INDEX caterpillar_idx_custom_record_lead ON caterpillar_custom_record(lead_id, created_at DESC);
CREATE INDEX caterpillar_idx_custom_record_data_gin ON caterpillar_custom_record USING GIN(record_data);

CREATE INDEX caterpillar_idx_integration_request ON caterpillar_integration_log(system_code, request_id);
CREATE INDEX caterpillar_idx_integration_status_time ON caterpillar_integration_log(process_status, started_at DESC);
CREATE INDEX caterpillar_idx_integration_business ON caterpillar_integration_log(business_type, business_id);
CREATE INDEX caterpillar_idx_audit_object_time ON caterpillar_audit_log(object_type, object_id, created_at DESC);
CREATE INDEX caterpillar_idx_audit_user_time ON caterpillar_audit_log(user_id, created_at DESC);

-- ============================================================
-- 10. 自动维护 updated_at
-- ============================================================

CREATE OR REPLACE FUNCTION caterpillar_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


DO $$
DECLARE
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'agency','system_user','role','channel','campaign','customer','customer_identity','customer_tag','touchpoint','lead','call_task','call_record','lead_assignment','order_record','custom_module','custom_field','custom_record'
    ]
    LOOP
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE ON %I '
            'FOR EACH ROW EXECUTE FUNCTION caterpillar_set_updated_at()',
            'caterpillar_trg_' || table_name || '_updated_at',
            'caterpillar_' || table_name
        );
    END LOOP;
END;
$$;
COMMIT;

-- 生产环境补充建议：
-- 1. mobile、address、identity_value 等敏感字段应在应用层或数据库层加密。
-- 2. behavior_event、integration_log、audit_log 数据量大时应按 occurred_at/started_at/created_at 按月分区。
-- 3. 禁止在 request_data、response_data、raw_data 中保存未脱敏的密码、Token及身份证号。
