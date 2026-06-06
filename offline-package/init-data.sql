-- ============================================
-- Fina Demo 离线版默认数据初始化
-- 创建默认租户、工作区、用户及绑定关系
-- ============================================

-- 默认租户
INSERT INTO lattice_tenants (id, name, description, status, metadata)
VALUES ('default', 'Default Tenant', 'Default tenant for offline deployment', 'active', '{}')
ON CONFLICT (id) DO NOTHING;

-- 默认工作区
INSERT INTO lattice_workspaces (id, tenant_id, name, description, storage_type)
VALUES ('default', 'default', 'Default Workspace', 'Default workspace for offline deployment', 'sandbox')
ON CONFLICT (id, tenant_id) DO NOTHING;

-- 默认项目
INSERT INTO lattice_projects (id, tenant_id, workspace_id, name, description)
VALUES ('default', 'default', 'default', 'Default Project', 'Default project for offline deployment')
ON CONFLICT (id, tenant_id) DO NOTHING;

-- 默认管理员用户（密码: admin）
INSERT INTO lattice_users (id, email, name, status, metadata)
VALUES ('admin', 'admin@fina.ai', 'Administrator', 'active', '{"isAdmin": true, "passwordHash": "f9a81477552594c79f2abc3fc099daa896a6e3a3590a55ffa392b6000412e80b"}')
ON CONFLICT (id) DO NOTHING;

-- 绑定管理员到默认租户
INSERT INTO lattice_user_tenant_links (user_id, tenant_id, role, metadata)
VALUES ('admin', 'default', 'admin', '{}')
ON CONFLICT (user_id, tenant_id) DO NOTHING;

-- ============================================
-- 初始化 Data Agent
-- ============================================
INSERT INTO lattice_assistants (id, tenant_id, name, description, graph_definition)
VALUES (
  'new-data-agent',
  'default',
  'Data Agent',
  'Data Agent',
  '{
    "key": "new-data-agent",
    "name": "Data Agent",
    "type": "deep_agent",
    "tools": [],
    "prompt": "\nYou are a Business Data Analysis Expert.\n\n\n\n",
    "skills": [],
    "subAgents": [],
    "middleware": [
      {
        "id": "filesystem-1",
        "name": "Filesystem",
        "type": "filesystem",
        "config": {"vmIsolation": "global"},
        "enabled": true,
        "description": "Provides file system operations for reading, writing, and managing files"
      },
      {
        "id": "code_eval-2",
        "name": "Code Evaluation",
        "type": "code_eval",
        "config": {},
        "enabled": false,
        "description": "Enables safe code execution"
      },
      {
        "id": "browser-3",
        "name": "Browser",
        "type": "browser",
        "config": {},
        "enabled": false,
        "description": "Provides browser automation capabilities"
      },
      {
        "id": "sql-4",
        "name": "SQL Database",
        "type": "sql",
        "config": {"databaseKeys": []},
        "enabled": false,
        "description": "Provides SQL database query capabilities"
      },
      {
        "id": "skill-5",
        "name": "Skills",
        "type": "skill",
        "config": {"skills": [], "readAll": true},
        "enabled": true,
        "description": "Provides skill loading capabilities for the agent"
      },
      {
        "id": "metrics-6",
        "name": "Metrics",
        "type": "metrics",
        "config": {},
        "enabled": false,
        "description": "Provides metrics querying capabilities for monitoring and observability"
      },
      {
        "id": "ask_user_to_clarify-7",
        "name": "Ask User To Clarify",
        "type": "ask_user_to_clarify",
        "config": {},
        "enabled": false,
        "description": "Enables the agent to ask users clarifying questions with predefined options and free-text input"
      },
      {
        "id": "widget-8",
        "name": "Widget",
        "type": "widget",
        "config": {},
        "enabled": false,
        "description": "Enables the agent to render interactive HTML widgets and visualizations"
      },
      {
        "id": "claw-9",
        "name": "Memory",
        "type": "claw",
        "config": {"injectBootstrapFiles": true},
        "enabled": true,
        "description": "Injects and manages memory/bootstrap files (such as AGENTS.md and USER.md) in the runtime workspace"
      },
      {
        "id": "date-10",
        "name": "Current Date",
        "type": "date",
        "config": {"timezone": "Asia/Shanghai"},
        "enabled": true,
        "description": "Injects the current date into the agent''s system prompt for time awareness"
      },
      {
        "id": "scheduler-11",
        "name": "Scheduler",
        "type": "scheduler",
        "config": {},
        "enabled": false,
        "description": "Enables the agent to schedule future work that re-enters through addMessage"
      },
      {
        "id": "topology-12",
        "name": "Topology",
        "type": "topology",
        "config": {},
        "enabled": false,
        "description": "Restricts which agents can delegate to which other agents. Define edges with purpose descriptions for non-technical readability."
      }
    ],
    "description": "Data Agent"
  }'
)
ON CONFLICT (id) DO NOTHING;
