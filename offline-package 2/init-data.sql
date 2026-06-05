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
VALUES ('admin', 'admin@localhost', 'Administrator', 'active', '{"isAdmin": true, "passwordHash": "f9a81477552594c79f2abc3fc099daa896a6e3a3590a55ffa392b6000412e80b"}')
ON CONFLICT (id) DO NOTHING;

-- 绑定管理员到默认租户
INSERT INTO lattice_user_tenant_links (user_id, tenant_id, role, metadata)
VALUES ('admin', 'default', 'admin', '{}')
ON CONFLICT (user_id, tenant_id) DO NOTHING;
