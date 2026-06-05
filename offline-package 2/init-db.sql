-- ============================================
-- Fina Demo 离线版数据库初始化脚本
-- ============================================

-- 创建 uploads 相关表（如果 prediction_app 需要）
-- 根据实际需求在此添加表结构初始化

-- 示例：创建一个简单的测试表
CREATE TABLE IF NOT EXISTS demo_table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 如果有其他初始化需求，请在此添加
