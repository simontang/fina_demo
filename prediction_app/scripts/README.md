# 数据导入脚本

## 功能

将 `raw_data/sales_data.csv` 导入到 PostgreSQL 数据库的 `sales_data` 表中。

## 使用方法

### 1. 安装依赖

```bash
cd prediction_app/scripts
pip install -r requirements.txt
```

### 2. 确保环境变量已配置

确保 `prediction_app/.env` 文件中包含数据库连接信息：
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`

### 3. 运行导入脚本

```bash
python import_sales_data.py
```

## 脚本功能

- ✅ 自动创建 `sales_data` 表（如果不存在）
- ✅ 创建必要的索引以优化查询性能
- ✅ 数据预处理（日期解析、类型转换等）
- ✅ 批量导入（默认每批 10000 条，提高导入速度）
- ✅ 显示导入进度
- ✅ 导入完成后显示统计信息
- ✅ 如果表中已有数据，会提示是否清空重新导入

## 表结构

```sql
CREATE TABLE sales_data (
    id SERIAL PRIMARY KEY,
    invoice_no VARCHAR(50),
    stock_code VARCHAR(50),
    description TEXT,
    quantity INTEGER,
    invoice_date TIMESTAMP,
    unit_price DECIMAL(10, 2),
    customer_id INTEGER,
    country VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 索引

脚本会自动创建以下索引：
- `idx_invoice_no` - 发票号索引
- `idx_stock_code` - 库存代码索引
- `idx_customer_id` - 客户ID索引
- `idx_invoice_date` - 发票日期索引

## 注意事项

- CSV 文件约有 54 万行数据，导入可能需要几分钟时间
- 确保数据库连接正常且有足够的存储空间
- 如果导入过程中断，可以重新运行脚本（会提示是否清空现有数据）
