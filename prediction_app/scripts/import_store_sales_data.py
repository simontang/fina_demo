#!/usr/bin/env python3
"""
将 store_sales_data.csv 导入到 PostgreSQL 数据库
"""
import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
script_dir = Path(__file__).parent.absolute()
prediction_app_dir = script_dir.parent.absolute()
project_root = prediction_app_dir.parent.absolute()
env_path = prediction_app_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"✅ 已加载环境变量: {env_path}")
else:
    logger.warning(f"⚠️  环境变量文件不存在: {env_path}")


def get_db_connection():
    """获取数据库连接"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        logger.info("✅ 数据库连接成功")
        return conn
    except Exception as e:
        logger.error(f"❌ 数据库连接失败: {e}")
        raise


def create_table(conn):
    """创建 store_sales_data 表"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS store_sales_data (
        id SERIAL PRIMARY KEY,
        row_id INTEGER,
        order_id VARCHAR(50),
        order_date DATE,
        ship_date DATE,
        ship_mode VARCHAR(50),
        customer_id VARCHAR(50),
        customer_name VARCHAR(200),
        segment VARCHAR(50),
        country VARCHAR(100),
        city VARCHAR(100),
        state VARCHAR(100),
        postal_code VARCHAR(20),
        region VARCHAR(50),
        product_id VARCHAR(50),
        category VARCHAR(50),
        sub_category VARCHAR(50),
        product_name TEXT,
        sales DECIMAL(10, 2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_order_id ON store_sales_data(order_id);
    CREATE INDEX IF NOT EXISTS idx_customer_id ON store_sales_data(customer_id);
    CREATE INDEX IF NOT EXISTS idx_order_date ON store_sales_data(order_date);
    CREATE INDEX IF NOT EXISTS idx_product_id ON store_sales_data(product_id);
    CREATE INDEX IF NOT EXISTS idx_category ON store_sales_data(category);
    CREATE INDEX IF NOT EXISTS idx_region ON store_sales_data(region);
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            conn.commit()
            logger.info("✅ 表 store_sales_data 创建成功（如果不存在）")
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ 创建表失败: {e}")
        raise


def parse_date(date_str):
    """解析日期字符串 - 格式：MM/DD/YYYY"""
    try:
        if pd.isna(date_str):
            return None
        # 格式：08/11/2017
        return pd.to_datetime(date_str, format='%m/%d/%Y').date()
    except:
        try:
            return pd.to_datetime(date_str).date()
        except:
            return None


def import_data(conn, csv_path, batch_size=10000):
    """导入数据到数据库"""
    logger.info(f"📂 开始读取 CSV 文件: {csv_path}")
    
    # 读取 CSV 文件，尝试不同的编码
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, low_memory=False, encoding=encoding)
            logger.info(f"✅ 使用编码 {encoding} 成功读取文件")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        logger.warning("⚠️  尝试使用 errors='replace' 读取文件")
        df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8', errors='replace')
    
    logger.info(f"📊 读取到 {len(df)} 行数据")
    
    # 数据预处理
    logger.info("🔄 开始数据预处理...")
    
    # 处理日期
    df['Order Date'] = df['Order Date'].apply(parse_date)
    df['Ship Date'] = df['Ship Date'].apply(parse_date)
    
    # 处理数值字段
    df['Row ID'] = pd.to_numeric(df['Row ID'], errors='coerce').fillna(0).astype(int)
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0.0)
    
    # 处理文本字段
    text_columns = ['Order ID', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment',
                   'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID',
                   'Category', 'Sub-Category', 'Product Name']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    
    # 检查表是否已有数据
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM store_sales_data")
        existing_count = cur.fetchone()[0]
        
        if existing_count > 0:
            logger.warning(f"⚠️  表中已有 {existing_count} 条数据")
            response = input("是否清空现有数据并重新导入？(y/N): ")
            if response.lower() == 'y':
                cur.execute("TRUNCATE TABLE store_sales_data")
                conn.commit()
                logger.info("✅ 已清空现有数据")
            else:
                logger.info("跳过导入，保留现有数据")
                return
    
    # 批量插入数据
    logger.info(f"📤 开始批量导入数据（批次大小: {batch_size}）...")
    
    insert_sql = """
    INSERT INTO store_sales_data (
        row_id, order_id, order_date, ship_date, ship_mode,
        customer_id, customer_name, segment, country, city, state,
        postal_code, region, product_id, category, sub_category,
        product_name, sales
    ) VALUES %s
    """
    
    total_rows = len(df)
    inserted_rows = 0
    
    try:
        with conn.cursor() as cur:
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                
                # 准备数据
                values = [
                    (
                        int(row['Row ID']) if pd.notna(row['Row ID']) else None,
                        str(row['Order ID']),
                        row['Order Date'] if pd.notna(row['Order Date']) else None,
                        row['Ship Date'] if pd.notna(row['Ship Date']) else None,
                        str(row['Ship Mode']),
                        str(row['Customer ID']),
                        str(row['Customer Name']),
                        str(row['Segment']),
                        str(row['Country']),
                        str(row['City']),
                        str(row['State']),
                        str(row['Postal Code']),
                        str(row['Region']),
                        str(row['Product ID']),
                        str(row['Category']),
                        str(row['Sub-Category']),
                        str(row['Product Name']),
                        float(row['Sales']) if pd.notna(row['Sales']) else 0.0
                    )
                    for _, row in batch.iterrows()
                ]
                
                # 批量插入
                execute_values(cur, insert_sql, values)
                conn.commit()
                
                inserted_rows += len(batch)
                progress = (inserted_rows / total_rows) * 100
                logger.info(f"📈 进度: {inserted_rows}/{total_rows} ({progress:.1f}%)")
        
        logger.info(f"✅ 数据导入完成！共导入 {inserted_rows} 条记录")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ 数据导入失败: {e}")
        raise


def main():
    """主函数"""
    # CSV 文件路径
    csv_path = project_root / "raw_data" / "store_sales_data.csv"
    
    if not csv_path.exists():
        logger.error(f"❌ CSV 文件不存在: {csv_path}")
        sys.exit(1)
    
    logger.info("🚀 开始导入 store_sales_data.csv 到数据库")
    logger.info(f"📁 CSV 文件路径: {csv_path}")
    
    # 连接数据库
    conn = None
    try:
        conn = get_db_connection()
        
        # 创建表
        create_table(conn)
        
        # 导入数据
        import_data(conn, csv_path)
        
        # 验证导入结果
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM store_sales_data")
            count = cur.fetchone()[0]
            logger.info(f"✅ 验证：数据库中现有 {count} 条记录")
            
            # 显示一些统计信息
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT order_id) as total_orders,
                    COUNT(DISTINCT customer_id) as total_customers,
                    COUNT(DISTINCT product_id) as total_products,
                    MIN(order_date) as earliest_date,
                    MAX(order_date) as latest_date,
                    SUM(sales) as total_sales
                FROM store_sales_data
            """)
            stats = cur.fetchone()
            logger.info(f"📊 统计信息:")
            logger.info(f"   - 总订单数: {stats[0]}")
            logger.info(f"   - 总客户数: {stats[1]}")
            logger.info(f"   - 总产品数: {stats[2]}")
            logger.info(f"   - 最早日期: {stats[3]}")
            logger.info(f"   - 最晚日期: {stats[4]}")
            logger.info(f"   - 总销售额: ${stats[5]:,.2f}")
        
    except Exception as e:
        logger.error(f"❌ 导入过程出错: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logger.info("✅ 数据库连接已关闭")


if __name__ == "__main__":
    main()
