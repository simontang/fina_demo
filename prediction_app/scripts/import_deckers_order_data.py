#!/usr/bin/env python3
"""
将 deckers_order_data.csv 导入到 PostgreSQL 数据库
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
    """创建 deckers_order_data 表"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS deckers_order_data (
        id SERIAL PRIMARY KEY,
        ou_name VARCHAR(200),
        customer_number VARCHAR(100),
        customer_name VARCHAR(200),
        brand VARCHAR(100),
        sales_order INTEGER,
        cust_po_number VARCHAR(100),
        order_type VARCHAR(50),
        head_status VARCHAR(50),
        line_number DECIMAL(10, 2),
        style VARCHAR(100),
        color VARCHAR(100),
        style_color VARCHAR(100),
        blank DECIMAL(10, 2),
        sku VARCHAR(100),
        sku_desc TEXT,
        upc BIGINT,
        line_status VARCHAR(50),
        comment1 TEXT,
        selling_season_code VARCHAR(50),
        ordered_date DATE,
        request_date DATE,
        schedule_date DATE,
        lad DATE,
        demand_class VARCHAR(100),
        open_qty INTEGER,
        rrp INTEGER,
        discount DECIMAL(10, 4),
        unit_selling_price_with_discount DECIMAL(10, 2),
        selling_price_with_discount DECIMAL(10, 2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_sales_order ON deckers_order_data(sales_order);
    CREATE INDEX IF NOT EXISTS idx_customer_number ON deckers_order_data(customer_number);
    CREATE INDEX IF NOT EXISTS idx_ordered_date ON deckers_order_data(ordered_date);
    CREATE INDEX IF NOT EXISTS idx_sku ON deckers_order_data(sku);
    CREATE INDEX IF NOT EXISTS idx_brand ON deckers_order_data(brand);
    CREATE INDEX IF NOT EXISTS idx_selling_season_code ON deckers_order_data(selling_season_code);
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            conn.commit()
            logger.info("✅ 表 deckers_order_data 创建成功（如果不存在）")
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ 创建表失败: {e}")
        raise


def parse_date(date_val):
    """解析日期"""
    try:
        if pd.isna(date_val):
            return None
        if isinstance(date_val, datetime):
            return date_val.date()
        if isinstance(date_val, pd.Timestamp):
            return date_val.date()
        return pd.to_datetime(date_val).date()
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
    
    # 处理日期字段
    date_columns = ['ORDERED_DATE', 'REQUEST_DATE', 'SCHEDULE_DATE', 'LAD']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_date)
    
    # 处理数值字段
    numeric_columns = {
        'SALES_ORDER': 'int64',
        'LINE_NUMBER': 'float64',
        'BLANK': 'float64',
        'UPC': 'int64',
        'OPEN_QTY': 'int64',
        'RRP': 'int64',
        'discount': 'float64',
        'unit selling price with discount': 'float64',
        'selling price with discount': 'float64'
    }
    for col, dtype in numeric_columns.items():
        if col in df.columns:
            if dtype == 'int64':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('Int64')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # 处理文本字段
    text_columns = ['OU_NAME', 'CUSTOMER_NUMBER', 'customer name', 'BRAND', 'CUST_PO_NUMBER',
                   'ORDER_TYPE', 'HEAD_STATUS', 'STYLE', 'COLOR', 'STYLE_COLOR', 'SKU',
                   'SKU_DESC', 'LINE_STATUS', 'COMMENT1', 'SELLING_SEASON_CODE', 'DEMAND_CLASS']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    
    # 检查表是否已有数据
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM deckers_order_data")
        existing_count = cur.fetchone()[0]
        
        if existing_count > 0:
            logger.warning(f"⚠️  表中已有 {existing_count} 条数据")
            response = input("是否清空现有数据并重新导入？(y/N): ")
            if response.lower() == 'y':
                cur.execute("TRUNCATE TABLE deckers_order_data")
                conn.commit()
                logger.info("✅ 已清空现有数据")
            else:
                logger.info("跳过导入，保留现有数据")
                return
    
    # 批量插入数据
    logger.info(f"📤 开始批量导入数据（批次大小: {batch_size}）...")
    
    insert_sql = """
    INSERT INTO deckers_order_data (
        ou_name, customer_number, customer_name, brand, sales_order,
        cust_po_number, order_type, head_status, line_number, style,
        color, style_color, blank, sku, sku_desc, upc, line_status,
        comment1, selling_season_code, ordered_date, request_date,
        schedule_date, lad, demand_class, open_qty, rrp, discount,
        unit_selling_price_with_discount, selling_price_with_discount
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
                        str(row['OU_NAME']) if pd.notna(row['OU_NAME']) else '',
                        str(row['CUSTOMER_NUMBER']) if pd.notna(row['CUSTOMER_NUMBER']) else '',
                        str(row['customer name']) if pd.notna(row['customer name']) else '',
                        str(row['BRAND']) if pd.notna(row['BRAND']) else '',
                        int(row['SALES_ORDER']) if pd.notna(row['SALES_ORDER']) else None,
                        str(row['CUST_PO_NUMBER']) if pd.notna(row['CUST_PO_NUMBER']) else '',
                        str(row['ORDER_TYPE']) if pd.notna(row['ORDER_TYPE']) else '',
                        str(row['HEAD_STATUS']) if pd.notna(row['HEAD_STATUS']) else '',
                        float(row['LINE_NUMBER']) if pd.notna(row['LINE_NUMBER']) else None,
                        str(row['STYLE']) if pd.notna(row['STYLE']) else '',
                        str(row['COLOR']) if pd.notna(row['COLOR']) else '',
                        str(row['STYLE_COLOR']) if pd.notna(row['STYLE_COLOR']) else '',
                        float(row['BLANK']) if pd.notna(row['BLANK']) else None,
                        str(row['SKU']) if pd.notna(row['SKU']) else '',
                        str(row['SKU_DESC']) if pd.notna(row['SKU_DESC']) else '',
                        int(row['UPC']) if pd.notna(row['UPC']) else None,
                        str(row['LINE_STATUS']) if pd.notna(row['LINE_STATUS']) else '',
                        str(row['COMMENT1']) if pd.notna(row['COMMENT1']) else '',
                        str(row['SELLING_SEASON_CODE']) if pd.notna(row['SELLING_SEASON_CODE']) else '',
                        row['ORDERED_DATE'] if pd.notna(row['ORDERED_DATE']) else None,
                        row['REQUEST_DATE'] if pd.notna(row['REQUEST_DATE']) else None,
                        row['SCHEDULE_DATE'] if pd.notna(row['SCHEDULE_DATE']) else None,
                        row['LAD'] if pd.notna(row['LAD']) else None,
                        str(row['DEMAND_CLASS']) if pd.notna(row['DEMAND_CLASS']) else '',
                        int(row['OPEN_QTY']) if pd.notna(row['OPEN_QTY']) else 0,
                        int(row['RRP']) if pd.notna(row['RRP']) else 0,
                        float(row['discount']) if pd.notna(row['discount']) else 0.0,
                        float(row['unit selling price with discount']) if pd.notna(row['unit selling price with discount']) else 0.0,
                        float(row['selling price with discount']) if pd.notna(row['selling price with discount']) else 0.0
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
    csv_path = project_root / "raw_data" / "deckers_order_data.csv"
    
    if not csv_path.exists():
        logger.error(f"❌ CSV 文件不存在: {csv_path}")
        sys.exit(1)
    
    logger.info("🚀 开始导入 deckers_order_data.csv 到数据库")
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
            cur.execute("SELECT COUNT(*) FROM deckers_order_data")
            count = cur.fetchone()[0]
            logger.info(f"✅ 验证：数据库中现有 {count} 条记录")
            
            # 显示一些统计信息
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT sales_order) as total_orders,
                    COUNT(DISTINCT customer_number) as total_customers,
                    COUNT(DISTINCT sku) as total_skus,
                    MIN(ordered_date) as earliest_date,
                    MAX(ordered_date) as latest_date,
                    SUM(selling_price_with_discount) as total_sales,
                    SUM(open_qty) as total_qty
                FROM deckers_order_data
            """)
            stats = cur.fetchone()
            logger.info(f"📊 统计信息:")
            logger.info(f"   - 总订单数: {stats[0]}")
            logger.info(f"   - 总客户数: {stats[1]}")
            logger.info(f"   - 总SKU数: {stats[2]}")
            logger.info(f"   - 最早日期: {stats[3]}")
            logger.info(f"   - 最晚日期: {stats[4]}")
            logger.info(f"   - 总销售额: ${stats[5]:,.2f}")
            logger.info(f"   - 总数量: {stats[6]:,}")
        
    except Exception as e:
        logger.error(f"❌ 导入过程出错: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logger.info("✅ 数据库连接已关闭")


if __name__ == "__main__":
    main()
