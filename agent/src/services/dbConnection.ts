import { Pool, PoolClient } from "pg";
import * as dotenv from "dotenv";
import * as path from "path";
import * as fs from "fs";

// 加载环境变量
// 优先尝试加载 prediction_app/.env
const predictionAppEnvPath = path.join(
  __dirname,
  "../../../prediction_app/.env"
);
if (fs.existsSync(predictionAppEnvPath)) {
  dotenv.config({ path: predictionAppEnvPath });
  console.log(`✅ Loaded database config from: ${predictionAppEnvPath}`);
} else {
  // 回退到 agent/.env
  dotenv.config({ path: path.join(__dirname, "../../.env") });
  console.log("⚠️ Using agent/.env for database config");
}

let pool: Pool | null = null;

/**
 * 获取 PostgreSQL 连接池
 */
export function getDbPool(): Pool {
  if (!pool) {
    // 从环境变量读取数据库配置
    // 优先使用 DATABASE_URL，如果没有则使用单独的配置项
    const databaseUrl = process.env.DATABASE_URL;
    
    if (databaseUrl) {
      pool = new Pool({
        connectionString: databaseUrl,
        max: 10,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 2000,
      });
    } else {
      // 从 prediction_app/.env 读取配置
      // 注意：这里需要确保环境变量已加载
      pool = new Pool({
        host: process.env.DB_HOST || "localhost",
        port: parseInt(process.env.DB_PORT || "5432"),
        database: process.env.DB_NAME || "postgres",
        user: process.env.DB_USER || "postgres",
        password: process.env.DB_PASSWORD || "",
        max: 10,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 2000,
      });
    }

    pool.on("error", (err) => {
      console.error("Unexpected error on idle client", err);
    });
  }

  return pool;
}

/**
 * 执行查询
 */
export async function query<T = any>(
  text: string,
  params?: any[]
): Promise<T[]> {
  const pool = getDbPool();
  const result = await pool.query(text, params);
  return result.rows;
}

/**
 * 执行单个查询并返回第一行
 */
export async function queryOne<T = any>(
  text: string,
  params?: any[]
): Promise<T | null> {
  const rows = await query<T>(text, params);
  return rows.length > 0 ? rows[0] : null;
}

/**
 * 关闭连接池
 */
export async function closePool(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
  }
}
