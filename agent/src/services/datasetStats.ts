import { query, queryOne } from "./dbConnection";

export interface ColumnStats {
  columnName: string;
  dataType: string;
  nullCount: number;
  uniqueCount?: number; // 字符串类型
  distribution?: {
    // 数值类型
    min: number;
    max: number;
    mean: number;
    median: number;
    quartiles: [number, number, number];
  };
}

export interface TimeRange {
  min: string | null;
  max: string | null;
}

// 简单的内存缓存
interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

const cache = new Map<string, CacheEntry<any>>();
const CACHE_TTL = 60 * 60 * 1000; // 1小时

/**
 * 获取缓存的统计信息
 */
function getCached<T>(key: string): T | null {
  const entry = cache.get(key);
  if (!entry) return null;

  const now = Date.now();
  if (now - entry.timestamp > CACHE_TTL) {
    cache.delete(key);
    return null;
  }

  return entry.data;
}

/**
 * 设置缓存
 */
function setCache<T>(key: string, data: T): void {
  cache.set(key, {
    data,
    timestamp: Date.now(),
  });
}

/**
 * 获取表的列信息
 */
export async function getTableColumns(
  tableName: string
): Promise<Array<{ column_name: string; data_type: string }>> {
  const sql = `
    SELECT 
      column_name,
      data_type
    FROM information_schema.columns
    WHERE table_name = $1
    ORDER BY ordinal_position
  `;
  return await query(sql, [tableName]);
}

/**
 * 获取表的记录数
 */
export async function getTableRowCount(tableName: string): Promise<number> {
  const cacheKey = `row_count:${tableName}`;
  const cached = getCached<number>(cacheKey);
  if (cached !== null) return cached;

  const sql = `SELECT COUNT(*) as count FROM ${tableName}`;
  const result = await queryOne<{ count: string }>(sql);
  const count = result ? parseInt(result.count) : 0;

  setCache(cacheKey, count);
  return count;
}

/**
 * 获取时间范围
 */
export async function getTimeRange(
  tableName: string,
  timeColumn: string
): Promise<TimeRange> {
  const cacheKey = `time_range:${tableName}:${timeColumn}`;
  const cached = getCached<TimeRange>(cacheKey);
  if (cached !== null) return cached;

  const sql = `
    SELECT 
      MIN(${timeColumn}) as min,
      MAX(${timeColumn}) as max
    FROM ${tableName}
  `;
  const result = await queryOne<{ min: string | null; max: string | null }>(
    sql
  );

  const timeRange: TimeRange = {
    min: result?.min || null,
    max: result?.max || null,
  };

  setCache(cacheKey, timeRange);
  return timeRange;
}

/**
 * 获取列的统计信息
 */
export async function getColumnStats(
  tableName: string,
  columnName: string,
  dataType: string
): Promise<ColumnStats> {
  const cacheKey = `column_stats:${tableName}:${columnName}`;
  const cached = getCached<ColumnStats>(cacheKey);
  if (cached !== null) return cached;

  // 获取空值数量
  const nullCountResult = await queryOne<{ count: string }>(
    `SELECT COUNT(*) as count FROM ${tableName} WHERE ${columnName} IS NULL`
  );
  const nullCount = nullCountResult
    ? parseInt(nullCountResult.count)
    : 0;

  const stats: ColumnStats = {
    columnName,
    dataType,
    nullCount,
  };

  // 根据数据类型计算不同的统计信息
  if (
    dataType.includes("char") ||
    dataType.includes("text") ||
    dataType === "varchar" ||
    dataType === "character varying"
  ) {
    // 字符串类型：计算唯一值数量
    const uniqueResult = await queryOne<{ count: string }>(
      `SELECT COUNT(DISTINCT ${columnName}) as count FROM ${tableName} WHERE ${columnName} IS NOT NULL`
    );
    stats.uniqueCount = uniqueResult ? parseInt(uniqueResult.count) : 0;
  } else if (
    dataType.includes("int") ||
    dataType.includes("numeric") ||
    dataType.includes("decimal") ||
    dataType.includes("real") ||
    dataType.includes("double") ||
    dataType === "float"
  ) {
    // 数值类型：计算分布统计
    const distributionResult = await queryOne<{
      min: string;
      max: string;
      mean: string;
      median: string;
      q25: string;
      q50: string;
      q75: string;
    }>(
      `
      SELECT 
        MIN(${columnName})::text as min,
        MAX(${columnName})::text as max,
        AVG(${columnName})::text as mean,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ${columnName})::text as median,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ${columnName})::text as q25,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ${columnName})::text as q50,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ${columnName})::text as q75
      FROM ${tableName}
      WHERE ${columnName} IS NOT NULL
    `
    );

    if (distributionResult) {
      stats.distribution = {
        min: parseFloat(distributionResult.min),
        max: parseFloat(distributionResult.max),
        mean: parseFloat(distributionResult.mean),
        median: parseFloat(distributionResult.median),
        quartiles: [
          parseFloat(distributionResult.q25),
          parseFloat(distributionResult.q50),
          parseFloat(distributionResult.q75),
        ],
      };
    }
  }

  setCache(cacheKey, stats);
  return stats;
}

/**
 * 获取所有列的统计信息
 */
export async function getAllColumnStats(
  tableName: string
): Promise<ColumnStats[]> {
  const columns = await getTableColumns(tableName);
  const statsPromises = columns.map((col) =>
    getColumnStats(tableName, col.column_name, col.data_type)
  );
  return await Promise.all(statsPromises);
}

/**
 * 清除缓存
 */
export function clearCache(): void {
  cache.clear();
}

/**
 * 清除特定表的缓存
 */
export function clearTableCache(tableName: string): void {
  const keysToDelete: string[] = [];
  for (const key of cache.keys()) {
    if (key.includes(tableName)) {
      keysToDelete.push(key);
    }
  }
  keysToDelete.forEach((key) => cache.delete(key));
}
