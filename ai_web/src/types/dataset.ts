/**
 * Dataset 相关 TypeScript 类型定义
 */

export interface Dataset {
  id: string;
  name: string;
  description: string;
  table_name: string;
  type: string;
  row_count: number;
  created_at: string;
  updated_at: string;
  tags: string[];
}

export interface ColumnStats {
  columnName: string;
  dataType: string;
  nullCount: number;
  uniqueCount?: number;
  distribution?: {
    min: number;
    max: number;
    mean: number;
    median: number;
    quartiles: [number, number, number];
  };
}

export interface ColumnInfo {
  name: string;
  type: string;
  stats: ColumnStats | null;
}

export interface TimeRange {
  min: string | null;
  max: string | null;
}

export interface DatasetDetail extends Dataset {
  column_count: number;
  time_range: TimeRange | null;
  columns: ColumnInfo[];
}

export interface DatasetPreview {
  records: Record<string, any>[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  total?: number;
}
