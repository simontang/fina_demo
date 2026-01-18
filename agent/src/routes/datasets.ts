import { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import * as fs from "fs";
import * as path from "path";
import { query } from "../services/dbConnection";
import {
  getTableRowCount,
  getTimeRange,
  getAllColumnStats,
  getTableColumns,
} from "../services/datasetStats";

interface DatasetConfig {
  id: string;
  name: string;
  description: string;
  table_name: string;
  type: string;
  created_at: string;
  updated_at: string;
  tags: string[];
  time_column?: string;
}

interface DatasetsConfig {
  datasets: DatasetConfig[];
}

/**
 * 读取数据集配置文件
 */
function loadDatasetsConfig(): DatasetsConfig {
  // 配置文件在 prediction_app/config/datasets.json
  // 尝试多个可能的路径
  const possiblePaths = [
    path.join(__dirname, "../../../prediction_app/config/datasets.json"),
    path.join(process.cwd(), "prediction_app/config/datasets.json"),
    path.join(process.cwd(), "../prediction_app/config/datasets.json"),
  ];

  for (const configPath of possiblePaths) {
    try {
      if (fs.existsSync(configPath)) {
        const content = fs.readFileSync(configPath, "utf-8");
        console.log(`✅ Loaded datasets config from: ${configPath}`);
        return JSON.parse(content);
      }
    } catch (error) {
      console.error(`Failed to load datasets config from ${configPath}:`, error);
    }
  }

  console.error("⚠️ Could not find datasets.json config file");
  return { datasets: [] };
}

/**
 * 获取数据集列表
 */
async function getDatasetsList(
  request: FastifyRequest,
  reply: FastifyReply
) {
  try {
    const config = loadDatasetsConfig();
    const datasets = await Promise.all(
      config.datasets.map(async (dataset) => {
        try {
          const rowCount = await getTableRowCount(dataset.table_name);
          return {
            id: dataset.id,
            name: dataset.name,
            description: dataset.description,
            table_name: dataset.table_name,
            type: dataset.type,
            row_count: rowCount,
            created_at: dataset.created_at,
            updated_at: dataset.updated_at,
            tags: dataset.tags,
          };
        } catch (error) {
          console.error(
            `Error getting stats for dataset ${dataset.id}:`,
            error
          );
          return {
            id: dataset.id,
            name: dataset.name,
            description: dataset.description,
            table_name: dataset.table_name,
            type: dataset.type,
            row_count: 0,
            created_at: dataset.created_at,
            updated_at: dataset.updated_at,
            tags: dataset.tags,
          };
        }
      })
    );

    return {
      success: true,
      data: datasets,
      total: datasets.length,
    };
  } catch (error: any) {
    reply.code(500);
    return {
      success: false,
      error: error.message || "Failed to fetch datasets",
    };
  }
}

/**
 * 获取数据集详情
 */
async function getDatasetDetail(
  request: FastifyRequest<{ Params: { id: string } }>,
  reply: FastifyReply
) {
  try {
    const { id } = request.params;
    const config = loadDatasetsConfig();
    const dataset = config.datasets.find((d) => d.id === id);

    if (!dataset) {
      reply.code(404);
      return {
        success: false,
        error: "Dataset not found",
      };
    }

    // 获取表的基本信息
    const rowCount = await getTableRowCount(dataset.table_name);
    const columns = await getTableColumns(dataset.table_name);

    // 获取时间范围
    let timeRange = null;
    if (dataset.time_column) {
      timeRange = await getTimeRange(dataset.table_name, dataset.time_column);
    }

    // 获取所有列的统计信息
    const columnStats = await getAllColumnStats(dataset.table_name);

    return {
      success: true,
      data: {
        id: dataset.id,
        name: dataset.name,
        description: dataset.description,
        table_name: dataset.table_name,
        type: dataset.type,
        row_count: rowCount,
        column_count: columns.length,
        created_at: dataset.created_at,
        updated_at: dataset.updated_at,
        tags: dataset.tags,
        time_range: timeRange,
        columns: columns.map((col) => {
          const stats = columnStats.find((s) => s.columnName === col.column_name);
          return {
            name: col.column_name,
            type: col.data_type,
            stats: stats || null,
          };
        }),
      },
    };
  } catch (error: any) {
    reply.code(500);
    return {
      success: false,
      error: error.message || "Failed to fetch dataset detail",
    };
  }
}

/**
 * 预览数据集数据（分页）
 */
async function previewDataset(
  request: FastifyRequest<{
    Params: { id: string };
    Querystring: { page?: string; pageSize?: string };
  }>,
  reply: FastifyReply
) {
  try {
    const { id } = request.params;
    const page = parseInt(request.query.page || "1");
    const pageSize = parseInt(request.query.pageSize || "20");

    const config = loadDatasetsConfig();
    const dataset = config.datasets.find((d) => d.id === id);

    if (!dataset) {
      reply.code(404);
      return {
        success: false,
        error: "Dataset not found",
      };
    }

    const offset = (page - 1) * pageSize;

    // 获取总记录数
    const rowCount = await getTableRowCount(dataset.table_name);

    // 获取分页数据
    const sql = `SELECT * FROM ${dataset.table_name} LIMIT $1 OFFSET $2`;
    const rows = await query(sql, [pageSize, offset]);

    return {
      success: true,
      data: {
        records: rows,
        total: rowCount,
        page,
        pageSize,
        totalPages: Math.ceil(rowCount / pageSize),
      },
    };
  } catch (error: any) {
    reply.code(500);
    return {
      success: false,
      error: error.message || "Failed to preview dataset",
    };
  }
}

/**
 * 注册数据集路由
 */
export function registerDatasetRoutes(app: FastifyInstance): void {
  // 获取数据集列表
  app.get("/datasets", getDatasetsList);

  // 获取数据集详情
  app.get("/datasets/:id", getDatasetDetail);

  // 预览数据集数据
  app.get("/datasets/:id/preview", previewDataset);
}
