# 数据集管理功能规格说明

## 1. 功能概述

数据集管理功能提供对存储在 PostgreSQL 数据库中的数据集进行查看、浏览和统计分析的能力。该功能包括数据集列表展示、详情查看、数据预览和列统计信息展示。

### 1.1 核心功能
- **数据集列表**：展示所有可用的数据集，包括基本信息（名称、描述、类型、记录数、标签等）
- **数据集详情**：展示数据集的完整信息，包括基本信息、列定义、统计信息和数据预览
- **数据预览**：支持分页浏览数据集的实际数据记录
- **列统计**：展示每个列的统计信息（空值数量、唯一值数量、数值分布等）
- **时间范围**：自动识别并展示数据集的时间范围（如果存在时间列）

## 2. 前端功能需求

### 2.1 数据集列表页 (`/admin/assets/datasets`)

#### 2.1.1 页面布局
- 使用 Ant Design 的 `List` 和 `Table` 组件
- 表格支持分页（默认每页 10 条）
- 表格行可点击，点击后跳转到详情页

#### 2.1.2 显示字段
| 字段 | 说明 | 显示方式 |
|------|------|----------|
| 数据集名称 | 数据集的名称 | 粗体，蓝色 (#1D70B8) |
| 描述 | 数据集的描述信息 | 灰色小字，显示在名称下方 |
| 类型 | 数据集类型（如 sales, inventory 等） | 蓝色标签 |
| 记录数 | 数据集的总记录数 | 粗体，深蓝色 (#0F3460)，带千分位分隔符 |
| 标签 | 数据集的标签列表 | 多个标签，圆角样式 |
| 最后更新 | 数据集的最后更新时间 | 格式化日期时间 |
| 操作 | 查看按钮 | 链接按钮，蓝色 |

#### 2.1.3 交互行为
- 点击表格行或"查看"按钮，跳转到详情页：`/admin/assets/datasets/{id}`
- 支持表格排序和筛选（由 Refine 的 `useTable` hook 提供）

### 2.2 数据集详情页 (`/admin/assets/datasets/{id}`)

#### 2.2.1 页面结构
页面分为以下几个部分（从上到下）：
1. **数据集头部信息**：名称、描述、标签
2. **基本信息卡片**：记录数、列数、时间范围
3. **列定义与统计卡片**（可选）：每个列的详细统计信息
4. **数据预览卡片**：实际数据记录的表格展示

#### 2.2.2 数据集头部信息
- **名称**：大标题，深蓝色 (#0F3460)
- **描述**：灰色文本，显示在名称下方
- **标签**：多个标签，圆角样式，可换行

#### 2.2.3 基本信息卡片
使用 `Statistic` 组件展示：
- **记录数**：大号数字，蓝色 (#1D70B8)，带"条"后缀
- **列数**：大号数字，蓝色 (#1D70B8)，带"列"后缀
- **时间范围**：如果存在时间列，显示最小和最大日期
  - 格式：`YYYY-MM-DD 至 YYYY-MM-DD`
  - 使用日历图标

#### 2.2.4 列定义与统计卡片
- **显示条件**：只有当 `dataset.columns` 存在且长度 > 0 时才显示
- **加载状态**：如果 `isLoading` 为 true，显示加载动画和提示文字
- **内容**：使用 `Collapse` 组件，每个列一个面板
  - 默认展开前 3 个列
  - 面板标题：列名（粗体）+ 类型标签（蓝色）
  - 面板内容：列统计信息（如果存在）

##### 列统计信息包含：
- **空值数量**：该列中 NULL 值的数量
- **唯一值数量**：该列中唯一值的数量（如果可计算）
- **数值分布**（仅数值类型列）：
  - 最小值
  - 最大值
  - 平均值（保留 2 位小数）
  - 中位数（保留 2 位小数）
  - 分位数：25%、50%、75%（保留 2 位小数）

#### 2.2.5 数据预览卡片
- **标题**：显示"数据预览"
- **右上角**：显示总记录数（蓝色粗体）
- **内容**：数据表格

##### 表格特性
- **列定义**：
  - 优先使用 `dataset.columns` 中的列定义
  - 如果 `dataset.columns` 为空，从预览数据的第一条记录推断列
  - 列标题显示列名和类型（如果可用）
- **列格式化**：
  - **日期时间**：格式化为 `YYYY-MM-DD HH:mm:ss`
  - **数值**：添加千分位分隔符，浮点数保留 2 位小数
  - **布尔值**：显示为绿色/灰色标签（是/否）
  - **长文本**：超过 100 字符时截断，显示省略号和 tooltip
- **排序**：数值列和日期列支持排序
- **分页**：
  - 默认每页 20 条
  - 支持切换每页大小：10, 20, 50, 100
  - 支持快速跳转（当总记录数 > 1000 时）
  - 显示分页信息：`第 X-Y 条，共 Z 条`
- **滚动**：
  - 水平滚动：`x: "max-content"`
  - 垂直滚动：`y: "calc(100vh - 500px)"`
- **样式**：
  - 小号表格 (`size="small"`)
  - 带边框 (`bordered`)
  - 表头固定 (`sticky={{ offsetHeader: 0 }}`)
  - 行悬停高亮

#### 2.2.6 加载状态处理
- **初始化阶段**：如果 `queryResult` 不存在，显示"正在初始化..."
- **加载阶段**：如果 `isLoading` 为 true，显示加载动画
- **错误处理**：如果 `isError` 为 true，显示错误提示
- **数据不存在**：如果 `dataset` 为 null，显示警告提示
- **渐进式加载**：
  - 即使 `dataset` 还在加载，只要有 `id` 就可以先显示基本信息
  - 列定义部分可以异步加载，不阻塞页面
  - 预览数据优先显示，不依赖列定义

## 3. 后端 API 需求

### 3.1 API 端点

所有 API 端点位于 Python FastAPI 服务（默认端口 8000），路径前缀：`/api/v1/datasets`

#### 3.1.1 获取数据集列表
```
GET /api/v1/datasets
```

**响应格式**：
```json
{
  "success": true,
  "data": [
    {
      "id": "sales_data",
      "name": "销售数据集",
      "description": "包含历史销售记录的完整数据集",
      "table_name": "sales_data",
      "type": "sales",
      "row_count": 541909,
      "created_at": "2026-01-18T00:00:00Z",
      "updated_at": "2026-01-18T00:00:00Z",
      "tags": ["sales", "historical", "core"]
    }
  ]
}
```

**实现逻辑**：
1. 从 `prediction_app/config/datasets.json` 读取数据集配置
2. 对每个数据集，查询数据库获取 `row_count`
3. 返回合并后的数据

#### 3.1.2 获取数据集详情
```
GET /api/v1/datasets/{dataset_id}
```

**响应格式**：
```json
{
  "success": true,
  "data": {
    "id": "sales_data",
    "name": "销售数据集",
    "description": "包含历史销售记录的完整数据集",
    "table_name": "sales_data",
    "type": "sales",
    "row_count": 541909,
    "column_count": 15,
    "time_range": {
      "min": "2020-01-01T00:00:00Z",
      "max": "2023-12-31T23:59:59Z"
    },
    "columns": [
      {
        "name": "date",
        "type": "date",
        "stats": {
          "nullCount": 0,
          "uniqueCount": 1095,
          "distribution": null
        }
      },
      {
        "name": "amount",
        "type": "numeric",
        "stats": {
          "nullCount": 0,
          "uniqueCount": 45231,
          "distribution": {
            "min": 10.5,
            "max": 99999.99,
            "mean": 1250.45,
            "median": 850.20,
            "quartiles": [450.0, 850.2, 1800.5]
          }
        }
      }
    ],
    "created_at": "2026-01-18T00:00:00Z",
    "updated_at": "2026-01-18T00:00:00Z",
    "tags": ["sales", "historical", "core"]
  }
}
```

**实现逻辑**：
1. 从配置文件读取数据集元数据
2. 查询数据库获取：
   - `row_count`：表的总行数
   - `column_count`：表的列数
   - `time_range`：如果配置了 `time_column`，查询该列的最小和最大值
   - `columns`：所有列的信息和统计
3. 对每个列计算统计信息（空值数量、唯一值数量、数值分布等）

#### 3.1.3 获取数据预览
```
GET /api/v1/datasets/{dataset_id}/preview?page=1&pageSize=20
```

**查询参数**：
- `page`：页码（从 1 开始）
- `pageSize`：每页记录数

**响应格式**：
```json
{
  "success": true,
  "data": {
    "records": [
      {
        "id": 1,
        "date": "2023-01-01",
        "amount": 1250.50,
        "product": "Product A",
        ...
      },
      ...
    ],
    "total": 541909,
    "page": 1,
    "pageSize": 20,
    "totalPages": 27096
  }
}
```

**实现逻辑**：
1. 使用 `LIMIT` 和 `OFFSET` 进行分页查询
2. 查询总记录数
3. 计算总页数
4. 返回记录和分页信息

**注意事项**：
- 每次查询使用新的数据库连接，避免事务冲突
- 查询完成后关闭连接

### 3.2 数据库查询

#### 3.2.1 获取表行数
```sql
SELECT COUNT(*) FROM {table_name}
```

#### 3.2.2 获取表列信息
```sql
SELECT 
  column_name,
  data_type
FROM information_schema.columns
WHERE table_name = '{table_name}'
ORDER BY ordinal_position
```

#### 3.2.3 获取时间范围
```sql
SELECT 
  MIN({time_column}) as min,
  MAX({time_column}) as max
FROM {table_name}
```

#### 3.2.4 获取列统计信息
```sql
-- 空值数量
SELECT COUNT(*) - COUNT({column_name}) as null_count
FROM {table_name}

-- 唯一值数量
SELECT COUNT(DISTINCT {column_name}) as unique_count
FROM {table_name}

-- 数值分布（仅数值类型）
SELECT 
  MIN({column_name}) as min,
  MAX({column_name}) as max,
  AVG({column_name}) as mean,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column_name}) as median,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column_name}) as q25,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column_name}) as q75
FROM {table_name}
WHERE {column_name} IS NOT NULL
```

### 3.3 配置文件

数据集配置存储在 `prediction_app/config/datasets.json`：

```json
{
  "datasets": [
    {
      "id": "sales_data",
      "name": "销售数据集",
      "description": "包含历史销售记录的完整数据集",
      "table_name": "sales_data",
      "type": "sales",
      "created_at": "2026-01-18T00:00:00Z",
      "updated_at": "2026-01-18T00:00:00Z",
      "tags": ["sales", "historical", "core"],
      "time_column": "date"
    }
  ]
}
```

**字段说明**：
- `id`：数据集唯一标识符
- `name`：数据集显示名称
- `description`：数据集描述
- `table_name`：数据库中的表名
- `type`：数据集类型
- `created_at`：创建时间（ISO 8601 格式）
- `updated_at`：更新时间（ISO 8601 格式）
- `tags`：标签数组
- `time_column`：时间列名称（可选，用于计算时间范围）

## 4. 数据结构

### 4.1 TypeScript 类型定义

```typescript
// 数据集基本信息
interface Dataset {
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

// 列统计信息
interface ColumnStats {
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

// 列信息
interface ColumnInfo {
  name: string;
  type: string;
  stats: ColumnStats | null;
}

// 时间范围
interface TimeRange {
  min: string | null;
  max: string | null;
}

// 数据集详情（扩展 Dataset）
interface DatasetDetail extends Dataset {
  column_count: number;
  time_range: TimeRange | null;
  columns: ColumnInfo[];
}

// 数据预览
interface DatasetPreview {
  records: Record<string, any>[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

// API 响应
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  total?: number;
}
```

## 5. 用户交互流程

### 5.1 查看数据集列表
1. 用户访问 `/admin/assets/datasets`
2. 前端调用 `GET /api/v1/datasets`
3. 显示数据集列表表格
4. 用户可以：
   - 浏览所有数据集
   - 点击表格行或"查看"按钮进入详情页

### 5.2 查看数据集详情
1. 用户点击数据集，跳转到 `/admin/assets/datasets/{id}`
2. 前端同时发起两个请求：
   - `GET /api/v1/datasets/{id}` - 获取详情
   - `GET /api/v1/datasets/{id}/preview?page=1&pageSize=20` - 获取预览数据
3. 页面渐进式加载：
   - 先显示数据集名称和基本信息（即使详情还在加载）
   - 然后显示列定义和统计（如果已加载）
   - 最后显示数据预览表格
4. 用户可以：
   - 查看数据集基本信息
   - 展开/折叠列统计面板
   - 浏览数据预览（分页、排序）
   - 切换每页显示数量

## 6. 技术实现细节

### 6.1 前端技术栈
- **框架**：React + TypeScript
- **UI 库**：Ant Design
- **数据获取**：Refine (`useShow`, `useTable`)
- **路由**：React Router
- **状态管理**：React Hooks (`useState`, `useEffect`)

### 6.2 后端技术栈
- **框架**：FastAPI (Python)
- **数据库**：PostgreSQL
- **数据库驱动**：psycopg2
- **配置管理**：JSON 文件 + dotenv

### 6.3 关键实现点

#### 6.3.1 渐进式加载
- 使用 `displayDataset` 作为后备，即使 `dataset` 未加载也能显示基本信息
- 列定义部分条件渲染，加载中显示加载状态
- 预览数据优先显示，不依赖列定义

#### 6.3.2 数据库连接管理
- 每次查询使用新的数据库连接
- 查询完成后立即关闭连接
- 避免事务冲突和连接泄漏

#### 6.3.3 列推断
- 如果 `dataset.columns` 为空，从预览数据的第一条记录推断列
- 根据值类型自动判断列类型（数值、日期、布尔、文本）

#### 6.3.4 数据格式化
- 日期时间：使用 `toLocaleString` 格式化
- 数值：添加千分位分隔符
- 布尔值：显示为标签
- 长文本：截断并显示 tooltip

## 7. 当前存在的问题和优化建议

### 7.1 已知问题
1. **加载状态**：页面可能长时间显示"正在初始化"，需要优化加载逻辑
2. **列统计性能**：对于大表，计算列统计信息可能较慢，建议：
   - 添加缓存机制
   - 异步计算统计信息
   - 提供"跳过统计"选项
3. **预览数据性能**：对于大表，分页查询可能较慢，建议：
   - 添加索引
   - 限制最大分页大小
   - 提供数据采样预览选项

### 7.2 优化建议

#### 7.2.1 性能优化
- **缓存机制**：对数据集列表、详情和统计信息进行缓存
- **异步加载**：列统计信息异步加载，不阻塞页面
- **虚拟滚动**：对于大量数据，使用虚拟滚动
- **数据采样**：提供数据采样预览选项（如随机采样 1000 条）

#### 7.2.2 功能增强
- **搜索和筛选**：在列表页添加搜索和筛选功能
- **导出功能**：支持导出数据集为 CSV/Excel
- **列筛选**：在预览表格中支持列筛选和排序
- **数据可视化**：为数值列提供简单的图表展示（直方图、箱线图等）
- **数据质量报告**：生成数据质量报告（缺失值、异常值等）

#### 7.2.3 用户体验优化
- **加载进度**：显示详细的加载进度
- **错误提示**：提供更友好的错误提示和重试机制
- **空状态**：优化空状态显示
- **响应式设计**：优化移动端显示

#### 7.2.4 代码优化
- **类型安全**：完善 TypeScript 类型定义
- **错误处理**：统一错误处理机制
- **代码复用**：提取公共组件和工具函数
- **测试**：添加单元测试和集成测试

## 8. 文件结构

```
fina_demo/
├── ai_web/
│   └── src/
│       ├── pages/
│       │   └── assets/
│       │       └── datasets/
│       │           ├── index.tsx          # 列表页
│       │           └── detail.tsx        # 详情页
│       ├── types/
│       │   └── dataset.ts                 # 类型定义
│       └── authProvider.ts                # API 调用逻辑
├── prediction_app/
│   ├── api/
│   │   └── datasets.py                   # API 端点实现
│   └── config/
│       └── datasets.json                     # 数据集配置
└── DATASET_MANAGEMENT_SPEC.md            # 本文档
```

## 9. API 调用示例

### 9.1 前端调用示例

```typescript
// 获取数据集列表
const { data } = await fetch('http://localhost:8000/api/v1/datasets');

// 获取数据集详情
const { data } = await fetch('http://localhost:8000/api/v1/datasets/sales_data');

// 获取数据预览
const { data } = await fetch(
  'http://localhost:8000/api/v1/datasets/sales_data/preview?page=1&pageSize=20'
);
```

### 9.2 使用 Refine Hooks

```typescript
// 列表页
const { tableProps } = useTable<Dataset>({
  resource: "datasets",
  pagination: { pageSize: 10 },
});

// 详情页
const { queryResult } = useShow<DatasetDetail>({
  resource: "datasets",
  id: id,
});
```

## 10. 环境配置

### 10.1 前端环境变量
```env
VITE_API_URL=http://localhost:3000
VITE_PYTHON_API_URL=http://localhost:8000
```

### 10.2 后端环境变量
```env
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

## 11. 部署说明

### 11.1 前端部署
- 构建命令：`pnpm build`
- 输出目录：`ai_web/dist`
- 使用 Nginx 提供静态文件服务

### 11.2 后端部署
- 启动命令：`python start_api.py`
- 默认端口：8000
- 使用 uvicorn 作为 ASGI 服务器

## 12. 测试建议

### 12.1 功能测试
- [ ] 数据集列表正常显示
- [ ] 数据集详情正常显示
- [ ] 数据预览正常显示和分页
- [ ] 列统计信息正确计算
- [ ] 时间范围正确识别
- [ ] 加载状态正确处理
- [ ] 错误状态正确处理

### 12.2 性能测试
- [ ] 大数据集（> 100 万行）的加载性能
- [ ] 多列数据集（> 50 列）的统计计算性能
- [ ] 并发请求的处理能力

### 12.3 兼容性测试
- [ ] 不同浏览器的兼容性
- [ ] 不同屏幕尺寸的响应式显示
- [ ] 不同数据类型的正确显示

---

**文档版本**：v1.0  
**最后更新**：2026-01-18  
**维护者**：开发团队
