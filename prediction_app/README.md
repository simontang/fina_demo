# Prediction App

预测应用服务，提供模型训练、部署和推理功能。

## 项目结构

```
prediction_app/
├── training/          # 模型训练服务（独立）
│   ├── train.py       # 训练脚本
│   ├── models/        # 保存训练好的模型
│   └── requirements.txt
├── api/              # API Gateway + 推理服务（合并）
│   ├── app.py        # FastAPI 主应用
│   ├── inference.py  # 推理逻辑
│   ├── deployment.py # 模型部署管理
│   ├── deployed_models/  # 已部署的模型
│   └── requirements.txt
└── shared/           # 共享代码
    ├── models/       # 模型定义和工厂
    └── utils/        # 工具函数（数据加载等）
```

## 快速开始

### 0. 导入数据到数据库

如果需要将 `raw_data/sales_data.csv` 导入到数据库：

```bash
# 安装脚本依赖
cd prediction_app/scripts
pip install -r requirements.txt

# 运行导入脚本
python import_sales_data.py
```

脚本会自动：
- 创建 `sales_data` 表（如果不存在）
- 读取 CSV 文件并预处理数据
- 批量导入数据到数据库
- 显示导入统计信息

### 1. 安装依赖

#### 训练服务
```bash
cd training
pip install -r requirements.txt
```

#### API 服务
```bash
cd api
pip install -r requirements.txt
```

### 2. 训练模型

```bash
cd training
python train.py \
  --data-path ../../raw_data/sales_data.csv \
  --model-type default \
  --output-dir training/models \
  --epochs 100 \
  --batch-size 32
```

### 3. 启动 API 服务

```bash
cd api
python app.py
```

或者使用 uvicorn：

```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API 服务将在 `http://localhost:8000` 启动。

## API 接口

### 健康检查
```
GET /health
```

### 模型推理
```
POST /api/v1/predict
Content-Type: application/json

{
  "data": {
    "feature1": 1.0,
    "feature2": 2.0
  },
  "model_name": "default"  // 可选
}
```

### 部署模型
```
POST /api/v1/models/deploy
Content-Type: application/json

{
  "model_path": "training/models/default_model.pkl",
  "model_name": "default",
  "version": "1.0.0"
}
```

### 列出模型
```
GET /api/v1/models
```

### 获取可用模型列表（内置 + 本地训练 + 已部署）
```
GET /api/v1/models/available
```

### 模型资产管理（从仓库根目录 `models/` 扫描）
```
GET /api/v1/model-assets
GET /api/v1/model-assets/{model_name}/{version}
```

### 移除模型
```
DELETE /api/v1/models/{model_name}
```

### 数据集管理

#### 获取数据集列表
```
GET /api/v1/datasets
```

返回所有数据集的基本信息，包括名称、描述、记录数等。

#### 获取数据集详情
```
GET /api/v1/datasets/{dataset_id}
```

返回数据集的完整信息，包括列定义、统计信息、时间范围等。

#### 预览数据集数据
```
GET /api/v1/datasets/{dataset_id}/preview?page=1&pageSize=20
```

返回分页的数据预览，支持翻页查看实际数据。

### 销量预测（Sales Forecast）

基于数据集的历史销售数据，构建时间特征 + 滞后/滚动特征并进行推理预测。

```
POST /api/v1/datasets/{dataset_id}/sales-forecast
Content-Type: application/json

{
  "model_id": "baseline_moving_average",   // 或 {name}:{version}（已部署模型）
  "target_entity_id": "85123A",
  "forecast_horizon": 7,
  "context_window_days": 60,
  "sales_metric": "quantity",             // quantity | revenue
  "promotion_factor": 1.0,
  "holiday_country": "CN",
  "rounding": "round"                     // round | floor | none
}
```

## 环境变量配置

### 创建环境变量文件

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

### 环境变量说明

#### 数据库配置
- `DATABASE_URL`: PostgreSQL 连接字符串（Python 格式）
- `DATABASE_JDBC_URL`: PostgreSQL 连接字符串（JDBC 格式）
- `DB_HOST`: 数据库主机地址
- `DB_PORT`: 数据库端口（默认: 5432）
- `DB_NAME`: 数据库名称
- `DB_USER`: 数据库用户名
- `DB_PASSWORD`: 数据库密码

#### API 服务配置
- `PORT`: API 服务端口（默认: 8000）

#### 模型路径配置
- `MODELS_DIR`: 训练模型存储目录（默认: training/models）
- `DEPLOYED_MODELS_DIR`: 已部署模型目录（默认: api/deployed_models）

### 训练服务
- 可通过命令行参数配置，见 `python train.py --help`

## 开发说明

### 添加新模型类型

1. 在 `shared/models/model_factory.py` 中创建新的模型类
2. 在 `create_model` 函数中注册新模型类型
3. 实现 `fit` 和 `predict` 方法

### 数据格式

训练数据应为 CSV 格式，最后一列为目标变量。

## 架构说明

- **训练服务独立**：训练任务通常需要大量资源，独立运行避免影响推理服务
- **推理 + Gateway 合并**：减少网络开销，简化部署，提高响应速度
- **模型部署管理**：通过 API 管理模型版本和生命周期

## 设计原则

### Python 作为主要后端服务语言

**设计决策**：所有数据管理和预测相关的 API 服务使用 Python 实现，代码位于 `prediction_app/api` 目录下。

**原因**：
1. **模型集成**：Python 是机器学习和数据科学领域的标准语言，便于直接调用训练好的模型进行预测和推理
2. **数据科学生态**：丰富的 Python 库（pandas, numpy, scikit-learn 等）便于数据处理和统计分析
3. **统一技术栈**：训练、推理、数据管理使用同一语言，降低技术栈复杂度
4. **开发效率**：Python 的简洁语法和丰富的库可以快速实现数据分析和统计功能

**适用范围**：
- 数据集管理 API（`/api/v1/datasets`）
- 模型推理 API（`/api/v1/predict`）
- 模型部署管理 API（`/api/v1/models`）
- 其他与数据分析和预测相关的服务

**例外情况**：
- Agent 服务（`agent/`）使用 TypeScript/Node.js，因为需要与 Lattice 框架集成
- 文件上传等通用服务可以保留在 Agent 服务中
