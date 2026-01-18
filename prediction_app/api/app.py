"""
API Gateway + 推理服务
提供模型推理接口和模型部署管理
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Silence noisy joblib physical-core detection warnings in restricted environments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ 已加载环境变量: {env_path}")
else:
    print(f"⚠️  环境变量文件不存在: {env_path}")

from api.inference import InferenceService
from api.deployment import ModelDeploymentManager
from api.datasets import router as datasets_router
from api.model_assets import router as model_assets_router
from api.model_assets import scan_model_assets

app = FastAPI(
    title="Prediction API",
    description="模型推理和部署管理 API",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
inference_service = InferenceService()
deployment_manager = ModelDeploymentManager()

# 注册数据集管理路由
app.include_router(datasets_router)
app.include_router(model_assets_router)


# 请求模型
class PredictionRequest(BaseModel):
    data: Dict[str, Any]
    model_name: Optional[str] = None


class DeployModelRequest(BaseModel):
    model_path: str
    model_name: str
    version: Optional[str] = "1.0.0"


@app.get("/")
async def root():
    """API 根路径"""
    return {
        "name": "Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "models": "/api/v1/models",
            "models_available": "/api/v1/models/available",
            "model_assets": "/api/v1/model-assets",
            "deploy": "/api/v1/models/deploy",
            "datasets": "/api/v1/datasets",
            "sales_forecast": "/api/v1/datasets/{dataset_id}/sales-forecast",
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "prediction-api"
    }


@app.post("/api/v1/predict")
async def predict(request: PredictionRequest):
    """
    模型推理接口
    
    Args:
        request: 包含预测数据和可选的模型名称
        
    Returns:
        预测结果
    """
    try:
        result = await inference_service.predict(
            data=request.data,
            model_name=request.model_name
        )
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models")
async def list_models():
    """获取已部署的模型列表"""
    try:
        models = deployment_manager.list_models()
        return {
            "success": True,
            "models": models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/available")
async def list_available_models():
    """获取可用于推理的模型列表（内置 + 训练目录 + 已部署）"""
    try:
        models = [
            {
                "id": "baseline_moving_average",
                "name": "Baseline Moving Average",
                "version": "builtin",
                "source": "builtin",
                "description": "7-day moving average baseline (no ML model file required)",
            }
        ]

        # Deployed models
        for m in deployment_manager.list_models():
            name = m.get("name")
            version = m.get("version")
            if name and version:
                models.append(
                    {
                        "id": f"{name}:{version}",
                        "name": name,
                        "version": version,
                        "source": "deployed",
                        "status": m.get("status", "active"),
                        "deployed_at": m.get("deployed_at"),
                    }
                )

        # Repo-level model assets (models/{name}/{version})
        for m in scan_model_assets():
            models.append(
                {
                    "id": m.get("id"),
                    "name": m.get("name"),
                    "version": m.get("version"),
                    "source": "assets",
                    "framework": m.get("framework"),
                    "task": m.get("task"),
                    "target_metric": m.get("target_metric"),
                    "trained_at": m.get("trained_at"),
                    "metrics": m.get("metrics"),
                }
            )

        # Local training models (training/models/*_model.pkl)
        try:
            for p in sorted(inference_service.models_dir.glob("*_model.pkl")):
                model_name = p.stem.replace("_model", "")
                models.append(
                    {
                        "id": model_name,
                        "name": model_name,
                        "version": "local",
                        "source": "training",
                        "path": str(p),
                    }
                )
        except Exception:
            # Keep endpoint resilient even if the folder is missing.
            pass

        return {"success": True, "models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/models/deploy")
async def deploy_model(request: DeployModelRequest):
    """
    部署模型
    
    Args:
        request: 包含模型路径、名称和版本
        
    Returns:
        部署结果
    """
    try:
        result = await deployment_manager.deploy(
            model_path=request.model_path,
            model_name=request.model_name,
            version=request.version
        )
        return {
            "success": True,
            "message": f"模型 {request.model_name} 部署成功",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/models/{model_name}")
async def remove_model(model_name: str):
    """移除已部署的模型"""
    try:
        result = deployment_manager.remove_model(model_name)
        return {
            "success": True,
            "message": f"模型 {model_name} 已移除"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量读取端口，默认 8000
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 启动 API 服务，端口: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
