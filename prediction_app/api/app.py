"""
API Gateway + Inference Service

Provides model inference endpoints and model deployment management.
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

# Add project root to sys.path so we can import `api.xxx`.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables (optional)
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded env file: {env_path}")
else:
    print(f"Env file not found (optional): {env_path}")

from api.inference import InferenceService
from api.deployment import ModelDeploymentManager
from api.datasets import router as datasets_router
from api.model_assets import router as model_assets_router
from api.model_assets import scan_model_assets

app = FastAPI(
    title="Prediction API",
    description="Model inference and deployment management API",
    version="1.0.0"
)

# Configure CORS (restrict origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
inference_service = InferenceService()
deployment_manager = ModelDeploymentManager()

# Register routers
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
    """API root"""
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
    """Health check"""
    return {
        "status": "healthy",
        "service": "prediction-api"
    }


@app.post("/api/v1/predict")
async def predict(request: PredictionRequest):
    """
    Model inference endpoint.
    
    Args:
        request: input feature dict + optional model name
        
    Returns:
        prediction result
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
    """List deployed models"""
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
    """List models available for inference (builtin + training folder + deployed + assets)."""
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
    Deploy a model.
    
    Args:
        request: model path, name, and version
        
    Returns:
        deployment result
    """
    try:
        result = await deployment_manager.deploy(
            model_path=request.model_path,
            model_name=request.model_name,
            version=request.version
        )
        return {
            "success": True,
            "message": f"Model '{request.model_name}' deployed successfully",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/models/{model_name}")
async def remove_model(model_name: str):
    """Remove a deployed model"""
    try:
        result = deployment_manager.remove_model(model_name)
        return {
            "success": True,
            "message": f"Model '{model_name}' removed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Read port from env (default: 8000)
    port = int(os.getenv("PORT", 8000))
    print(f"Starting API server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
