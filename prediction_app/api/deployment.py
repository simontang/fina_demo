"""
模型部署管理
负责模型的部署、版本管理和生命周期管理
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelDeploymentManager:
    """模型部署管理器"""
    
    def __init__(
        self,
        models_dir: str = None,
        deployed_dir: str = None
    ):
        """
        初始化部署管理器
        
        Args:
            models_dir: 训练模型目录，如果为 None 则使用默认路径
            deployed_dir: 已部署模型目录，如果为 None 则使用默认路径
        """
        project_root = Path(__file__).parent.parent
        
        if models_dir is None:
            models_dir = project_root / "training" / "models"
        if deployed_dir is None:
            deployed_dir = project_root / "api" / "deployed_models"
        
        self.models_dir = Path(models_dir)
        self.deployed_dir = Path(deployed_dir)
        self.deployed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.deployed_dir / "metadata.json"
        self.metadata: Dict = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """加载模型元数据"""
        if self.metadata_file.exists():
            import json
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """保存模型元数据"""
        import json
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    async def deploy(
        self,
        model_path: str,
        model_name: str,
        version: str = "1.0.0"
    ) -> Dict:
        """
        部署模型
        
        Args:
            model_path: 模型文件路径
            model_name: 模型名称
            version: 模型版本
            
        Returns:
            部署信息
        """
        source_path = Path(model_path)
        if not source_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 创建部署目录
        deploy_path = self.deployed_dir / model_name / version
        deploy_path.mkdir(parents=True, exist_ok=True)
        
        # 复制模型文件
        dest_file = deploy_path / f"{model_name}_v{version}.pkl"
        shutil.copy2(source_path, dest_file)
        
        # 更新元数据
        if model_name not in self.metadata:
            self.metadata[model_name] = {}
        
        self.metadata[model_name][version] = {
            "path": str(dest_file),
            "deployed_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # 设置当前版本
        self.metadata[model_name]["current_version"] = version
        
        self._save_metadata()
        
        logger.info(f"模型部署成功: {model_name} v{version}")
        
        return {
            "model_name": model_name,
            "version": version,
            "path": str(dest_file),
            "deployed_at": self.metadata[model_name][version]["deployed_at"]
        }
    
    def list_models(self) -> List[Dict]:
        """
        列出所有已部署的模型
        
        Returns:
            模型列表
        """
        models = []
        for model_name, info in self.metadata.items():
            if isinstance(info, dict) and "current_version" in info:
                current_version = info["current_version"]
                version_info = info.get(current_version, {})
                models.append({
                    "name": model_name,
                    "version": current_version,
                    "deployed_at": version_info.get("deployed_at"),
                    "status": version_info.get("status", "active")
                })
        return models
    
    def remove_model(self, model_name: str) -> bool:
        """
        移除模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否成功移除
        """
        if model_name not in self.metadata:
            raise ValueError(f"模型不存在: {model_name}")
        
        # 删除模型目录
        model_dir = self.deployed_dir / model_name
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # 从元数据中移除
        del self.metadata[model_name]
        self._save_metadata()
        
        logger.info(f"模型已移除: {model_name}")
        return True
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息
        """
        return self.metadata.get(model_name)
