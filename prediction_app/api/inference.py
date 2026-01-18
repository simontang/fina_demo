"""
推理服务
负责加载模型和执行预测
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.models.model_factory import load_model

logger = logging.getLogger(__name__)


class InferenceService:
    """推理服务类"""
    
    def __init__(self, models_dir: str = None):
        """
        初始化推理服务
        
        Args:
            models_dir: 模型文件目录，如果为 None 则使用默认路径
        """
        if models_dir is None:
            # 默认使用项目根目录下的 training/models
            project_root = Path(__file__).parent.parent
            models_dir = project_root / "training" / "models"
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Any] = {}
        self.default_model_name: Optional[str] = None
    
    def _load_model(self, model_name: str) -> Any:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            加载的模型对象
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # 查找模型文件
        model_path = self.models_dir / f"{model_name}_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        logger.info(f"加载模型: {model_path}")
        model = load_model(str(model_path))
        
        self.loaded_models[model_name] = model
        if self.default_model_name is None:
            self.default_model_name = model_name
        
        return model
    
    async def predict(
        self,
        data: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> Any:
        """
        执行预测
        
        Args:
            data: 输入数据
            model_name: 模型名称，如果为 None 则使用默认模型
            
        Returns:
            预测结果
        """
        # 确定使用的模型
        if model_name is None:
            if self.default_model_name is None:
                # 尝试加载第一个可用的模型
                model_files = list(self.models_dir.glob("*_model.pkl"))
                if not model_files:
                    raise ValueError("没有可用的模型，请先训练或部署模型")
                model_name = model_files[0].stem.replace("_model", "")
            else:
                model_name = self.default_model_name
        
        # 加载模型
        model = self._load_model(model_name)
        
        # 执行预测
        try:
            result = model.predict(data)
            return result
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise
    
    def get_loaded_models(self) -> list:
        """获取已加载的模型列表"""
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_name: str):
        """卸载模型以释放内存"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if self.default_model_name == model_name:
                self.default_model_name = None
            logger.info(f"已卸载模型: {model_name}")
