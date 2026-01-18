"""
模型训练服务
用于训练预测模型，支持从 raw_data 目录读取数据
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.utils.data_loader import load_data
from shared.models.model_factory import create_model


def train_model(
    data_path: str,
    model_type: str = "default",
    output_dir: str = "training/models",
    epochs: int = 100,
    batch_size: int = 32,
    **kwargs
):
    """
    训练模型
    
    Args:
        data_path: 训练数据路径
        model_type: 模型类型
        output_dir: 模型输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        **kwargs: 其他训练参数
    """
    print(f"🚀 开始训练模型: {model_type}")
    print(f"📁 数据路径: {data_path}")
    print(f"💾 输出目录: {output_dir}")
    
    # 加载数据
    print("📊 加载训练数据...")
    train_data, val_data = load_data(data_path)
    
    # 创建模型
    print(f"🏗️  创建模型: {model_type}")
    model = create_model(model_type, **kwargs)
    
    # 训练模型
    print("🎯 开始训练...")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_type}_model.pkl")
    model.save(model_path)
    print(f"✅ 模型已保存到: {model_path}")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description="训练预测模型")
    parser.add_argument(
        "--data-path",
        type=str,
        default="../raw_data/sales_data.csv",
        help="训练数据路径"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="default",
        help="模型类型"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/models",
        help="模型输出目录"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批次大小"
    )
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        model_type=args.model_type,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
