#!/usr/bin/env python3
"""
启动 API 服务的便捷脚本
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
project_root = Path(__file__).parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ 已加载环境变量: {env_path}")

# 确保项目根目录在 Python 路径中，以便 import api.xxx
sys.path.insert(0, str(project_root))

# 启动服务（必须用 api.app:app，因为 app.py 里有 from api.inference 等）
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 启动 API 服务，端口: {port}")
    uvicorn.run("api.app:app", host="0.0.0.0", port=port, reload=True)
