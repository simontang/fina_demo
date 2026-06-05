#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES_DIR="$SCRIPT_DIR/images"

echo "========================================"
echo "  Fina Demo 离线镜像加载工具"
echo "========================================"

if ! command -v docker &> /dev/null; then
    echo "❌ 错误：未检测到 Docker，请先安装 Docker"
    exit 1
fi

echo ""
echo "正在加载离线镜像..."
echo ""

for tar_file in "$IMAGES_DIR"/*.tar; do
    if [ -f "$tar_file" ]; then
        filename=$(basename "$tar_file")
        echo "📦 加载: $filename"
        docker load -i "$tar_file"
        echo "✅ 完成"
        echo ""
    fi
done

echo "========================================"
echo "  所有镜像加载完成！"
echo "========================================"
echo ""
echo "接下来可以运行: ./start.sh"
