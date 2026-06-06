#!/bin/bash
set -e

# ============================================
# Fina Demo 离线安装包构建脚本
# ============================================
# 用法: ./build-offline-package.sh
# 输出: ../fina-offline-package.zip
#
# 注意：所有镜像均从线上仓库拉取 linux/amd64 (x86) 版本，
#       确保对方 x86 机器可正常运行。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_DIR="$SCRIPT_DIR"
IMAGES_DIR="$PACKAGE_DIR/images"

VERSION="${VERSION:-$(date +%Y%m%d)}"
ZIP_NAME="fina-offline-package-${VERSION}.zip"

echo "========================================"
echo "  Fina Demo 离线安装包构建"
echo "  版本: $VERSION"
echo "  架构: linux/amd64 (x86)"
echo "========================================"

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 错误：未检测到 Docker"
    exit 1
fi

# 清理旧镜像文件
mkdir -p "$IMAGES_DIR"
rm -f "$IMAGES_DIR"/*.tar

echo ""
echo "📦 步骤 1/4: 拉取 ai_web 镜像 (x86)..."
docker pull --platform linux/amd64 \
  finai-cn-shanghai.cr.volces.com/default/fina-demo-ai-web:latest
docker tag \
  finai-cn-shanghai.cr.volces.com/default/fina-demo-ai-web:latest \
  fina-offline/ai-web:latest

echo ""
echo "📦 步骤 2/4: 拉取 agent 镜像 (x86)..."
docker pull --platform linux/amd64 \
  finai-cn-shanghai.cr.volces.com/default/fina-demo-agent:latest
docker tag \
  finai-cn-shanghai.cr.volces.com/default/fina-demo-agent:latest \
  fina-offline/agent:latest

echo ""
echo "📦 步骤 3/4: 拉取 Sandbox 镜像 (x86)..."
docker pull --platform linux/amd64 \
  enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox:latest
docker tag \
  enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox:latest \
  fina-offline/all-in-one-sandbox:latest

echo ""
echo "📦 步骤 4/4: 拉取 PostgreSQL 镜像 (x86)..."
docker pull --platform linux/amd64 postgres:15-alpine
docker tag postgres:15-alpine fina-offline/postgres:15-alpine

echo ""
echo "💾 导出镜像为 tar 文件..."

echo "   - ai_web.tar"
docker save fina-offline/ai-web:latest > "$IMAGES_DIR/ai_web.tar"

echo "   - agent.tar"
docker save fina-offline/agent:latest > "$IMAGES_DIR/agent.tar"

echo "   - all-in-one-sandbox.tar"
docker save fina-offline/all-in-one-sandbox:latest > "$IMAGES_DIR/all-in-one-sandbox.tar"

echo "   - postgres-15-alpine.tar"
docker save fina-offline/postgres:15-alpine > "$IMAGES_DIR/postgres-15-alpine.tar"

# 复制前端静态资源（logo, favicon）
echo ""
echo "📝 复制前端配置文件..."
mkdir -p "$PACKAGE_DIR/configs/fina"
cp "$PROJECT_ROOT/configs/fina/"* "$PACKAGE_DIR/configs/fina/" 2>/dev/null || {
    echo "   ⚠️  未找到项目 configs/fina/ 下的文件，使用默认配置"
}

# 生成压缩包
echo ""
echo "📦 打包为 zip..."
cd "$PROJECT_ROOT"
zip -r "$ZIP_NAME" offline-package/ \
  -x "offline-package/build-offline-package.sh" \
  -x "offline-package/.env" \
  -x "offline-package/.env.offline" \
  2>/dev/null || {
    echo "   使用 tar.gz 替代..."
    tar --exclude='offline-package/.env' \
      --exclude='offline-package/.env.offline' \
      -czf "${ZIP_NAME%.zip}.tar.gz" offline-package/
    ZIP_NAME="${ZIP_NAME%.zip}.tar.gz"
  }

# 显示结果
echo ""
echo "========================================"
echo "  ✅ 离线安装包构建完成！"
echo "========================================"
echo ""
echo "输出文件: $PROJECT_ROOT/$ZIP_NAME"
echo ""
echo "镜像大小："
du -sh "$IMAGES_DIR/"*.tar 2>/dev/null || true
echo ""
echo "总大小："
du -sh "$PROJECT_ROOT/$ZIP_NAME" 2>/dev/null || true
echo ""
echo "📖 使用方式："
echo "   1. 将 $ZIP_NAME 发送给目标机器"
echo "   2. 解压后进入 offline-package 目录"
echo "   3. 运行 ./load-images.sh 加载镜像"
echo "   4. 运行 ./start.sh 启动服务"
echo ""
