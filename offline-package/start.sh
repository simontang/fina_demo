#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  Fina Demo 离线版启动脚本"
echo "========================================"

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 错误：未检测到 Docker，请先安装 Docker"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ 错误：未检测到 docker compose 插件，请先安装"
    exit 1
fi

# 检查镜像是否已加载
required_images=(
    "fina-offline/agent:latest"
    "fina-offline/ai-web:latest"
)

missing_images=()
for img in "${required_images[@]}"; do
    if ! docker image inspect "$img" &> /dev/null; then
        missing_images+=("$img")
    fi
done

if [ ${#missing_images[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  检测到以下镜像未加载："
    for img in "${missing_images[@]}"; do
        echo "   - $img"
    done
    echo ""
    echo "请先运行: ./load-images.sh"
    exit 1
fi

# 创建必要目录
mkdir -p uploads configs/fina logs

# 复制默认配置文件（如果不存在）
if [ ! -f "configs/fina/config.json" ]; then
    echo "📝 创建默认前端配置..."
    cat > configs/fina/config.json << 'EOF'
{
  "appName": "AlphaFina",
  "logoFilename": "logo.png",
  "faviconFilename": "favicon.ico"
}
EOF
fi

# ============================================
# 交互式配置 LLM 参数
# ============================================

echo ""
echo "🔧 配置大模型参数"
echo ""

# 读取当前 .env 中的值（如果存在）
CURRENT_LLM_URL=""
CURRENT_API_KEY=""
if [ -f ".env" ]; then
    CURRENT_LLM_URL=$(grep "^LLM_BASE_URL=" .env | cut -d '=' -f2- | tr -d '"' || true)
    CURRENT_API_KEY=$(grep "^API_KEY3=" .env | cut -d '=' -f2- | tr -d '"' || true)
fi

# LLM Base URL
while true; do
    if [ -n "$CURRENT_LLM_URL" ]; then
        read -rp "   LLM Base URL [当前: $CURRENT_LLM_URL]: " input_url
        NEW_LLM_URL="${input_url:-$CURRENT_LLM_URL}"
    else
        read -rp "   LLM Base URL: " NEW_LLM_URL
    fi

    if [ -n "$NEW_LLM_URL" ]; then
        break
    fi
    echo "   ❌ LLM Base URL 不能为空，请重新输入"
done

# API Key
while true; do
    if [ -n "$CURRENT_API_KEY" ]; then
        read -rsp "   API Key [当前已设置，按回车保留，或输入新值]: " input_key
        echo ""
        NEW_API_KEY="${input_key:-$CURRENT_API_KEY}"
    else
        read -rsp "   API Key: " NEW_API_KEY
        echo ""
    fi

    if [ -n "$NEW_API_KEY" ]; then
        break
    fi
    echo "   ❌ API Key 不能为空，请重新输入"
done

# 写入 .env 文件
echo ""
echo "📝 写入 .env 文件..."

cat > .env << EOF
LLM_BASE_URL=$NEW_LLM_URL
API_KEY3=$NEW_API_KEY
EOF

echo "✅ 配置已写入 .env"

# 启动服务
echo ""
echo "🚀 启动服务..."
echo ""

docker compose -f docker-compose.offline.yml --env-file .env up -d

# 等待 Agent 就绪
echo ""
echo "⏳ 等待 Agent 服务就绪..."
MAX_WAIT=90
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf http://localhost:5702/health &> /dev/null; then
        echo "✅ Agent 服务已就绪"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo "   已等待 ${WAITED}s..."
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "⚠️  Agent 服务启动超时"
fi

echo ""
echo "========================================"
echo "  ✅ 服务启动成功！"
echo "========================================"
echo ""
echo "访问地址："
echo "  🌐 Web 界面:    http://localhost:5701"
echo "  🤖 Agent API:   http://localhost:5702"
echo ""
echo "查看日志："
echo "  docker compose -f docker-compose.offline.yml logs -f"
echo ""
echo "停止服务："
echo "  docker compose -f docker-compose.offline.yml down"
echo ""
