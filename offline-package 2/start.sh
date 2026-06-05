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
    echo "   下载地址: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ 错误：未检测到 docker-compose，请先安装"
    echo "   Docker Desktop 已自带 docker-compose"
    exit 1
fi

# 检查镜像是否已加载
required_images=(
    "fina-offline/postgres:15-alpine"
    "fina-offline/agent:latest"
    "fina-offline/ai-web:latest"
    "fina-offline/all-in-one-sandbox:latest"
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
mkdir -p uploads lattice_store configs/fina

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
# 交互式配置大模型参数
# ============================================

# 读取当前 .env.offline 中的值（如果存在）
CURRENT_LLM_URL=""
CURRENT_API_KEY=""
if [ -f ".env.offline" ]; then
    CURRENT_LLM_URL=$(grep "^LLM_BASE_URL=" .env.offline | cut -d '=' -f2- | tr -d '"' || true)
    CURRENT_API_KEY=$(grep "^API_KEY3=" .env.offline | cut -d '=' -f2- | tr -d '"' || true)
fi

# 使用环境变量或交互式输入
if [ -n "$LLM_BASE_URL" ]; then
    NEW_LLM_URL="$LLM_BASE_URL"
else
    echo ""
    echo "🔧 大模型配置（必填）"
    echo ""
    echo "   示例 LLM Base URL："
    echo "     • 火山方舟: https://ark.cn-beijing.volces.com/api/v3"
    echo "     • Moonshot: https://api.moonshot.cn/v1"
    echo "     • 深度求索: https://api.deepseek.com/v1"
    echo "     • OpenAI:   https://api.openai.com/v1"
    echo ""

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
fi

if [ -n "$API_KEY3" ]; then
    NEW_API_KEY="$API_KEY3"
else
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
fi

# 更新 .env.offline
if [ -n "$NEW_LLM_URL" ]; then
    if grep -q "^LLM_BASE_URL=" .env.offline 2>/dev/null; then
        sed -i.bak "s|^LLM_BASE_URL=.*|LLM_BASE_URL=$NEW_LLM_URL|" .env.offline && rm -f .env.offline.bak
    else
        echo "LLM_BASE_URL=$NEW_LLM_URL" >> .env.offline
    fi
fi

if [ -n "$NEW_API_KEY" ]; then
    if grep -q "^API_KEY3=" .env.offline 2>/dev/null; then
        sed -i.bak "s|^API_KEY3=.*|API_KEY3=$NEW_API_KEY|" .env.offline && rm -f .env.offline.bak
    else
        echo "API_KEY3=$NEW_API_KEY" >> .env.offline
    fi
fi

# 启动服务
echo ""
echo "🚀 启动服务..."
echo ""

if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

$COMPOSE_CMD -f docker-compose.offline.yml --env-file .env.offline up -d

# ============================================
# 等待 Agent 就绪并初始化默认数据
# ============================================

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
    echo "⚠️  Agent 服务启动超时，跳过数据初始化"
else
    # 检查是否已初始化过数据
    if [ ! -f ".init-done" ]; then
        echo ""
        echo "🗄️  初始化默认数据（租户 / 工作区 / 用户）..."

        # 等待 Postgres 就绪
        sleep 3

        # 执行初始化 SQL
        docker exec -i fina-postgres psql -U fina -d fina_db < init-data.sql 2>/dev/null || {
            echo "   ⚠️  数据初始化可能已存在，跳过"
        }

        # 创建标记文件
        touch .init-done
        echo "✅ 默认数据初始化完成"
        echo ""
        echo "   默认账户："
        echo "     租户: default"
        echo "     工作区: default"
        echo "     用户: admin@localhost"
        echo "     密码: admin"
    else
        echo "✅ 数据已初始化，跳过"
    fi
fi

echo ""
echo "========================================"
echo "  ✅ 服务启动成功！"
echo "========================================"
echo ""
echo "访问地址："
echo "  🌐 Web 界面:    http://localhost:5701"
echo "  🤖 Agent API:   http://localhost:5702"
echo "  📦 Sandbox:     http://localhost:8080"
echo "  🐘 PostgreSQL:  localhost:5432"
echo ""
echo "查看日志："
echo "  $COMPOSE_CMD -f docker-compose.offline.yml logs -f"
echo ""
echo "停止服务："
echo "  $COMPOSE_CMD -f docker-compose.offline.yml down"
echo ""
