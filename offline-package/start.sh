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

echo ""
echo "========================================"
echo "  ✅ 服务已启动！"
echo "========================================"
echo ""
echo "访问地址："
echo "  🌐 Web 界面:    http://localhost:5701"
echo "  🤖 Agent API:   http://localhost:5702"
echo "  📦 Sandbox:     http://localhost:8080"
echo "  🐘 PostgreSQL:  localhost:5432"
echo ""
echo "初始化数据（首次运行需要）："
echo "  ./init-data.sh"
echo ""
echo "查看日志："
echo "  docker compose -f docker-compose.offline.yml logs -f"
echo ""
echo "停止服务："
echo "  docker compose -f docker-compose.offline.yml down"
echo ""


docker compose -f docker-compose.offline.yml --env-file .env up -d

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
    echo ""
    echo "🗄️  初始化默认数据（租户 / 工作区 / 用户）..."

    # 等待 Postgres 就绪
    sleep 3

    # 执行初始化 SQL（ON CONFLICT 保证幂等）
    docker exec -i fina-postgres psql -U fina -d fina_db < init-data.sql || {
        echo "   ⚠️  数据初始化失败"
    }

    echo "✅ 默认数据初始化完成"
    echo ""
    echo "   默认账户："
    echo "     租户: default"
    echo "     工作区: default"
    echo "     用户: admin@localhost"
    echo "     密码: admin"
fi

# 初始化预置 Skills（等待鉴权服务就绪后注册）
echo ""
echo "🔧 初始化预置 Skills..."

SKILL_MAX_WAIT=60
SKILL_WAITED=0
while [ $SKILL_WAITED -lt $SKILL_MAX_WAIT ]; do
    LOGIN_RESP=$(curl -sf -X POST http://localhost:5702/api/auth/login \
      -H "Content-Type: application/json" \
      -d '{"email":"admin@localhost","password":"admin"}')
    TOKEN=$(echo "$LOGIN_RESP" | grep -o '"token":"[^"]*"' | head -1 | sed 's/"token":"//;s/"//')

    if [ -n "$TOKEN" ]; then
        echo "   ✅ 鉴权就绪"
        break
    fi
    sleep 2
    SKILL_WAITED=$((SKILL_WAITED + 2))
    echo "   已等待 ${SKILL_WAITED}s..."
done

if [ -z "$TOKEN" ]; then
    echo "   ⚠️  鉴权服务超时，跳过 Skill 初始化"
else
    # 等待沙盒服务就绪
    SAND_MAX_WAIT=60
    SAND_WAITED=0
    while [ $SAND_WAITED -lt $SAND_MAX_WAIT ]; do
        if curl -sf http://localhost:8080/health &> /dev/null; then
            echo "   ✅ 沙盒就绪"
            break
        fi
        sleep 2
        SAND_WAITED=$((SAND_WAITED + 2))
        echo "   已等待 ${SAND_WAITED}s..."
    done
    if [ $SAND_WAITED -ge $SAND_MAX_WAIT ]; then
        echo "   ⚠️  沙盒服务超时，跳过 Skill 初始化"
    else
    # 创建 chart-markdown skill
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:5702/api/skills/chart-markdown" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -H "x-tenant-id: default" \
      -H "x-workspace-id: default" \
      -H "x-project-id: default" \
      --data-binary @- << 'SKILLEOF'
{"name":"chart-markdown","description":"Selects an appropriate chart type (bar, line, pie, scatter, heatmap, funnel, gauge, radar) from data analysis or query results and generates a chart markdown block containing JSON for external chart libraries. Use when the user asks to visualize query results, analysis data, or to create bar charts, line charts, pie charts, scatter plots, or other chart types from tabular or aggregated data.","metadata":{"category":"global"},"content":"# Chart Markdown Block\n\n## Board-pack embedding mode\n\nWhen invoked from `final-board-pack-structured-writer`, generate charts in section context:\n\n- Output a chart block immediately after the related section conclusion.\n- Include a bilingual chart title when board pack is bilingual.\n- Prefer concise board-ready visuals (trend, comparison, contribution, mix, risk status).\n- Return one chart block per analytic point; do not batch unrelated charts into one block.\n- If required numeric fields are missing, return `chart_status: skipped` and list missing fields.\n\n## Chart type selection\n\nChoose by data shape and question:\n\n- **Bar** (bar): Compare categories or time periods. Use category xAxis, value yAxis. Multiple series for grouped or stacked bars.\n- **Line** (line): Show trends over time. Use category or time xAxis, value yAxis. Multiple series for multiple metrics.\n- **Pie** (pie): Show composition or share. No axes. Data: `[{value: number, name: string}, ...]`. Use `radius: [\\\"40%\\\", \\\"70%\\\"]` for donut.\n- **Scatter** (scatter): Correlation or distribution. Use value xAxis and value yAxis. Data: `[[x, y], [x, y], ...]`.\n- **Heatmap** (heatmap): Two dimensions + value. Category xAxis and yAxis. Data: `[[xIndex, yIndex, value], ...]`.\n- **Funnel** (funnel): Sequential stages or conversion. Data: `[{value, name}, ...]`. Use `sort: \\\"ascending\\\"` or `\\\"descending\\\"`.\n- **Gauge** (gauge): Single KPI or progress. One series with `data: [{value, name}]`.\n- **Radar** (radar): Mult...","tools":[],"tags":[],"active":true}
SKILLEOF
)
    echo "   [$HTTP_CODE] chart-markdown"
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
echo "  docker compose -f docker-compose.offline.yml logs -f"
echo ""
echo "停止服务："
echo "  docker compose -f docker-compose.offline.yml down"
echo ""
