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
mkdir -p uploads configs/fina

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

echo ""
echo "📋 使用 .env.offline 中的配置启动服务"
echo "   （如需修改配置，请直接编辑 .env.offline 文件）"

# 启动服务
echo ""
echo "🚀 启动服务..."
echo ""

if ! docker compose version &> /dev/null; then
    echo "❌ 错误：未检测到 docker compose 插件，请先安装"
    echo "   Ubuntu/Debian: apt install docker-compose-plugin"
    exit 1
fi

docker compose -f docker-compose.offline.yml up -d

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


if ! docker compose version &> /dev/null; then
    echo "❌ 错误：未检测到 docker compose 插件，请先安装"
    echo "   Ubuntu/Debian: apt install docker-compose-plugin"
    exit 1
fi

docker compose -f docker-compose.offline.yml up -d

# ============================================
# 等待 Agent 就绪
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
    echo "⚠️  Agent 服务启动超时"
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
{"name":"chart-markdown","description":"Selects an appropriate chart type (bar, line, pie, scatter, heatmap, funnel, gauge, radar) from data analysis or query results and generates a chart markdown block containing JSON for external chart libraries. Use when the user asks to visualize query results, analysis data, or to create bar charts, line charts, pie charts, scatter plots, or other chart types from tabular or aggregated data.","metadata":{"category":"global"},"content":"# Chart Markdown Block\n\n## Board-pack embedding mode\n\nWhen invoked from `final-board-pack-structured-writer`, generate charts in section context:\n\n- Output a chart block immediately after the related section conclusion.\n- Include a bilingual chart title when board pack is bilingual.\n- Prefer concise board-ready visuals (trend, comparison, contribution, mix, risk status).\n- Return one chart block per analytic point; do not batch unrelated charts into one block.\n- If required numeric fields are missing, return `chart_status: skipped` and list missing fields.\n\n## Chart type selection\n\nChoose by data shape and question:\n\n- **Bar** (bar): Compare categories or time periods. Use category xAxis, value yAxis. Multiple series for grouped or stacked bars.\n- **Line** (line): Show trends over time. Use category or time xAxis, value yAxis. Multiple series for multiple metrics.\n- **Pie** (pie): Show composition or share. No axes. Data: `[{value: number, name: string}, ...]`. Use `radius: [\"40%\", \"70%\"]` for donut.\n- **Scatter** (scatter): Correlation or distribution. Use value xAxis and value yAxis. Data: `[[x, y], [x, y], ...]`.\n- **Heatmap** (heatmap): Two dimensions + value. Category xAxis and yAxis. Data: `[[xIndex, yIndex, value], ...]`.\n- **Funnel** (funnel): Sequential stages or conversion. Data: `[{value, name}, ...]`. Use `sort: \"ascending\"` or `\"descending\"`.\n- **Gauge** (gauge): Single KPI or progress. One series with `data: [{value, name}]`.\n- **Radar** (radar): Multi-dimensional comparison. Define `radar.indicator` and series with `value` arrays.\n\n## Output format\n\nEmit a single fenced code block with language **chart**. Body is one JSON object:\n\n- **table**: Array of rows (original or summary data) for reference.\n- **echarts**: ECharts option object (title, tooltip, legend, xAxis, yAxis, series, etc.).\n\n```chart\n{\n  \"table\": [[\"Category\", \"Value\"], [\"A\", 10], [\"B\", 20]],\n  \"echarts\": { ... }\n}\n```\n\n## Schema requirements\n\n- **title**: `{\"text\": \"Clear chart title\"}`.\n- **tooltip**: Use `trigger: \"axis\"` for bar/line; `trigger: \"item\"` for pie/scatter/funnel.\n- **xAxis / yAxis**: Omit for pie/funnel/gauge. Use `type: \"category\"` or `\"time\"` or `\"value\"`; provide `data` when type is category.\n- **series**: One or more items. Each has `type`, `name`, `data`. Pie data: `[{value, name}]`. Scatter data: `[[x,y], ...]`. Bar/line: array of values or category-mapped values.\n- **legend**: Include when there are multiple series.\n\n## Best practices\n\n- Use clear, business-facing titles and axis labels (not raw field names).\n- Format numbers (percent, currency, thousands) in tooltip and axis labels.\n- For time series, use `xAxis.type: \"time\"` and consistent date format.\n- Show important values on the chart with `series.label` when useful.\n\n\n## Examples\n\n### Bar chart (category comparison)\n\n```chart\n{\n  \"table\": [[\"Product\", \"Sales\"], [\"Product A\", 320], [\"Product B\", 280], [\"Product C\", 250]],\n  \"echarts\": {\n    \"title\": {\"text\": \"Top products by sales\"},\n    \"tooltip\": {\"trigger\": \"axis\"},\n    \"xAxis\": {\"type\": \"category\", \"data\": [\"Product A\", \"Product B\", \"Product C\"], \"name\": \"Product\"},\n    \"yAxis\": {\"type\": \"value\", \"name\": \"Sales\"},\n    \"series\": [{\"type\": \"bar\", \"data\": [320, 280, 250], \"label\": {\"show\": true, \"position\": \"top\"}}]\n  }\n}\n```\n\n### Pie chart (composition)\n\n```chart\n{\n  \"table\": [[\"Channel\", \"Count\"], [\"Online\", 450], [\"Offline\", 320], [\"Partner\", 180]],\n  \"echarts\": {\n    \"title\": {\"text\": \"Customer acquisition by channel\"},\n    \"tooltip\": {\"trigger\": \"item\", \"formatter\": \"{b}: {c} ({d}%)\"},\n    \"legend\": {\"orient\": \"vertical\", \"right\": \"10%\", \"top\": \"center\"},\n    \"series\": [{\n      \"type\": \"pie\",\n      \"radius\": [\"40%\", \"70%\"],\n      \"data\": [\n        {\"value\": 450, \"name\": \"Online\"},\n        {\"value\": 320, \"name\": \"Offline\"},\n        {\"value\": 180, \"name\": \"Partner\"}\n      ],\n      \"label\": {\"formatter\": \"{b}\\n{d}%\"}\n    }]\n  }\n}\n```"}
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
