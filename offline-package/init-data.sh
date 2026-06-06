#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  Fina Demo 数据初始化脚本"
echo "========================================"

# 等待 Postgres 就绪
echo ""
echo "⏳ 等待 Postgres 就绪..."
PG_MAX_WAIT=60
PG_WAITED=0
while [ $PG_WAITED -lt $PG_MAX_WAIT ]; do
    if docker exec fina-postgres pg_isready -U fina -d fina_db >/dev/null 2>&1; then
        echo "✅ Postgres 已就绪"
        break
    fi
    sleep 2
    PG_WAITED=$((PG_WAITED + 2))
    echo "   已等待 ${PG_WAITED}s..."
done

if [ $PG_WAITED -ge $PG_MAX_WAIT ]; then
    echo "⚠️  Postgres 启动超时，退出"
    exit 1
fi

# 执行初始化 SQL
echo ""
echo "🗄️  初始化默认数据（租户 / 工作区 / 用户）..."

if [ ! -f "init-data.sql" ]; then
    echo "❌ 未找到 init-data.sql 文件"
    exit 1
fi

docker exec -i fina-postgres psql -U fina -d fina_db < init-data.sql || {
    echo "⚠️  数据初始化失败"
    exit 1
}

echo "✅ 默认数据初始化完成"
echo ""
echo "   默认账户："
echo "     租户: default"
echo "     工作区: default"
echo "     用户: admin@fina.ai"
echo "     密码: admin"

# 等待 Agent 就绪
echo ""
echo "⏳ 等待 Agent 服务就绪..."
AGENT_MAX_WAIT=90
AGENT_WAITED=0
while [ $AGENT_WAITED -lt $AGENT_MAX_WAIT ]; do
    if curl -sf http://localhost:5702/health &> /dev/null; then
        echo "✅ Agent 已就绪"
        break
    fi
    sleep 2
    AGENT_WAITED=$((AGENT_WAITED + 2))
    echo "   已等待 ${AGENT_WAITED}s..."
done

if [ $AGENT_WAITED -ge $AGENT_MAX_WAIT ]; then
    echo "⚠️  Agent 启动超时，跳过 Skill 初始化"
    exit 1
fi

# 等待鉴权服务就绪
echo ""
echo "🔧 初始化预置 Skills..."

SKILL_MAX_WAIT=60
SKILL_WAITED=0
TOKEN=""
while [ $SKILL_WAITED -lt $SKILL_MAX_WAIT ]; do
    LOGIN_RESP=$(curl -sf -X POST http://localhost:5702/api/auth/login \
      -H "Content-Type: application/json" \
      -d '{"email":"admin@fina.ai","password":"admin"}')
    
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
    exit 1
fi

# 等待沙盒服务就绪
SAND_MAX_WAIT=60
SAND_WAITED=0
while [ $SAND_WAITED -lt $SAND_MAX_WAIT ]; do
    if curl -sf -o /dev/null http://localhost:8080/; then
        echo "   ✅ 沙盒就绪"
        break
    fi
    sleep 2
    SAND_WAITED=$((SAND_WAITED + 2))
    echo "   已等待 ${SAND_WAITED}s..."
done

if [ $SAND_WAITED -ge $SAND_MAX_WAIT ]; then
    echo "   ⚠️  沙盒服务超时，跳过 Skill 初始化"
    exit 1
fi

# 创建 chart-markdown skill
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:5702/api/skills" \
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

echo ""
echo "========================================"
echo "  ✅ 数据初始化完成！"
echo "========================================"
