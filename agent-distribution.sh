#!/bin/bash

# Agent 多副本分布报告 (38)
# 依赖 nginx 的 agent_dist 日志（log_format agent_dist，记录 $upstream_addr 与 $http_x_axiom_affinity）
# 日志文件: /var/log/nginx/agent_dist.log

set -euo pipefail

SSH_HOST="${SSH_HOST:-root@38.76.211.27}"

ssh "$SSH_HOST" 'bash -s' <<'REMOTE'
set -e
f=/var/log/nginx/agent_dist.log

if [ ! -f "$f" ]; then
    echo "日志不存在: $f （请先在 nginx agent location 配置 agent_dist 日志）"
    exit 1
fi

echo "================ Agent 多副本分布报告 ================"
echo "总请求: $(wc -l < "$f")"
echo "起始:   $(head -1 "$f" | cut -d] -f1 | sed 's/^[^[]*//')"
echo "最新:   $(tail -1 "$f" | cut -d] -f1 | sed 's/^[^[]*//')"
echo

echo "--- 按副本 (5702=agent-1 / 5712=agent-2) ---"
awk '{print $(NF-2)}' "$f" | sort | uniq -c | sort -rn
echo

echo "--- 端点 × 副本 (Top 15) ---"
awk '{e=$5; sub(/\?.*/,"",e); print e, $(NF-2)}' "$f" | sort | uniq -c | sort -rn | head -15
echo

echo "--- thread(affinity) -> 副本 (Top 20) ---"
awk '$(NF-1) ~ /aff=[0-9a-fA-F-]{6,}/ {print $(NF-1), $(NF-2)}' "$f" | sort | uniq -c | sort -rn | head -20
echo

echo "--- 状态码 ---"
awk '{print $7}' "$f" | sort | uniq -c | sort -rn
echo

echo "--- 客户端 IP (Top 10) ---"
awk '{print $1}' "$f" | sort | uniq -c | sort -rn | head -10
echo

echo "--- 最近 15 条 ---"
tail -15 "$f"
echo "====================================================="
REMOTE
