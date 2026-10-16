#!/bin/bash

# 部署 38 (ada.alphafina.cn) 的 nginx 配置
#   nginx-38.conf          -> /etc/nginx/sites-enabled/default
#   nginx-38-upstream.conf -> /etc/nginx/sites-enabled/00-agent-upstream.conf
#     （含 map + upstream agent_cluster：thread 级粘性，多副本 agent）
# 部署前在服务器上备份到 /root/nginx-backups/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="${SSH_HOST:-root@38.76.211.27}"
BACKUP_DIR="/root/nginx-backups"

MAIN_SRC="${SCRIPT_DIR}/nginx-38.conf"
UPSTREAM_SRC="${SCRIPT_DIR}/nginx-38-upstream.conf"
MAIN_DST="/etc/nginx/sites-enabled/default"
UPSTREAM_DST="/etc/nginx/sites-enabled/00-agent-upstream.conf"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

DRY_RUN=false
case "${1:-}" in
    --dry-run) DRY_RUN=true ;;
    --help|-h) echo "用法: $0 [--dry-run]  (env: SSH_HOST)"; exit 0 ;;
    "") ;;
    *) log_error "未知参数: $1"; exit 1 ;;
esac

for f in "$MAIN_SRC" "$UPSTREAM_SRC"; do
    [ -f "$f" ] || { log_error "缺少文件: $f"; exit 1; }
done

log_info "目标: $SSH_HOST"
log_info "  $MAIN_SRC -> $MAIN_DST"
log_info "  $UPSTREAM_SRC -> $UPSTREAM_DST"

if [[ "$DRY_RUN" == true ]]; then
    log_info "[dry-run] 将上传并在服务器上备份 + nginx -t + reload（不实际执行）"
    exit 0
fi

TS="$(date +%Y%m%d%H%M%S)"
log_info "备份现有配置到 $BACKUP_DIR ..."
ssh "$SSH_HOST" "mkdir -p $BACKUP_DIR; cp -a $MAIN_DST $BACKUP_DIR/default.$TS 2>/dev/null || true; cp -a $UPSTREAM_DST $BACKUP_DIR/00-agent-upstream.conf.$TS 2>/dev/null || true"

log_info "上传配置..."
scp "$MAIN_SRC" "$SSH_HOST:$MAIN_DST"
scp "$UPSTREAM_SRC" "$SSH_HOST:$UPSTREAM_DST"

log_info "nginx -t + reload ..."
if ssh "$SSH_HOST" "nginx -t && nginx -s reload"; then
    log_success "完成（备份: ${BACKUP_DIR}/*.${TS}）"
else
    log_error "失败；可回滚: cp ${BACKUP_DIR}/default.${TS} ${MAIN_DST}"
    exit 1
fi
