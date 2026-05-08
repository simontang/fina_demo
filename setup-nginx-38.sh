#!/bin/bash

# Nginx 配置部署脚本 (38.76.211.27)
# 将 nginx 配置上传到服务器并重载

set -euo pipefail

SSH_USER="root"
SSH_HOST="38.76.211.27"
NGINX_CONF="/etc/nginx/sites-available/fina-demo"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

TMPFILE=$(mktemp)

cat > "$TMPFILE" << 'ENDCONF'
server {
    listen 80 default_server;
    server_name _;

    location /api/metrics/ {
        proxy_pass http://127.0.0.1:5704/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
        proxy_connect_timeout 10s;
    }
    location = /api/metrics {
        return 301 $scheme://$host/api/metrics/;
    }

    location /api {
        proxy_pass http://127.0.0.1:5702;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    location = / { return 301 /admin/; }

    location /admin {
        proxy_pass http://127.0.0.1:5701;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_redirect off;
    }

    location /sandbox/global {
        rewrite ^/sandbox/global/?(.*) /$1 break;
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        proxy_buffering off;
        proxy_request_buffering off;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
ENDCONF

log_info "上传 nginx 配置到 $SSH_HOST ..."
if scp "$TMPFILE" "$SSH_USER@$SSH_HOST:$NGINX_CONF"; then
    log_success "配置已上传"
else
    log_error "上传失败"
    rm -f "$TMPFILE"
    exit 1
fi
rm -f "$TMPFILE"

log_info "确保 default 软链接存在..."
ssh "$SSH_USER@$SSH_HOST" "ln -sf $NGINX_CONF /etc/nginx/sites-enabled/default"

log_info "测试 nginx 配置..."
if ssh "$SSH_USER@$SSH_HOST" "nginx -t"; then
    log_success "nginx 配置语法正确"
else
    log_error "nginx 配置有误，检查 $NGINX_CONF"
    exit 1
fi

log_info "重载 nginx..."
if ssh "$SSH_USER@$SSH_HOST" "systemctl reload nginx 2>/dev/null || service nginx reload"; then
    log_success "nginx 重载成功"
else
    log_error "nginx 重载失败"
    exit 1
fi

echo
log_success "完成！通过 http://$SSH_HOST/admin/ 访问"
