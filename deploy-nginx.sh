#!/bin/bash

# Deploy nginx config for Fina Demo to the server.
# Before each deploy, backs up the current config on the server.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CONF="${SCRIPT_DIR}/nginx-fina-demo.conf"

# Server config (must match docker-transfer.sh for consistency)
DEPLOY_USER="${DEPLOY_USER:-deploy}"
DEPLOY_HOST="${DEPLOY_HOST:-14.103.152.204}"
DEPLOY_PATH="${DEPLOY_PATH:-/app/fina_demo}"

# Nginx paths on the server (require sudo to modify)
NGINX_SITE_AVAILABLE="/etc/nginx/sites-available/fina-demo"
NGINX_SITES_AVAILABLE_DIR="/etc/nginx/sites-available"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy nginx-fina-demo.conf to the server. Backs up current config before overwriting."
    echo ""
    echo "Options:"
    echo "  --dry-run    Upload and backup only; do not replace config or reload nginx"
    echo "  --help, -h   Show this help"
    echo ""
    echo "Env (optional): DEPLOY_USER, DEPLOY_HOST, DEPLOY_PATH"
    exit 0
}

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

if [[ ! -f "$LOCAL_CONF" ]]; then
    log_error "Local config not found: $LOCAL_CONF"
    exit 1
fi

log_info "Deploying nginx config to $DEPLOY_USER@$DEPLOY_HOST"
echo "----------------------------------------"

# 1. Upload local config to deploy path (no root needed)
REMOTE_UPLOAD="$DEPLOY_PATH/nginx-fina-demo.conf"
log_info "Uploading $LOCAL_CONF to $DEPLOY_USER@$DEPLOY_HOST:$REMOTE_UPLOAD ..."
if ! scp "$LOCAL_CONF" "$DEPLOY_USER@$DEPLOY_HOST:$REMOTE_UPLOAD"; then
    log_error "Upload failed"
    exit 1
fi
log_success "Uploaded"

# 2. On server: backup current config, then replace and reload (sudo)
BACKUP_NAME="fina-demo.bak.$(date +%Y%m%d-%H%M%S)"
if [[ "$DRY_RUN" == true ]]; then
    log_info "[dry-run] Would: backup $NGINX_SITE_AVAILABLE -> $NGINX_SITES_AVAILABLE_DIR/$BACKUP_NAME"
    log_info "[dry-run] Would: copy $REMOTE_UPLOAD -> $NGINX_SITE_AVAILABLE, then nginx -t && systemctl reload nginx"
    log_success "Dry run done. No change on server nginx."
    echo "----------------------------------------"
    exit 0
fi

log_info "On server: backup current config, install new config, and reload nginx..."
SSH_CMD="sudo cp $NGINX_SITE_AVAILABLE $NGINX_SITES_AVAILABLE_DIR/$BACKUP_NAME && \
         sudo cp $REMOTE_UPLOAD $NGINX_SITE_AVAILABLE && \
         sudo nginx -t && \
         sudo systemctl reload nginx"
if ssh "$DEPLOY_USER@$DEPLOY_HOST" "$SSH_CMD"; then
    log_success "Backup saved as $NGINX_SITES_AVAILABLE_DIR/$BACKUP_NAME"
    log_success "Config updated and nginx reloaded"
else
    log_error "Server backup/install/reload failed (backup may still have been created)"
    exit 1
fi

echo "----------------------------------------"
log_success "Nginx deploy finished."
