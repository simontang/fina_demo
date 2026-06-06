#!/bin/bash

# 214 服务器离线部署脚本（使用 offline-package 配置）
set -euo pipefail

DEPLOY_USER="root"
DEPLOY_HOST="14.103.67.214"
DEPLOY_PATH="/app/fina_demo_offline"
VOLCANO_REGISTRY="${VOLCANO_REGISTRY:-finai-cn-shanghai.cr.volces.com}"
VOLCANO_USERNAME="${VOLCANO_USERNAME:-}"
VOLCANO_PASSWORD="${VOLCANO_PASSWORD:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

deploy() {
    log_info "目标服务器: $DEPLOY_USER@$DEPLOY_HOST"
    log_info "部署路径: $DEPLOY_PATH"
    echo "----------------------------------------"

    # 创建目录并上传文件
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "mkdir -p $DEPLOY_PATH"
    
    log_info "上传 docker-compose.offline.yml..."
    scp "offline-package/docker-compose.offline.yml" "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/docker-compose.yml"
    log_success "上传成功"

    log_info "上传 .env.offline..."
    if [[ -f offline-package/.env.offline ]]; then
        scp "offline-package/.env.offline" "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/.env"
        log_success "环境变量文件上传成功"
    fi

    log_info "上传 configs..."
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "mkdir -p $DEPLOY_PATH/configs"
    scp -r configs/* "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/configs/" 2>/dev/null || true
    log_success "configs 上传成功"

    # 登录火山仓库
    log_info "登录火山仓库..."
    local login_cmd="echo '$VOLCANO_PASSWORD' | docker login $VOLCANO_REGISTRY -u '$VOLCANO_USERNAME' --password-stdin"
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "$login_cmd"
    log_success "登录成功"

    # 部署
    log_info "执行部署..."
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "cd $DEPLOY_PATH && docker-compose pull && docker-compose up -d"
    log_success "部署完成！"
    
    echo "----------------------------------------"
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "cd $DEPLOY_PATH && docker-compose ps"
}

package_zip() {
    local zip_name="fina-demo-offline-$(date +%Y%m%d-%H%M%S).zip"
    log_info "打包离线部署包: $zip_name"
    
    local tmpdir=$(mktemp -d)
    cp offline-package/docker-compose.offline.yml "$tmpdir/docker-compose.yml"
    cp offline-package/.env.offline "$tmpdir/.env" 2>/dev/null || true
    cp -r configs "$tmpdir/"
    
    (cd "$tmpdir" && zip -r "/Users/simon/code/fina_demo/$zip_name" .)
    rm -rf "$tmpdir"
    
    log_success "打包完成: ./$zip_name"
}

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --deploy         部署到 214 服务器（默认）"
    echo "  --package, -p    打包 zip 离线包"
    echo "  --help, -h       显示帮助"
    echo ""
    echo "环境变量:"
    echo "  VOLCANO_USERNAME    火山仓库用户名"
    echo "  VOLCANO_PASSWORD    火山仓库密码"
    exit 1
}

main() {
    echo "=========================================="
    echo "  Fina Demo 离线部署 (214)"
    echo "=========================================="
    echo

    local action="deploy"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h) usage ;;
            --package|-p) action="package"; shift ;;
            --deploy) action="deploy"; shift ;;
            *)
                log_error "无效参数: $1"
                usage
                ;;
        esac
    done

    if [[ "$action" == "package" ]]; then
        package_zip
    else
        if [[ -z "$VOLCANO_USERNAME" || -z "$VOLCANO_PASSWORD" ]]; then
            log_error "VOLCANO_USERNAME/VOLCANO_PASSWORD 未设置"
            exit 1
        fi
        deploy
        echo
        log_success "所有操作完成！"
        echo "=========================================="
    fi
}

main "$@"
