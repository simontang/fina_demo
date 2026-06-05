#!/bin/bash

# 214 服务器离线部署脚本
# 镜像已从火山仓库同步，此脚本仅负责上传配置并部署
# 支持：docker-compose 部署 / zip 包部署

set -euo pipefail

SERVICE_1="fina-demo-agent"
SERVICE_2="fina-demo-prediction-app"
SERVICE_3="fina-demo-ai-web"
SERVICE_4="fina-demo-metrics-server"

DEPLOY_USER="root"
DEPLOY_HOST="14.103.67.214"
DEPLOY_PATH="/app/fina_demo"
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

get_compose_service_name() {
    case $1 in
        "$SERVICE_1") echo "agent" ;;
        "$SERVICE_2") echo "prediction_app" ;;
        "$SERVICE_3") echo "ai_web_fina ai_web_evario" ;;
        "$SERVICE_4") echo "metrics_server" ;;
        *) echo "" ;;
    esac
}

package_zip() {
    local zip_name="fina-demo-offline-$(date +%Y%m%d-%H%M%S).zip"
    log_info "打包离线部署包: $zip_name"
    
    local tmpdir=$(mktemp -d)
    cp docker-compose.offline.yml "$tmpdir/docker-compose.yml"
    cp -r configs "$tmpdir/"
    
    if [[ -f .env ]]; then
        cp .env "$tmpdir/"
    fi
    
    (cd "$tmpdir" && zip -r "/Users/simon/code/fina_demo/$zip_name" .)
    rm -rf "$tmpdir"
    
    log_success "打包完成: ./$zip_name"
    echo ""
    echo "使用方式:"
    echo "  1. 上传到 214 服务器: scp $zip_name root@14.103.67.214:/tmp/"
    echo "  2. 解压: ssh root@14.103.67.214 'unzip -o /tmp/$zip_name -d /app/fina_demo'"
    echo "  3. 部署: ssh root@14.103.67.214 'cd /app/fina_demo && docker-compose up -d'"
    echo ""
}

deploy() {
    local services_to_deploy=("$@")

    log_info "目标服务器: $DEPLOY_USER@$DEPLOY_HOST"
    log_info "部署路径: $DEPLOY_PATH"
    echo "----------------------------------------"

    # 1. 上传 compose 文件
    log_info "上传 docker-compose.offline.yml..."
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "mkdir -p $DEPLOY_PATH"
    if scp "docker-compose.offline.yml" "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/docker-compose.yml"; then
        log_success "docker-compose.yml 上传成功"
    else
        log_error "上传失败"
        return 1
    fi

    # 2. 上传 configs
    log_info "上传 configs 目录..."
    if scp -r configs/* "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/configs/" 2>/dev/null || true; then
        log_success "configs 上传成功"
    fi

    # 3. 登录火山仓库
    log_info "在服务器登录火山仓库..."
    local login_cmd="echo '$VOLCANO_PASSWORD' | docker login $VOLCANO_REGISTRY -u '$VOLCANO_USERNAME' --password-stdin"
    if ! ssh "$DEPLOY_USER@$DEPLOY_HOST" "$login_cmd"; then
        log_error "火山仓库登录失败"
        return 1
    fi

    # 4. 拉取并部署
    local compose_services=""
    if [[ ${#services_to_deploy[@]} -gt 0 ]]; then
        for svc in "${services_to_deploy[@]}"; do
            local cs=$(get_compose_service_name "$svc")
            if [[ -n "$cs" ]]; then
                compose_services="$compose_services $cs"
            fi
        done
        log_info "将部署服务:$compose_services"
    else
        log_info "将部署所有服务"
    fi

    local ssh_cmd="cd $DEPLOY_PATH"
    if [[ -n "$compose_services" ]]; then
        ssh_cmd="$ssh_cmd && docker-compose pull$compose_services"
        ssh_cmd="$ssh_cmd && docker-compose stop$compose_services && docker-compose rm -f$compose_services && docker-compose up -d$compose_services"
    else
        ssh_cmd="$ssh_cmd && docker-compose pull && docker-compose down --remove-orphans && docker-compose up --force-recreate -d"
    fi

    log_info "执行部署..."
    if ssh "$DEPLOY_USER@$DEPLOY_HOST" "$ssh_cmd"; then
        log_success "部署完成！"
    else
        log_error "部署失败"
        return 1
    fi
    echo "----------------------------------------"
}

usage() {
    echo "用法: $0 [选项] [服务编号...]"
    echo ""
    echo "服务编号:"
    echo "  1 - $SERVICE_1"
    echo "  2 - $SERVICE_2"
    echo "  3 - $SERVICE_3"
    echo "  4 - $SERVICE_4"
    echo ""
    echo "选项:"
    echo "  --help, -h       显示帮助"
    echo "  --package, -p    打包 zip 离线部署包（不上传/部署）"
    echo "  --deploy-only    只部署，不检查火山仓库登录"
    echo ""
    echo "示例:"
    echo "  $0                    # 部署所有服务"
    echo "  $0 1 3                # 只部署 agent 和 web"
    echo "  $0 --package          # 生成 zip 包"
    echo ""
    echo "环境变量:"
    echo "  VOLCANO_USERNAME    火山仓库用户名"
    echo "  VOLCANO_PASSWORD    火山仓库密码"
    exit 1
}

main() {
    echo "=========================================="
    echo "  Fina Demo 离线部署 (214 ← 火山仓库)"
    echo "=========================================="
    echo

    local services=()
    local package_only=false
    local deploy_only=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h) usage ;;
            --package|-p) package_only=true; shift ;;
            --deploy-only) deploy_only=true; shift ;;
            1) services+=("$SERVICE_1"); shift ;;
            2) services+=("$SERVICE_2"); shift ;;
            3) services+=("$SERVICE_3"); shift ;;
            4) services+=("$SERVICE_4"); shift ;;
            *)
                log_error "无效参数: $1"
                usage
                ;;
        esac
    done

    if [[ "$package_only" == true ]]; then
        package_zip
        exit 0
    fi

    if [[ "$deploy_only" != true ]]; then
        if [[ -z "$VOLCANO_USERNAME" || -z "$VOLCANO_PASSWORD" ]]; then
            log_error "VOLCANO_USERNAME/VOLCANO_PASSWORD 未设置"
            exit 1
        fi
    fi

    if [[ ${#services[@]} -eq 0 ]]; then
        log_info "未指定服务，将部署所有服务"
    fi

    deploy "${services[@]}"

    echo
    log_success "所有操作完成！"
    echo "=========================================="
}

main "$@"
