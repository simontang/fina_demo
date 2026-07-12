#!/bin/bash

# GHCR → Volcano Engine Registry 镜像同步脚本
# 从 GitHub Container Registry 拉取镜像并推送到火山引擎容器镜像仓库

set -euo pipefail

SERVICE_1="fina-demo-agent"
SERVICE_2="fina-demo-prediction-app"
SERVICE_3="fina-demo-ai-web"
SERVICE_4="fina-demo-metrics-server"
SERVICE_5="fina-demo-cdp-service"

GITHUB_REGISTRY="${GITHUB_REGISTRY:-ghcr.io}"
GITHUB_USERNAME="${GITHUB_USERNAME:-409zhangshu}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

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

check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker 未运行或无法访问，请先启动 Docker"
        exit 1
    fi
    log_success "Docker 检查通过"
}

login_registries() {
    log_info "登录到 GitHub Container Registry..."
    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "GITHUB_TOKEN 未设置。请 export GITHUB_TOKEN=... 后重试"
        exit 1
    fi
    if echo "$GITHUB_TOKEN" | docker login $GITHUB_REGISTRY -u $GITHUB_USERNAME --password-stdin; then
        log_success "GitHub Container Registry 登录成功"
    else
        log_error "GitHub Container Registry 登录失败"
        exit 1
    fi

    log_info "登录到 Volcano Engine Container Registry..."
    if [[ -z "$VOLCANO_USERNAME" || -z "$VOLCANO_PASSWORD" ]]; then
        log_error "VOLCANO_USERNAME/VOLCANO_PASSWORD 未设置"
        exit 1
    fi
    if echo "$VOLCANO_PASSWORD" | docker login $VOLCANO_REGISTRY -u $VOLCANO_USERNAME --password-stdin; then
        log_success "Volcano Engine Container Registry 登录成功"
    else
        log_error "Volcano Engine Container Registry 登录失败"
        exit 1
    fi
    echo
}

transfer_service() {
    local service=$1

    log_info "同步服务: $service"
    echo "----------------------------------------"

    log_info "拉取 $service (linux/amd64)..."
    if docker pull --platform linux/amd64 $GITHUB_REGISTRY/$GITHUB_USERNAME/$service:latest; then
        log_success "拉取成功"
    else
        log_error "拉取失败: $GITHUB_REGISTRY/$GITHUB_USERNAME/$service:latest"
        return 1
    fi

    log_info "打标签: $VOLCANO_REGISTRY/default/$service:latest"
    if docker tag $GITHUB_REGISTRY/$GITHUB_USERNAME/$service:latest $VOLCANO_REGISTRY/default/$service:latest; then
        log_success "标签打成功"
    else
        log_error "标签打失败"
        return 1
    fi

    log_info "推送到 Volcano Engine..."
    if docker push $VOLCANO_REGISTRY/default/$service:latest; then
        log_success "推送成功"
    else
        log_error "推送失败"
        return 1
    fi

    log_success "服务 $service 同步完成"
    echo "----------------------------------------"
    echo
}

usage() {
    echo "用法: $0 [服务编号]..."
    echo ""
    echo "服务编号:"
    echo "  1 - $SERVICE_1"
    echo "  2 - $SERVICE_2"
    echo "  3 - $SERVICE_3"
    echo "  4 - $SERVICE_4"
    echo "  5 - $SERVICE_5"
    echo ""
    echo "示例:"
    echo "  $0              # 同步所有服务"
    echo "  $0 1 3          # 只同步服务 1 和 3"
    echo ""
    echo "环境变量 (必需):"
    echo "  GITHUB_TOKEN           GitHub Container Registry token"
    echo "  VOLCANO_USERNAME       Volcano Engine 用户名"
    echo "  VOLCANO_PASSWORD       Volcano Engine 密码"
    exit 1
}

main() {
    echo "=========================================="
    echo "  GHCR → Volcano Engine 镜像同步工具"
    echo "=========================================="
    echo

    local services=()

    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h) usage ;;
            1) services+=("$SERVICE_1"); shift ;;
            2) services+=("$SERVICE_2"); shift ;;
            3) services+=("$SERVICE_3"); shift ;;
            4) services+=("$SERVICE_4"); shift ;;
            5) services+=("$SERVICE_5"); shift ;;
            *)
                log_error "无效参数: $1"
                usage
                ;;
        esac
    done

    if [[ ${#services[@]} -eq 0 ]]; then
        log_info "未指定服务，将同步所有服务"
        services=("$SERVICE_1" "$SERVICE_2" "$SERVICE_3" "$SERVICE_4" "$SERVICE_5")
    fi

    check_docker
    echo
    login_registries

    local failed=()
    for service in "${services[@]}"; do
        if ! transfer_service "$service"; then
            failed+=("$service")
        fi
    done

    if [[ ${#failed[@]} -gt 0 ]]; then
        log_error "以下服务同步失败: ${failed[*]}"
        exit 1
    fi

    log_success "全部同步完成！"
    echo "=========================================="
}

main "$@"
