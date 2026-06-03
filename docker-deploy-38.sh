#!/bin/bash

# Fina Demo 部署脚本 (38.76.211.27)
# 直接从 GitHub Container Registry 拉取镜像并部署，无需经过 Volcano Engine

set -euo pipefail

SERVICE_1="fina-demo-agent"
SERVICE_2="fina-demo-prediction-app"
SERVICE_3="fina-demo-ai-web"
SERVICE_4="fina-demo-metrics-server"

GHCR_REGISTRY="${GHCR_REGISTRY:-ghcr.io}"
GHCR_USERNAME="${GHCR_USERNAME:-409zhangshu}"
GHCR_TOKEN="${GHCR_TOKEN:-}"

DEPLOY_USER="root"
DEPLOY_HOST="38.76.211.27"
DEPLOY_PATH="/app/fina_demo"

COMPOSE_FILE="docker-compose.prod.ghcr.yml"
ENV_FILE="${ENV_FILE:-}"

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
    echo "  --help, -h          显示此帮助信息"
    echo "  --env-file FILE     指定要上传的环境变量文件"
    echo "  --no-env            不上传 .env 文件，使用服务器现有配置"
    echo "  --compose-only      只上传 docker-compose 文件，不执行部署"
    echo ""
    echo "示例:"
    echo "  $0                              # 部署所有服务"
    echo "  $0 1 3                          # 只部署服务1和服务3"
    echo "  $0 --env-file .env.prod         # 使用指定的环境变量文件"
    echo "  $0 --no-env                     # 不更改服务器上的 .env"
    echo ""
    echo "环境变量:"
    echo "  GHCR_TOKEN                      # GitHub Container Registry token (必需)"
    echo "  GHCR_USERNAME                   # GitHub 用户名 (默认: 409zhangshu)"
    exit 1
}

deploy() {
    local services_to_deploy=("$@")

    log_info "目标服务器: $DEPLOY_USER@$DEPLOY_HOST"
    log_info "部署路径: $DEPLOY_PATH"
    echo "----------------------------------------"

    # 1. 上传 compose 文件
    log_info "上传 $COMPOSE_FILE 到服务器..."
    if scp "$COMPOSE_FILE" "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/docker-compose.yml"; then
        log_success "docker-compose.yml 上传成功"
    else
        log_error "docker-compose.yml 上传失败"
        return 1
    fi

    # 2. 上传 .env 文件
    local env_file_to_upload=""
    if [[ -n "$ENV_FILE" ]]; then
        if [[ -f "$ENV_FILE" ]]; then
            env_file_to_upload="$ENV_FILE"
            log_info "使用指定的环境变量文件: $ENV_FILE"
        else
            log_error "指定的环境变量文件不存在: $ENV_FILE"
            log_warning "将跳过 .env 文件上传"
        fi
    else
        if [[ -f .env.prod ]]; then
            env_file_to_upload=".env.prod"
            log_info "找到生产环境配置文件: .env.prod"
        elif [[ -f .env ]]; then
            env_file_to_upload=".env"
            log_warning "找到本地开发环境文件: .env"
            if [[ -t 0 ]]; then
                read -p "是否继续使用本地 .env 文件？(y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    env_file_to_upload=""
                fi
            else
                log_info "非交互模式，跳过 .env 上传"
                env_file_to_upload=""
            fi
        fi
    fi

    if [[ -n "$env_file_to_upload" ]]; then
        log_info "上传 $env_file_to_upload 到服务器..."
        if scp "$env_file_to_upload" "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/.env"; then
            log_success ".env 上传成功"
        else
            log_warning ".env 上传失败，将使用服务器现有的 .env 文件或默认值"
        fi
    else
        log_info "未上传 .env 文件，使用服务器现有配置"
    fi

    # 3. 在服务器上登录 GHCR
    if [[ -n "$GHCR_TOKEN" ]]; then
        log_info "在服务器上登录 GitHub Container Registry..."
        local login_cmd="echo '$GHCR_TOKEN' | docker login $GHCR_REGISTRY -u $GHCR_USERNAME --password-stdin"
        if ! ssh "$DEPLOY_USER@$DEPLOY_HOST" "$login_cmd"; then
            log_error "服务器上登录 GHCR 失败"
            return 1
        fi
        log_success "GHCR 登录成功"
    else
        log_warning "GHCR_TOKEN 未设置，假设服务器已登录"
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
        log_info "将部署以下服务:$compose_services"
    else
        log_info "将部署所有服务"
    fi

    local ssh_cmd="cd $DEPLOY_PATH"

    # 传递 GHCR 相关变量给 docker-compose
    ssh_cmd="$ssh_cmd && export GHCR_REGISTRY='$GHCR_REGISTRY' GHCR_USERNAME='$GHCR_USERNAME'"

    if [[ -n "$compose_services" ]]; then
        ssh_cmd="$ssh_cmd && docker compose pull$compose_services"
        ssh_cmd="$ssh_cmd && docker compose stop$compose_services && docker compose rm -f$compose_services && docker compose up -d$compose_services"
    else
        ssh_cmd="$ssh_cmd && docker compose pull"
        ssh_cmd="$ssh_cmd && docker compose down --remove-orphans && docker compose up --force-recreate -d"
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

main() {
    echo "=========================================="
    echo "  Fina Demo 部署工具 (GHCR → 38.76.211.27)"
    echo "=========================================="
    echo

    local services=()
    local compose_only=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h) usage ;;
            --env-file)
                [[ -z "${2:-}" ]] && { log_error "--env-file 需要指定文件路径"; usage; }
                ENV_FILE="$2"
                shift 2
                ;;
            --no-env)
                ENV_FILE="/dev/null"
                shift
                ;;
            --compose-only)
                compose_only=true
                shift
                ;;
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

    if [[ ${#services[@]} -eq 0 ]]; then
        log_info "未指定具体服务，将部署所有服务"
    fi

    if [[ "$compose_only" == true ]]; then
        log_info "仅上传 compose 文件..."
        scp "$COMPOSE_FILE" "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/docker-compose.yml" \
            && log_success "上传完成" \
            || log_error "上传失败"
        exit 0
    fi

    if [[ ${#services[@]} -gt 0 ]]; then
        deploy "${services[@]}"
    else
        deploy
    fi

    echo
    log_success "所有操作完成！"
    echo "=========================================="
}

main "$@"
