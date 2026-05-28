#!/bin/bash

# Fina Demo 部署脚本 - 服务器 14.103.67.214
# 用于将镜像从 GitHub Container Registry 传输到 Volcano Engine Container Registry
# 并自动部署到服务器 14.103.67.214

# 定义服务配置
SERVICE_1="fina-demo-agent"
SERVICE_2="fina-demo-prediction-app"
SERVICE_3="fina-demo-ai-web"
SERVICE_4="fina-demo-metrics-server"

# GitHub Container Registry 配置
GITHUB_REGISTRY="${GITHUB_REGISTRY:-ghcr.io}"
GITHUB_USERNAME="${GITHUB_USERNAME:-409zhangshu}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# Volcano Engine Container Registry 配置
VOLCANO_REGISTRY="${VOLCANO_REGISTRY:-finai-cn-shanghai.cr.volces.com}"
VOLCANO_USERNAME="${VOLCANO_USERNAME:-}"
VOLCANO_PASSWORD="${VOLCANO_PASSWORD:-}"

# 服务器部署配置
DEPLOY_USER="root"
DEPLOY_HOST="14.103.67.214"
DEPLOY_PATH="/app/fina_demo"

# 环境变量文件配置
ENV_FILE="${ENV_FILE:-}"
ENV_FILE_TO_UPLOAD=""

# 颜色输出配置
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Docker 是否运行
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker 未运行或无法访问，请先启动 Docker"
        exit 1
    fi
    log_success "Docker 检查通过"
}

# 传输指定服务的镜像
transfer_service() {
    local service=$1
    local step=1
    
    log_info "开始传输服务: $service"
    echo "----------------------------------------"
    
    # 步骤1: 从 GitHub Container Registry 拉取镜像
    log_info "[$step/3] 从 GitHub Container Registry 拉取 $service 镜像 (linux/amd64)..."
    if docker pull --platform linux/amd64 $GITHUB_REGISTRY/$GITHUB_USERNAME/$service:latest; then
        log_success "成功拉取 $service 镜像"
    else
        log_error "拉取 $service 镜像失败"
        log_error "请确保镜像已在 GitHub Container Registry 中存在: $GITHUB_REGISTRY/$GITHUB_USERNAME/$service:latest"
        return 1
    fi
    step=$((step + 1))
    
    # 步骤2: 为 Volcano Engine Container Registry 打标签
    log_info "[$step/3] 为 $service 镜像打 Volcano Engine Container Registry 标签..."
    if docker tag $GITHUB_REGISTRY/$GITHUB_USERNAME/$service:latest $VOLCANO_REGISTRY/default/$service:latest; then
        log_success "成功为 $service 镜像打标签"
    else
        log_error "为 $service 镜像打标签失败"
        return 1
    fi
    step=$((step + 1))
    
    # 步骤3: 推送镜像到 Volcano Engine Container Registry
    log_info "[$step/3] 推送 $service 镜像到 Volcano Engine Container Registry..."
    if docker push $VOLCANO_REGISTRY/default/$service:latest; then
        log_success "成功推送 $service 镜像"
    else
        log_error "推送 $service 镜像失败"
        return 1
    fi
    
    log_success "服务 $service 镜像传输完成！"
    echo "----------------------------------------"
    echo
}

# 显示使用说明
usage() {
    echo "用法: $0 [服务编号]..."
    echo "  服务编号:"
    echo "    1 - $SERVICE_1"
    echo "    2 - $SERVICE_2"
    echo "    3 - $SERVICE_3"
    echo "    4 - $SERVICE_4"
    echo ""
    echo "选项:"
    echo "    --help, -h          显示此帮助信息"
    echo "    --no-deploy         只传输镜像，不执行部署"
    echo "    --deploy-only       只执行部署，不传输镜像"
    echo "    --env-file FILE     指定要上传的环境变量文件"
    echo ""
    echo "示例:"
    echo "    $0                      # 传输所有服务并部署"
    echo "    $0 1 3                  # 只传输服务1和服务3并部署"
    echo "    $0 --no-deploy          # 传输所有服务但不部署"
    echo "    $0 --deploy-only        # 只执行部署"
    echo "    $0 --env-file .env.prod # 使用指定的环境变量文件"
    exit 1
}

# 登录到容器注册表
login_registries() {
    log_info "登录到容器注册表..."
    
    # 登录到 GitHub Container Registry
    log_info "登录到 GitHub Container Registry..."
    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "Missing GITHUB_TOKEN. Export it before running, e.g. 'export GITHUB_TOKEN=...'."
        exit 1
    fi
    if echo "$GITHUB_TOKEN" | docker login $GITHUB_REGISTRY -u $GITHUB_USERNAME --password-stdin; then
        log_success "成功登录到 GitHub Container Registry"
    else
        log_error "登录到 GitHub Container Registry 失败"
        exit 1
    fi
    
    # 登录到 Volcano Engine Container Registry
    log_info "登录到 Volcano Engine Container Registry..."
    if [[ -z "$VOLCANO_USERNAME" || -z "$VOLCANO_PASSWORD" ]]; then
        log_error "Missing VOLCANO_USERNAME/VOLCANO_PASSWORD. Export them before running."
        exit 1
    fi
    if echo "$VOLCANO_PASSWORD" | docker login $VOLCANO_REGISTRY -u $VOLCANO_USERNAME --password-stdin; then
        log_success "成功登录到 Volcano Engine Container Registry"
    else
        log_error "登录到 Volcano Engine Container Registry 失败"
        exit 1
    fi
    
    echo
}

# Map service names to docker-compose service names
get_compose_service_name() {
    local service=$1
    case $service in
        "$SERVICE_1")
            echo "agent"
            ;;
        "$SERVICE_2")
            echo "prediction_app"
            ;;
        "$SERVICE_3")
            echo "ai_web_fina ai_web_evario"
            ;;
        "$SERVICE_4")
            echo "metrics_server"
            ;;
        *)
            echo ""
            ;;
    esac
}

select_env_file() {
    ENV_FILE_TO_UPLOAD=""

    if [[ -n "$ENV_FILE" ]]; then
        if [[ -f "$ENV_FILE" ]]; then
            ENV_FILE_TO_UPLOAD="$ENV_FILE"
            log_info "使用指定的环境变量文件: $ENV_FILE"
        else
            log_error "指定的环境变量文件不存在: $ENV_FILE"
            log_warning "将跳过 .env 文件上传，使用服务器上现有的配置"
        fi
    elif [[ -f .env.prod ]]; then
        ENV_FILE_TO_UPLOAD=".env.prod"
        log_info "找到生产环境配置文件: .env.prod"
    elif [[ -f .env.fina_demo ]]; then
        ENV_FILE_TO_UPLOAD=".env.fina_demo"
        log_info "找到项目特定配置文件: .env.fina_demo"
    elif [[ -f .env ]]; then
        log_warning "找到本地开发环境文件: .env"
        log_warning "建议使用 .env.prod 或 .env.fina_demo 用于生产部署"
        if [[ -t 0 ]]; then
            read -p "是否继续使用本地 .env 文件？(y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                ENV_FILE_TO_UPLOAD=".env"
            else
                log_info "跳过 .env 文件上传，使用服务器上现有的配置"
            fi
        else
            log_info "非交互模式，跳过 .env 上传，使用服务器上现有配置"
        fi
    else
        log_info "未找到环境变量文件，将使用服务器上现有的 .env 文件或默认值"
    fi
}

deploy_to_server() {
    local services_to_deploy=("$@")

    select_env_file

    log_info "开始部署到服务器 $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH ..."
    echo "----------------------------------------"

    ssh $DEPLOY_USER@$DEPLOY_HOST "mkdir -p $DEPLOY_PATH"

    log_info "上传 docker-compose.214.yml 到服务器..."
    if scp "docker-compose.214.yml" $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/docker-compose.yml; then
        log_success "docker-compose.yml 上传成功"
    else
        log_error "docker-compose.yml 上传失败"
        return 1
    fi

    log_info "上传 configs/ 目录到服务器..."
    ssh $DEPLOY_USER@$DEPLOY_HOST "mkdir -p $DEPLOY_PATH/configs"
    if scp -r configs/* $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/configs/; then
        log_success "configs/ 目录上传成功"
    else
        log_error "configs/ 目录上传失败"
        return 1
    fi

    if [[ -n "$ENV_FILE_TO_UPLOAD" ]]; then
        log_info "上传环境变量文件 $ENV_FILE_TO_UPLOAD 到服务器..."
        scp "$ENV_FILE_TO_UPLOAD" $DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/.env \
            || log_warning "环境变量文件上传失败，将使用服务器上现有的 .env 文件或默认值"
    fi

    log_info "在服务器上登录到 Volcano Engine Container Registry..."
    local login_cmd="echo '$VOLCANO_PASSWORD' | docker login $VOLCANO_REGISTRY -u $VOLCANO_USERNAME --password-stdin"
    if ! ssh $DEPLOY_USER@$DEPLOY_HOST "$login_cmd"; then
        log_error "服务器上登录容器注册表失败"
        return 1
    fi

    local compose_services=""
    for service in "${services_to_deploy[@]}"; do
        local compose_service=$(get_compose_service_name "$service")
        if [[ -n "$compose_service" ]]; then
            compose_services="$compose_services $compose_service"
        fi
    done

    local ssh_cmd="cd $DEPLOY_PATH && docker-compose pull$compose_services"
    if [[ -n "$compose_services" ]]; then
        ssh_cmd="$ssh_cmd && docker-compose stop$compose_services && docker-compose rm -f$compose_services && docker-compose up -d$compose_services"
    else
        ssh_cmd="$ssh_cmd && docker-compose down --remove-orphans && docker-compose up --force-recreate -d"
    fi

    log_info "连接到服务器 $DEPLOY_USER@$DEPLOY_HOST 执行部署..."
    if ssh $DEPLOY_USER@$DEPLOY_HOST "$ssh_cmd"; then
        log_success "部署完成！"
    else
        log_error "部署失败"
        return 1
    fi
    echo "----------------------------------------"
    echo
}

# 主程序开始
main() {
    echo "=========================================="
    echo "       Fina Demo Docker 镜像传输工具"
    echo "       目标服务器: 14.103.67.214"
    echo "=========================================="
    echo
    
    # 解析命令行参数
    local services_to_transfer=()
    local should_deploy=true
    local transfer_images=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                usage
                ;;
            --no-deploy)
                should_deploy=false
                shift
                ;;
            --deploy-only)
                transfer_images=false
                shift
                ;;
            --env-file)
                if [[ -z "$2" ]]; then
                    log_error "--env-file 需要指定文件路径"
                    usage
                fi
                ENV_FILE="$2"
                shift 2
                ;;
            1)
                services_to_transfer+=("$SERVICE_1")
                shift
                ;;
            2)
                services_to_transfer+=("$SERVICE_2")
                shift
                ;;
            3)
                services_to_transfer+=("$SERVICE_3")
                shift
                ;;
            4)
                services_to_transfer+=("$SERVICE_4")
                shift
                ;;
            *)
                log_error "无效的服务编号: $1"
                usage
                ;;
        esac
    done
    
    # 如果没有指定服务，则传输所有服务
    if [[ ${#services_to_transfer[@]} -eq 0 ]]; then
        log_info "未指定具体服务，将传输所有服务..."
        services_to_transfer=("$SERVICE_1" "$SERVICE_2" "$SERVICE_3" "$SERVICE_4")
    fi
    
    # 检查 Docker
    check_docker
    echo
    
    # 传输镜像
    if [[ "$transfer_images" == true ]]; then
        # 登录到容器注册表
        login_registries
        
        # 传输服务镜像
        local failed_services=()
        for service in "${services_to_transfer[@]}"; do
            if ! transfer_service "$service"; then
                failed_services+=("$service")
            fi
        done
        
        # 检查是否有失败的服务
        if [[ ${#failed_services[@]} -gt 0 ]]; then
            log_error "以下服务传输失败: ${failed_services[*]}"
            log_warning "继续执行部署可能会导致问题"
            read -p "是否继续部署？(y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "用户取消部署"
                exit 1
            fi
        else
            log_success "所有服务镜像传输完成！"
        fi
        
        echo
    fi
    
    # 执行部署
    if [[ "$should_deploy" == true ]]; then
        deploy_to_server "${services_to_transfer[@]}"
        log_success "所有操作完成！"
    else
        log_info "跳过部署步骤"
    fi
    
    echo "=========================================="
}

# 执行主程序
main "$@"
