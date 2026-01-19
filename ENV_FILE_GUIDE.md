# 环境变量文件管理指南

## 上传原则

### 核心原则

1. **分离开发和生产环境**
   - 本地开发环境的 `.env` **不应直接**上传到生产服务器
   - 生产环境应使用独立的配置文件（如 `.env.prod`）

2. **项目特定配置**
   - 不同项目（如 `fuli` 和 `fina_demo`）应使用不同的环境变量文件
   - 推荐命名：`.env.{项目名}`（如 `.env.fina_demo`）

3. **安全性优先**
   - 敏感信息（API 密钥、密码等）应通过服务器端密钥管理系统管理
   - 避免在版本控制中提交包含敏感信息的 `.env` 文件

## 文件优先级

脚本会按以下优先级查找环境变量文件：

1. **`--env-file` 指定的文件**（最高优先级）
   ```bash
   ./docker-transfer.sh --env-file .env.production
   ```

2. **`.env.prod`**（生产环境推荐）
   - 专门用于生产环境的配置
   - 不会被 Git 跟踪（应在 `.gitignore` 中）

3. **`.env.{项目名}`**（项目特定）
   - 例如：`.env.fina_demo`、`.env.fuli`
   - 用于区分不同项目的配置

4. **`.env`**（本地开发，需确认）
   - 如果找到本地 `.env`，脚本会提示确认
   - 建议不要直接使用，而是创建 `.env.prod`

## 使用示例

### 场景 1：使用生产环境配置文件

```bash
# 1. 创建生产环境配置文件
cp .env.example .env.prod

# 2. 编辑 .env.prod，填入生产环境的值
vim .env.prod

# 3. 部署（会自动使用 .env.prod）
./docker-transfer.sh
```

### 场景 2：指定特定的环境变量文件

```bash
# 使用自定义命名的环境变量文件
./docker-transfer.sh --env-file .env.production
```

### 场景 3：不同项目使用不同配置

```bash
# FULI 项目
cd /path/to/fuli
./docker-transfer.sh --env-file .env.fuli

# Fina Demo 项目
cd /path/to/fina_demo
./docker-transfer.sh --env-file .env.fina_demo
```

### 场景 4：服务器端管理环境变量

如果环境变量完全由服务器端管理（推荐用于生产环境）：

```bash
# 不指定任何环境变量文件，使用服务器上现有的配置
./docker-transfer.sh --no-env-upload
# 或者直接跳过上传，脚本会自动使用服务器上的 .env
```

## 最佳实践

### 1. 项目结构建议

```
项目根目录/
├── .env.example          # 模板文件（提交到 Git）
├── .env                  # 本地开发（不提交）
├── .env.prod             # 生产环境（不提交）
├── .env.fina_demo        # 项目特定配置（可选，不提交）
└── docker-transfer.sh    # 部署脚本
```

### 2. .gitignore 配置

确保以下文件不被提交：

```gitignore
# 环境变量文件
.env
.env.*
!.env.example
```

### 3. 服务器端配置

在生产服务器上：

```bash
# 服务器上的配置路径
/app/fina_demo/.env       # 由部署脚本上传或手动管理
```

### 4. 多项目部署

如果同一台服务器部署多个项目：

```bash
# FULI 项目
DEPLOY_PATH="/app/fuli"
./docker-transfer.sh --env-file .env.fuli

# Fina Demo 项目
DEPLOY_PATH="/app/fina_demo"
./docker-transfer.sh --env-file .env.fina_demo
```

## 安全注意事项

1. **不要提交敏感信息**
   - `.env`、`.env.prod` 等文件不应提交到 Git
   - 使用 `.env.example` 作为模板

2. **使用密钥管理服务**
   - 生产环境敏感信息应使用密钥管理服务（如 AWS Secrets Manager、HashiCorp Vault）
   - 环境变量文件仅用于非敏感配置

3. **定期轮换密钥**
   - 定期更新 API 密钥和密码
   - 更新后重新部署服务

4. **最小权限原则**
   - 服务器上的 `.env` 文件权限应设置为 `600`（仅所有者可读写）
   ```bash
   chmod 600 /app/fina_demo/.env
   ```

## 故障排查

### 问题：脚本找不到环境变量文件

**解决方案：**
- 检查文件是否存在：`ls -la .env*`
- 使用 `--env-file` 明确指定文件路径
- 或者跳过上传，使用服务器上现有的配置

### 问题：部署后服务无法启动

**可能原因：**
- 环境变量文件未正确上传
- 服务器上的 `.env` 文件格式错误
- 缺少必需的环境变量

**解决方案：**
```bash
# 检查服务器上的 .env 文件
ssh deploy@14.103.161.174 "cat /app/fina_demo/.env"

# 手动上传环境变量文件
scp .env.prod deploy@14.103.161.174:/app/fina_demo/.env
```

## 总结

- ✅ **推荐**：使用 `.env.prod` 用于生产环境
- ✅ **推荐**：不同项目使用不同的环境变量文件
- ✅ **推荐**：敏感信息通过服务器端密钥管理
- ❌ **不推荐**：直接上传本地开发环境的 `.env`
- ❌ **不推荐**：在 Git 中提交包含敏感信息的 `.env` 文件
