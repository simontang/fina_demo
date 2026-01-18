# Python API 启动问题说明

## 问题一：`[Errno 48] Address already in use`（端口被占用）

**原因**：8000 端口已被其他 Python/uvicorn 进程占用（包括已退出的僵尸进程仍可能占着端口）。

**解决**：
```bash
# 查看占用 8000 的进程
lsof -i :8000

# 结束进程（把 PID 换成上面看到的数字）
kill -9 <PID>

# 或换端口启动
PORT=8001 python start_api.py
```

---

## 问题二：`ImportError: email-validator version >= 2.0 required`

**原因**：FastAPI/Pydantic 依赖 `email-validator>=2.0`，当前环境未安装或版本不够。

**解决**：用**虚拟环境**安装依赖（推荐，避免污染系统 Python）：

```bash
cd prediction_app

# 1. 创建虚拟环境
python3 -m venv .venv

# 2. 激活（Windows 用 .venv\Scripts\activate）
source .venv/bin/activate

# 3. 安装依赖
pip install -r api/requirements.txt

# 4. 启动（先确保 8000 没被占用）
python start_api.py
```

如果必须用系统 Python 且不想建 venv，可尝试：
```bash
pip3 install --user email-validator
```
（在 Homebrew Python 下可能仍会受 `externally-managed-environment` 限制，建议用 venv。）

---

## 问题三：`Could not import module "app"`（已修复）

**原因**：`start_api.py` 之前 `chdir` 到 `api/` 并用 `app:app` 启动，导致 `from api.inference` 等找不到 `api` 包。

**修复**：已改为在项目根目录用 `api.app:app` 启动，并保证 `project_root` 在 `sys.path` 中。
