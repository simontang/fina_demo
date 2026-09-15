# Webhook 回调集成指南（面向接收方）

- 适用：订阅 platform-service 事件、在自己的服务里接收回调的开发者
- 关联：[WEBHOOK-API.md](./WEBHOOK-API.md)（注册与发布接口）、[GATEWAY-INTEGRATION.md](./GATEWAY-INTEGRATION.md)（网关对接）

---

## 1. 协议总览

| 项 | 值 |
|---|---|
| 传输 | **HTTPS POST**，`Content-Type: application/json`，请求体是事件的 `data` |
| 鉴权 | **签名验证**（Standard Webhooks 规范，对称 **HMAC-SHA256**）——没有 OAuth / mTLS / Bearer |
| 投递语义 | **at-least-once**（可能重复投递，按 `webhook-id` 幂等去重） |
| 成功判定 | 接收方返回 **2xx**；非 2xx 或超时 → 自动重试（指数退避） |
| 顺序 | **不保证**；需要顺序时按 `webhook-timestamp` 或业务键自行处理 |
| 超时 | 接收方应**快速返回**（建议秒级），重活在返回后异步做 |

每个事件会对**每个匹配该 eventType/filterTypes 的目标**各投递一次（fan-out），单次投递的结果独立记录。

## 2. 注册 endpoint 需要提供什么

调用（租户由调用方的凭据/头决定，见网关文档）：

```http
POST /api/webhooks/destinations
X-Tenant-Id: <你的租户>
Content-Type: application/json

{
  "url": "https://your-service.example.com/hooks/platform",
  "filterTypes": ["job.completed", "gate.passed"],
  "channels": ["client_a"],
  "description": "生产环境回调"
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `url` | ✅ | 接收回调的 **HTTPS** 地址（生产环境勿用内网地址，见 §8） |
| `filterTypes` | ✅ | Svix endpoint event type 过滤；一个目标可订阅多个 |
| `channels` | ❌ | Svix channel 过滤；不填表示不按 channel 过滤 |
| `description` | ❌ | 备注，便于运维识别 |

可用 eventType：`import.completed`、`gate.passed`、`decision.captured`、`job.completed`、`run.published`（新 eventType 首次发布时会自动注册）。

响应（**`secret` 只在此时返回一次，请立即安全保存**）：

```json
{
  "endpointId": "ep_3JJ17t0nwoyTXRXjA9ItTucQdae",
  "url": "https://your-service.example.com/hooks/platform",
  "filterTypes": ["job.completed", "gate.passed"],
  "channels": ["client_a"],
  "secret": "whsec_f4MW7wAjPx…"
}
```

### 接收方需要准备的东西

| 事项 | 说明 |
|---|---|
| **HTTPS 端点** | 公网可达、证书有效；能接收 POST JSON |
| **保存 `secret`** | 形如 `whsec_<base64>`；**按目标一把**，不同目标/租户各不相同。用于验签，泄露等于他人可伪造回调 |
| **幂等存储** | 记录已处理的 `webhook-id`，重复投递直接返回 2xx |
| **快速 ACK** | 先返回 2xx，再异步处理业务 |
| **失败告警** | 非 2xx 会触发重试；持续失败需要你在自己侧告警 |

## 3. 每次回调携带的头

```
POST /hooks/platform
webhook-id: msg_3JHTFnhm9CTgBOgRzyeRaJ9peDk
webhook-timestamp: 1789316892
webhook-signature: v1,IBSl8td+sPuCKa2qk3/nitf92oObm9RlsroTqEcuHzY=
content-type: application/json

{"gate":"scope_confirmed","decisions":["D-2026-001"]}
```

| 头 | 含义 |
|---|---|
| `webhook-id` | 消息唯一标识，**也是幂等键**（重复投递时值相同） |
| `webhook-timestamp` | 投递时刻的 **Unix 秒** |
| `webhook-signature` | 形如 `v1,<base64>`；**可含多个、空格分隔**（密钥轮换时） |

> 兼容性：本服务（svix 1.101.0）发送 `webhook-*` 头名；二进制内同时保留 `svix-*` 旧名。稳妥做法是**两族都读**（本仓库的 `scripts/mock-receiver.py` 就是这么实现的）。

## 4. 验签怎么做（对称 HMAC）

算法：对 `"{webhook-id}.{webhook-timestamp}.{原始请求体}"` 做 HMAC-SHA256，密钥是 `secret` 去掉 `whsec_` 前缀后的 **base64 解码字节**，结果再 base64。

```python
import base64, hmac, hashlib, time

SECRET = "whsec_f4MW7wAjPx…"          # 注册时保存的那把
TOLERANCE_SECONDS = 300                # 5 分钟；可按需收紧

def verify(headers, raw_body: bytes) -> bool:
    msg_id = headers.get("webhook-id") or headers.get("svix-id")
    ts     = headers.get("webhook-timestamp") or headers.get("svix-timestamp")
    sig    = headers.get("webhook-signature") or headers.get("svix-signature")
    if not (msg_id and ts and sig):
        return False

    # ① 时间戳容差 —— 防重放
    try:
        if abs(time.time() - int(ts)) > TOLERANCE_SECONDS:
            return False
    except ValueError:
        return False

    # ② 计算期望签名（注意：用「原始字节」，不要先 json.loads 再重新序列化）
    key = base64.b64decode(SECRET.removeprefix("whsec_"))
    signed = f"{msg_id}.{ts}.".encode() + raw_body
    expected = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()

    # ③ 逐个比对（可能有多枚，任一匹配即通过）；常量时间比较
    for part in sig.split(" "):
        if part.startswith("v1,") and hmac.compare_digest(part[3:], expected):
            return True
    return False
```

**接入顺序建议**：先读**原始请求体**（框架里通常是 `request.body` / `@RequestBody byte[]`），验签通过后再 `json.loads`。若框架已经把 body 解析成对象，务必用它能提供的"原始字节"接口，别用反序列化后的对象再 `dumps` —— key 顺序与空白差异会让验签必然失败（这是最常见的集成坑）。

### 现成库（推荐，省去手写）

| 语言 | 库 |
|---|---|
| Python | `standardwebhooks`（`pip install standardwebhooks`） |
| TypeScript/JS | `standardwebhooks` 或 `svix` |
| Java | `com.svix:svix` 的 `Webhook.verify(...)` |
| Go | `github.com/svix/svix-webhooks/go` |

## 5. 响应与重试

| 你返回 | 结果 |
|---|---|
| `2xx` | 投递成功，不再重试 |
| `4xx` / `5xx` / 超时 / 连接失败 | 记为 `fail`，按退避策略**自动重试**；查询接口会给出下次重试时间 |

每次投递的状态可通过 API 查询（运维/排障用）：

```http
GET /api/webhooks/messages/{messageId}/attempts
→ [{"endpointId":"ep_…","url":"https://…","status":"success"}]
→ [{"endpointId":"ep_…","url":"https://…","status":"fail","nextAttempt":"2026-09-14T05:42:00Z"}]
```

`status` 取值为可读文本（`success` / `pending` / `fail`）。

**生产实测**（同一事件 fan-out 到两个目标）：
- 目标返回 `200` → `status: success`
- 目标返回 `404` → `status: fail` + `nextAttempt`（已排定重试）

## 6. 幂等与顺序

- **幂等**：同一 `webhook-id` 可能被投递多次（at-least-once）。用 `webhook-id` 建唯一索引或去重表，重复时直接返回 2xx。
- **顺序**：不保证。若业务关心先后，用 `webhook-timestamp` 或事件里的业务字段做判定，不要依赖到达顺序。

## 7. 密钥轮换

`webhook-signature` 可包含**多枚**签名（空格分隔），用于轮换期平滑过渡：客户端应"**任一匹配即通过**"。轮换流程：先让接收方同时接受新旧两把密钥，再在平台侧更新目标密钥，最后移除旧密钥。

## 8. 内网收端与 SSRF 防护

生产环境开启了严格 SSRF 防护（`SVIX_WHITELIST_SUBNETS=[]`）：**发往私网地址的目标会被拒绝**（实测：指向内网容器的投递被拦）。因此：

- 接收方必须是**公网可达**地址；
- 若确有内网收端需求，需要在平台侧显式把该网段加入白名单——**这会削弱防护**，请评估后再定。

## 9. 本地联调

本仓库提供了参考收端与脚本：

```bash
# 起一个自带验签的收端（会打印 signature-valid）
python3 platform-service/scripts/mock-receiver.py --port 5909 --out /tmp/recv.log

# 注册目标（返回 secret，把它传给收端：RECEIVER_WEBHOOK_SECRET=whsec_…）
bash platform-service/scripts/provision-destination.sh <tenant> http://host.docker.internal:5909/catch "job.completed"

# 发布事件
python3 platform-service/scripts/publish.py --tenant <tenant> --event-type job.completed --payload '{"jobId":"1"}'

# 全链路冒烟（建目标 → 验签收端 → 发布 → 验签）
bash platform-service/scripts/webhook-smoke.sh
```

## 10. 可选增强（需平台侧小改动）

当前接口只提供**签名密钥**这一种鉴权。如果你的安全要求还包含以下任一项，可以加（服务侧改动很小，半天内）：

1. **自定义认证头**：为目标配置一个静态头（如 `Authorization: Bearer xxx`），服务在投递时附带；
2. **来源 IP 白名单**：平台侧固定出口 IP 后，你在防火墙上放行；
3. **不对称签名（ed25519）**：你只需持有公钥，平台持有私钥——避免接收方持有可伪造的对称密钥。

## 11. 常见集成错误

| 症状 | 原因 |
|---|---|
| 验签必然失败 | 用**反序列化后再序列化**的 body 计算签名；必须用原始字节 |
| 偶发验签失败 | 只比对了 `webhook-signature` 的第一枚（轮换期有多枚）；应遍历全部 |
| 重放攻击窗口 | 没有校验 `webhook-timestamp` 容差 |
| 重复处理业务 | 没有按 `webhook-id` 幂等 |
| 投递超时被重试 | 在 HTTP 处理里同步做了慢操作（应快速 2xx，异步处理） |
