### 1. 分群定义

**基础路径**: `/api/v1/segment-definitions`

#### 1.1 查询分群定义列表

```http
GET /api/v1/segment-definitions
```

**请求头**:


| 头部            | 说明                    |
| ------------- | --------------------- |
| `X-Tenant-Id` | 租户标识（可选，默认 `default`） |


**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": [
    {
      "id": 1,
      "tenantId": "default",
      "name": "高价值客户",
      "description": "近30天消费超过5000元的客户",
      "datasourceId": 1,
      "querySql": "SELECT customer_id, SUM(amount) AS total FROM orders ...",
      "status": 1,
      "createdAt": "2026-07-13 10:00:00",
      "updatedAt": "2026-07-13 10:00:00"
    }
  ]
}
```

#### 1.1.1 分页查询分群定义

分页查询用于需要按页浏览分群定义的界面，原列表接口保持不变。

```http
GET /api/v1/segment-definitions/page?page=1&pageSize=20&keyword=customer
```

`page` 默认值为 `1`，`pageSize` 默认值为 `20`、最大值为 `200`。可选的 `keyword` 会在去除首尾空格后对名称进行不区分大小写的包含搜索；空字符串等同于不搜索，`%`、`_` 和 `\\` 按普通字符处理。结果按 `updatedAt`、`id` 倒序排列。

**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [],
    "total": 0,
    "page": 1,
    "pageSize": 20
  }
}
```



#### 1.2 查询单个分群定义

```http
GET /api/v1/segment-definitions/{id}
```

**路径参数**:


| 参数   | 类型     | 说明      |
| ---- | ------ | ------- |
| `id` | `Long` | 分群定义 ID |




#### 1.3 创建分群定义

```http
POST /api/v1/segment-definitions
Content-Type: application/json
X-Tenant-Id: default
```

**请求体**:


| 字段             | 类型       | 必填  | 说明                                    |
| -------------- | -------- | --- | ------------------------------------- |
| `name`         | `string` | ✅   | 分群名称                                  |
| `description`  | `string` | —   | 分群描述                                  |
| `datasourceId` | `Long`   | ✅   | 关联数据源 ID（对应 `t_datasource_config.id`） |
| `querySql`     | `string` | ✅   | 分群 SQL（仅允许 SELECT / WITH）             |
| `status`       | `int`    | —   | 状态：1-启用，0-禁用（默认 1）                    |


**SQL 安全规则**:

- 必须以 `SELECT` 或 `WITH` 开头
- 禁止包含分号（单条语句）
- 禁止写操作/DDL 关键字：`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, `MERGE`, `CALL`, `GRANT`, `REVOKE`, `VACUUM`, `ANALYZE`

**请求示例**:

```json
{
  "name": "高价值客户",
  "description": "近30天消费超过5000元的客户",
  "datasourceId": 1,
  "querySql": "SELECT customer_id, SUM(amount) AS total FROM orders WHERE created_at > NOW() - INTERVAL '30 days' GROUP BY customer_id HAVING SUM(amount) > 5000",
  "status": 1
}
```

**响应**: 返回创建成功的 `SegmentDefinitionVO` 对象。

#### 1.4 更新分群定义

```http
PUT /api/v1/segment-definitions/{id}
Content-Type: application/json
```

请求体字段同创建接口。

#### 1.5 删除分群定义

```http
DELETE /api/v1/segment-definitions/{id}
```

逻辑删除（`deleted = 1`），关联的分群数据不会级联删除。

**响应**:

```json
{
  "code": 200,
  "message": "success"
}
```



#### 1.6 执行分群处理

```http
POST /api/v1/segment-definitions/{id}/process
Content-Type: application/json
```

这是 CDP 服务最核心的操作：连接该分群关联的 CDP 数据源，执行 `querySql`，将结果集序列化为 JSON 保存到 `t_segment_data` 表。

**路径参数**:


| 参数   | 类型     | 说明      |
| ---- | ------ | ------- |
| `id` | `Long` | 分群定义 ID |


**请求体** (可选):


| 字段       | 类型                    | 说明                                    |
| -------- | --------------------- | ------------------------------------- |
| `params` | `Map<String, Object>` | SQL 命名参数，用于替换 SQL 中的 `:paramName` 占位符 |


**请求示例**:

```json
{
  "params": {
    "min_amount": 5000,
    "start_date": "2026-06-01"
  }
}
```

对应的 SQL 可使用命名参数：

```sql
SELECT customer_id, SUM(amount) AS total
FROM orders
WHERE created_at >= :start_date::date
GROUP BY customer_id
HAVING SUM(amount) > :min_amount
```

**响应**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": 1,
    "tenantId": "default",
    "definitionId": 1,
    "runId": "6a2f41a3-8b7c-4d1e-9f5a-3c8d2e1b0f4a",
    "dataJson": "[{\"customer_id\":\"C001\",\"total\":6200.00},{\"customer_id\":\"C002\",\"total\":5800.00}]",
    "rowCount": 2,
    "createdAt": "2026-07-13 10:05:00",
    "updatedAt": "2026-07-13 10:05:00"
  }
}
```

`runId` 为本次执行的唯一标识（UUID），可关联到具体的分群数据快照。

---



### 2. 分群数据

**基础路径**: `/api/v1/segment-data`

#### 2.1 查询分群数据列表（分页）

```http
GET /api/v1/segment-data?definitionId=1&page=1&pageSize=20
```

**查询参数**:


| 参数             | 类型     | 必填  | 说明          |
| -------------- | ------ | --- | ----------- |
| `definitionId` | `Long` | —   | 按分群定义 ID 过滤 |
| `page`         | `int`  | —   | 页码（默认 1）    |
| `pageSize`     | `int`  | —   | 每页条数（默认 20） |


**响应**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [
      {
        "id": 1,
        "tenantId": "default",
        "definitionId": 1,
        "runId": "6a2f41a3-8b7c-4d1e-9f5a-3c8d2e1b0f4a",
        "dataJson": "[{\"customer_id\":\"C001\",\"total\":6200.00}]",
        "rowCount": 2,
        "createdAt": "2026-07-13 10:05:00",
        "updatedAt": "2026-07-13 10:05:00"
      }
    ],
    "total": 1,
    "page": 1,
    "pageSize": 20
  }
}
```



#### 2.2 查询单条分群数据

```http
GET /api/v1/segment-data/{id}
```



#### 2.3 手动创建分群数据

```http
POST /api/v1/segment-data
Content-Type: application/json
```

**请求体**:


| 字段             | 类型       | 必填  | 说明               |
| -------------- | -------- | --- | ---------------- |
| `definitionId` | `Long`   | ✅   | 所属分群定义 ID        |
| `runId`        | `string` | —   | 执行标识             |
| `dataJson`     | `string` | ✅   | 分群结果 JSON（字符串格式） |




#### 2.4 更新分群数据

```http
PUT /api/v1/segment-data/{id}
Content-Type: application/json
```

请求体字段同创建接口。

#### 2.5 删除分群数据

```http
DELETE /api/v1/segment-data/{id}
```

逻辑删除。

---

### 3. 营销活动

**基础路径**: `/api/v1/marketing-campaigns`

营销活动是多租户对象，所有接口都通过 `X-Tenant-Id` 隔离数据。复杂策略字段使用 JSON object 表达，服务端以 TEXT 保存。

#### 3.1 查询营销活动列表

```http
GET /api/v1/marketing-campaigns?type=reactivation&status=scheduled&page=1&pageSize=20&keyword=dormant
X-Tenant-Id: default
```

**查询参数**:

| 参数       | 类型       | 必填 | 说明 |
| ---------- | ---------- | ---- | ---- |
| `type`     | `string`   | —    | 按活动类型过滤，如 `reactivation` |
| `status`   | `string`   | —    | 按状态过滤：`draft`, `scheduled`, `running`, `stopped`, `completed` |
| `page`     | `int`      | —    | 页码，默认 1 |
| `pageSize` | `int`      | —    | 每页条数，默认 20，最大 200 |
| `keyword`  | `string`   | —    | 名称关键词，不区分大小写；首尾空格会被忽略，SQL 通配符按普通字符处理 |

#### 3.2 创建营销活动

```http
POST /api/v1/marketing-campaigns
Content-Type: application/json
X-Tenant-Id: default
```

**最小请求示例**:

```json
{
  "name": "沉睡会员唤醒",
  "description": "120 天未购会员唤醒活动",
  "type": "reactivation",
  "status": "draft",
  "goal": "提升沉睡会员 30 天复购率",
  "startTime": "2026-07-20 09:00:00",
  "endTime": "2026-08-03 23:59:59",
  "mainSegmentDataId": 12
}
```

完整策略 JSON 结构见 3.7。

**字段说明**:

| 字段 | 类型 | 必填 | 说明 |
| ---- | ---- | ---- | ---- |
| `name` | `string` | ✅ | 活动名称 |
| `type` | `string` | ✅ | 活动类型，自由字符串 |
| `goal` | `string` | ✅ | 活动目标 |
| `startTime` | `datetime` | ✅ | 开始时间 |
| `endTime` | `datetime` | ✅ | 结束时间，必须晚于 `startTime` |
| `mainSegmentDataId` | `Long` | — | 主人群包，对应同租户 `segment_data.id` |
| `status` | `string` | — | 默认 `draft` |
| `*Strategy`, `statistics` | `object` | — | 策略或统计 JSON，默认 `{}` |

#### 3.3 查询、更新、删除单个活动

```http
GET /api/v1/marketing-campaigns/{id}
PUT /api/v1/marketing-campaigns/{id}
DELETE /api/v1/marketing-campaigns/{id}
```

`PUT` 请求体同创建接口。`DELETE` 为逻辑删除。

#### 3.4 启动活动

```http
POST /api/v1/marketing-campaigns/{id}/start
```

允许从 `draft` 或 `scheduled` 启动，成功后状态变为 `running`，并写入 `actualStartedAt`。如果 `endTime` 已过期会返回冲突错误。

#### 3.5 停止活动

```http
POST /api/v1/marketing-campaigns/{id}/stop
```

允许从 `scheduled` 或 `running` 停止，成功后状态变为 `stopped`，并写入 `actualStoppedAt`。

#### 3.6 定时活动

```http
POST /api/v1/marketing-campaigns/{id}/schedule
Content-Type: application/json
```

请求体可为空，也可以覆盖活动时间：

```json
{
  "startTime": "2026-07-20 09:00:00",
  "endTime": "2026-08-03 23:59:59"
}
```

成功后状态变为 `scheduled`。`cdp-service` 内置调度器默认每 60 秒自动执行：

- `scheduled` 且 `startTime <= now < endTime` → `running`
- `running` 且 `endTime <= now` → `completed`

可通过环境变量调整：

- `CAMPAIGN_SCHEDULER_ENABLED=false`
- `CAMPAIGN_SCHEDULER_FIXED_DELAY_MS=60000`

#### 3.7 推荐 Campaign JSON 结构 v1

服务端 v1 只校验这些字段是合法 JSON，并把它们保存到 `TEXT` 列；内部结构建议按下面的约定统一，方便后续 agent、投放服务和统计服务读取。

**核心引用关系**:

| Key | 定义位置 | 被谁引用 | 用途 |
| --- | --- | --- | --- |
| `subSegmentKey` | `segmentationStrategy.subSegments[].subSegmentKey` | control/channel/offer/wave/AB/statistics | 活动内子人群包的稳定业务 key |
| `channelKey` | `contentChannelStrategy.channels[].channelKey` | `waveStrategy.waves[].channelKeys`, `abTestStrategy.variants[].channelKey` | 渠道和模板组合的稳定 key |
| `offerCode` | `offerStrategy.offers[].offerCode` | `waveStrategy.waves[].offerCodes`, `abTestStrategy.variants[].offerCode` | 权益策略的稳定 key |
| `waveId` | `waveStrategy.waves[].waveId` | 后续 wave 的 `entryRule.fromWaveIds` | 多波段之间的行为依赖 |
| `variantId` | `abTestStrategy.variants[].variantId` | `statistics.abTest.variantMetrics[]` | A/B 测试分组 key |

**字段设计建议**:

| 字段 | 建议结构 | 说明 |
| --- | --- | --- |
| `segmentationStrategy` | `source`, `subSegments`, `assignment`, `exclusions` | `source.segmentDataId` 对应主人群包快照；`subSegments` 在主人群包内继续切分 |
| `controlGroupStrategy` | `enabled`, `ratio`, `method`, `unit`, `stratifyBy`, `excludeFromWaves` | 建议 `stratifyBy` 包含 `subSegmentKey`，保证每个子人群内都留 control group |
| `contentChannelStrategy` | `channels[]` | 每个 channel 用 `eligibleSubSegmentKeys` 声明适用子人群 |
| `offerStrategy` | `offers[]`, `allocation` | 每个 offer 用 `eligibleSubSegmentKeys` 声明适用子人群 |
| `waveStrategy` | `waves[]` | 每波用 `eligibleSubSegmentKeys`, `channelKeys`, `offerCodes` 组合人群、渠道和权益 |
| `abTestStrategy` | `scope`, `variants[]`, `winnerPolicy` | `scope.subSegmentKeys` 限定哪些子人群进入实验 |
| `statistics` | `audience`, `delivery`, `conversion`, `revenue`, `abTest` | v1 只保存统计快照，不自动计算 |

下面是一个“沉睡会员唤醒”活动的完整 JSON。这个例子里，主人群包来自 `mainSegmentDataId=101`；活动内部再切成高价值和中价值两个子人群；control group 在每个子人群内各抽 10%；高价值人群使用更高优惠并进入 A/B 文案测试；第二波只触达第一波未响应的人。

同一份示例也保存为 [marketing-campaign-reactivation-example.json](/Users/cid/gitlab/fina_demo/docs/api/marketing-campaign-reactivation-example.json)。

```json
{
  "name": "沉睡会员唤醒-2026-07",
  "description": "对 120 天未购会员做两波唤醒，高价值人群测试两种短信文案。",
  "type": "reactivation",
  "status": "draft",
  "goal": "提升沉睡会员 30 天复购率，并评估优惠券增量效果。",
  "startTime": "2026-07-20 09:00:00",
  "endTime": "2026-08-03 23:59:59",
  "mainSegmentDataId": 101,
  "segmentationStrategy": {
    "version": "1.0",
    "audienceKey": "customer_id",
    "source": {
      "segmentDataId": 101,
      "segmentDefinitionId": 21,
      "runId": "seg_run_dormant_20260719",
      "description": "120 天未购且允许营销触达的会员"
    },
    "subSegments": [
      {
        "subSegmentKey": "hv_dormant",
        "name": "高价值沉睡会员",
        "priority": 1,
        "criteria": [
          {"field": "lifetime_value", "operator": ">=", "value": 10000},
          {"field": "days_since_last_purchase", "operator": ">=", "value": 120}
        ],
        "tags": ["high_value", "dormant"]
      },
      {
        "subSegmentKey": "mv_dormant",
        "name": "中价值沉睡会员",
        "priority": 2,
        "criteria": [
          {"field": "lifetime_value", "operator": ">=", "value": 3000},
          {"field": "lifetime_value", "operator": "<", "value": 10000},
          {"field": "days_since_last_purchase", "operator": ">=", "value": 120}
        ],
        "tags": ["medium_value", "dormant"]
      }
    ],
    "assignment": {
      "mode": "first_match_by_priority",
      "fallbackSubSegmentKey": "mv_dormant"
    },
    "exclusions": [
      {"field": "marketing_consent", "operator": "=", "value": false},
      {"field": "unsubscribe_status", "operator": "=", "value": true}
    ]
  },
  "controlGroupStrategy": {
    "enabled": true,
    "method": "deterministic_hash_holdout",
    "unit": "customer",
    "ratio": 0.1,
    "seed": "campaign_id",
    "stratifyBy": ["subSegmentKey"],
    "excludeFromWaves": true
  },
  "contentChannelStrategy": {
    "version": "1.0",
    "defaultLocale": "zh-CN",
    "channels": [
      {
        "channelKey": "sms_primary",
        "channel": "sms",
        "templateKey": "sms_dormant_coupon_v1",
        "eligibleSubSegmentKeys": ["hv_dormant", "mv_dormant"],
        "sendWindow": {
          "timezone": "Asia/Shanghai",
          "start": "09:00",
          "end": "20:00"
        },
        "frequencyCap": {
          "maxMessages": 2,
          "windowDays": 7
        },
        "variables": ["customer_name", "offer_value", "expiry_date"]
      },
      {
        "channelKey": "wechat_fallback",
        "channel": "wechat",
        "templateKey": "wechat_dormant_reminder_v1",
        "eligibleSubSegmentKeys": ["hv_dormant", "mv_dormant"],
        "fallbackForChannelKeys": ["sms_primary"]
      }
    ]
  },
  "offerStrategy": {
    "version": "1.0",
    "budget": {
      "currency": "CNY",
      "maxTotalCost": 50000
    },
    "offers": [
      {
        "offerCode": "coupon_80_hv",
        "type": "coupon",
        "value": 80,
        "currency": "CNY",
        "validDays": 7,
        "eligibleSubSegmentKeys": ["hv_dormant"],
        "perCustomerLimit": 1
      },
      {
        "offerCode": "coupon_30_mv",
        "type": "coupon",
        "value": 30,
        "currency": "CNY",
        "validDays": 7,
        "eligibleSubSegmentKeys": ["mv_dormant"],
        "perCustomerLimit": 1
      }
    ],
    "allocation": {
      "method": "by_sub_segment",
      "rules": [
        {"subSegmentKey": "hv_dormant", "offerCode": "coupon_80_hv"},
        {"subSegmentKey": "mv_dormant", "offerCode": "coupon_30_mv"}
      ]
    }
  },
  "waveStrategy": {
    "enabled": true,
    "timezone": "Asia/Shanghai",
    "waves": [
      {
        "waveId": "wave_1",
        "name": "首轮触达",
        "scheduledAt": "2026-07-20 10:00:00",
        "eligibleSubSegmentKeys": ["hv_dormant", "mv_dormant"],
        "channelKeys": ["sms_primary"],
        "offerCodes": ["coupon_80_hv", "coupon_30_mv"],
        "entryRule": {
          "excludeGroups": ["control"]
        }
      },
      {
        "waveId": "wave_2",
        "name": "未响应二次触达",
        "scheduledAt": "2026-07-27 10:00:00",
        "eligibleSubSegmentKeys": ["hv_dormant", "mv_dormant"],
        "channelKeys": ["wechat_fallback"],
        "offerCodes": ["coupon_80_hv", "coupon_30_mv"],
        "entryRule": {
          "fromWaveIds": ["wave_1"],
          "excludeGroups": ["control"],
          "includeIf": [
            {"metric": "purchased", "operator": "=", "value": false},
            {"metric": "clicked", "operator": "=", "value": false}
          ]
        }
      }
    ]
  },
  "abTestStrategy": {
    "enabled": true,
    "scope": {
      "subSegmentKeys": ["hv_dormant"],
      "waveIds": ["wave_1"]
    },
    "unit": "customer",
    "primaryMetric": "repurchase_rate_14d",
    "variants": [
      {
        "variantId": "A",
        "name": "权益优先文案",
        "trafficRatio": 0.5,
        "channelKey": "sms_primary",
        "templateKey": "sms_hv_offer_first_v1",
        "offerCode": "coupon_80_hv"
      },
      {
        "variantId": "B",
        "name": "情感唤醒文案",
        "trafficRatio": 0.5,
        "channelKey": "sms_primary",
        "templateKey": "sms_hv_emotional_v1",
        "offerCode": "coupon_80_hv"
      }
    ],
    "winnerPolicy": {
      "method": "fixed_horizon",
      "minSampleSizePerVariant": 1000,
      "confidence": 0.95
    }
  },
  "statistics": {
    "version": "1.0",
    "lastComputedAt": null,
    "audience": {
      "targetCount": 0,
      "controlCount": 0,
      "treatmentCount": 0,
      "bySubSegment": {
        "hv_dormant": {"targetCount": 0, "controlCount": 0, "treatmentCount": 0},
        "mv_dormant": {"targetCount": 0, "controlCount": 0, "treatmentCount": 0}
      }
    },
    "delivery": {
      "sent": 0,
      "delivered": 0,
      "opened": 0,
      "clicked": 0,
      "failed": 0
    },
    "conversion": {
      "converted": 0,
      "conversionRate": null,
      "incrementalLift": null
    },
    "revenue": {
      "currency": "CNY",
      "grossRevenue": 0,
      "incrementalRevenue": null,
      "offerCost": 0,
      "grossMargin": null
    },
    "abTest": {
      "winnerVariantId": null,
      "variantMetrics": [
        {"variantId": "A", "sampleSize": 0, "conversionRate": null},
        {"variantId": "B", "sampleSize": 0, "conversionRate": null}
      ]
    }
  }
}
```

使用时建议按这个顺序理解：`mainSegmentDataId` 选主人群包，`segmentationStrategy.subSegments` 在主人群包内生成子人群，control group 在每个子人群内留出，渠道和 offer 通过 `eligibleSubSegmentKeys` 绑定子人群，wave 决定什么时候对哪些子人群用哪些渠道/offer，AB test 只在 `scope.subSegmentKeys` 指定的人群内分流。
