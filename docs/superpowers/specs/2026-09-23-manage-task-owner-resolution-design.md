# manage_task create 的 owner 解析与纠错设计

- 日期：2026-09-23
- 状态：Draft（设计已定稿，待实现）
- 范围：上游 `agentic/packages/core/src/middlewares/taskMiddleware.ts`（含 prompt 文案、单测）；本仓库 `fina_demo/agent` 仅 bump `@axiom-lattice/core` 版本并重新部署
- 前置（不在本仓库代码内）：`agentic` 发布包含本修复的 core 版本

## 1. 背景与目标

深智元租户 eval 运行中出现如下失败，导致整条用例被判定失败（`lattice_eval_run_results.error`）：

```
null value in column "owner_id" of relation "lattice_tasks" violates not-null constraint
```

该错误在 eval 结果里出现 8 次（另有 2 次 `manage_task` 入参 schema 不匹配、1 次 recursion、1 次 capability runtime），且同一 case 时过时挂（模型是否/何时调用 `manage_task create` 非确定）。

目标：

1. **默认不报错**：在 agent 运行时（有 `assistant_id`、无 `user_id`）省略 `ownerType` 时，自动推断为 agent 并正确填充 `ownerId`。
2. **错误可纠正**：当确实无法解析 owner 时，返回 agent 可读并自行纠正的**工具结果**（`{success:false, code, error, hint}`），**不再抛异常**打断整个 run。
3. 修正误导性的 owner 默认文案。

## 2. 现状与根因

`manage_task` create 分支（`packages/core/src/middlewares/taskMiddleware.ts`）：

- `ownerId` 兜底链（`:1016-1019`）：
  ```ts
  const ownerId = input.ownerId
    || trustedScope?.assistantId
    || (input.ownerType === "agent" ? rc.assistant_id : null)
    || rc.user_id;
  ```
  只有当 `ownerType` **显式**为 `"agent"` 时才取 `rc.assistant_id`。
- `effectiveOwnerType`（`:1094-1096`）：`input.ownerType || "user"` —— 省略即 `"user"`。
- `effectiveOwnerId`（`:1097`）：非委派路径直接用上面的 `ownerId`。
- `store.create` 前**没有** `effectiveOwnerId` 非空校验（`:1169-1172`），空值直达 DB → NOT NULL 崩溃。
- 文案误导：
  - schema（`:771`）：`Owner type. Defaults to 'user' if omitted`
  - 工具描述（`:2488`）：`No ownerType/ownerId: auto-assigned to current user`
  - 中间件注入 prompt（`:2468-2471`）：`### Ownership defaults - No params: ownerType defaults to "user" with current user's ID`

eval 的 runConfig（core `LatticeEval.executeAgentStep`）只注入 `assistant_id`，没有 `user_id`，于是「省略 → 归当前 user」的规则必然产出 `ownerId = null`。

## 3. 决策总览

| 项 | 决策 |
|---|---|
| 省略 `ownerType` 时 | 按运行时身份推断：有 `user_id`→`user`；否则有 `assistant_id`→`agent`；都没有→`user`（兼容旧默认） |
| `ownerId` 解析 | `input.ownerId` → `trustedScope?.assistantId` → （解析出的 type 为 agent 时）`rc.assistant_id` → `rc.user_id` |
| 解析失败 | 返回 `{success:false, code:"TASK_OWNER_REQUIRED", error, hint}`，**不调用 `store.create`、不抛异常** |
| trusted project | 行为不变（仍强制 agent owner，仍走 roster 校验） |
| 委派（delegated） | 行为不变（仍强制 agent + `rc.assistant_id`） |
| DB 约束 | 不改（`owner_id NOT NULL` 保留） |
| 文案 | 更新 schema/工具描述/prompt 的 Ownership defaults，描述运行时推断 |

## 4. 详细设计

### 4.1 ownerType 推断

```ts
const runtimeOwnerType: 'user' | 'agent' =
  (rc.user_id as string | undefined) ? 'user'
  : (rc.assistant_id as string | undefined) ? 'agent'
  : 'user';

const requestedOwnerType = (input.ownerType as 'user' | 'agent' | undefined) ?? runtimeOwnerType;
```

- 显式 `input.ownerType` 优先；
- 无显式值时，用户态运行（有 `user_id`）仍归 user，agent 态运行归 agent；
- 两者都无时保持 `"user"`，行为与旧版一致（后续由 4.3 兜底拦截）。

### 4.2 ownerId 解析

```ts
const ownerId = input.ownerId
  || trustedScope?.assistantId
  || (requestedOwnerType === 'agent' ? (rc.assistant_id as string | undefined) : undefined)
  || (rc.user_id as string | undefined);
```

`effectiveOwnerType` / `effectiveOwnerId` 沿用现有委派与 trusted 分支覆盖逻辑，仅将「`input.ownerType || "user"`」替换为「`requestedOwnerType`」，并将上面的 `ownerId` 作为非委派路径的值。

### 4.3 兜底：可纠正的工具错误

在 `store.create` 之前、`effectiveOwnerType`/`effectiveOwnerId` 计算之后，新增守卫：

```ts
if (typeof effectiveOwnerId !== 'string' || effectiveOwnerId.length === 0) {
  return JSON.stringify({
    success: false,
    code: 'TASK_OWNER_REQUIRED',
    error: 'Cannot resolve task owner: no user or agent identity in this run.',
    hint: 'Retry with ownerType:"agent" (or provide an explicit ownerId).',
  });
}
```

- 返回的是工具观察文本，模型可在同一 run 内按 `hint` 重试，而不是整步抛异常。
- 不覆盖委派/trusted 分支已有的校验（它们继续先行返回各自错误）。

### 4.4 文案修正

- schema（`:771`）：`Owner type. Defaults to the runtime actor (agent runtime → "agent", user runtime → "user").`
- 工具描述（`:2487-2490`）与 prompt（`:2468-2471`）：把「auto-assigned to current user」改为按运行时身份推断的说明，并明确 agent 运行时省略即归当前 agent。

## 5. 非目标

- 不修改 `lattice_tasks.owner_id NOT NULL` 或其它 DB 约束。
- 不修改 trusted-project / delegated 的 owner 策略与权限校验。
- 不修改 `update` / `set_status` / `add_activity` / `delete` 的既有 owner 语义（普通任务本就按 workspace/project 作用域放行）。
- 不修改 eval runner 注入 `user_id`（运行时推断已覆盖；如需可另案）。

## 6. 测试

`packages/core/src/middlewares/__tests__/taskMiddleware.test.ts` 新增：

1. 省略 `ownerType` + 仅有 `assistant_id` → `store.create` 收到 `ownerType:"agent"`、`ownerId:<assistant_id>`，返回 success。
2. 省略 `ownerType` + 仅有 `user_id` → `ownerType:"user"`、`ownerId:<user_id>`。
3. 省略 `ownerType` + 二者皆无 → 返回 `TASK_OWNER_REQUIRED`，且 `store.create` **未被调用**（用现有 `mockStore.create` 断言）。
4. 显式 `ownerType:"user"` + 显式 `ownerId` + 有 `assistant_id` → 尊重显式值（`ownerType:"user"`，`ownerId` 为给定值，不被 `assistant_id` 覆盖）。

沿用现有 mock store / `createdMockTasks` 断言风格；运行 `pnpm --filter @axiom-lattice/core test -- taskMiddleware`。

## 7. 交付与部署

1. 在 `agentic` 实现 + 单测 + 通过 core 的 `tsc --noEmit` 与测试。
2. 发布新 core 版本。
3. 在 `fina_demo/agent` bump `@axiom-lattice/core`（`pnpm up`）并重新构建镜像/部署。
4. 回归：对深智元 `agent-638cpl`（Nova）重跑原失败 case，确认不再出现 `owner_id` NOT NULL。
