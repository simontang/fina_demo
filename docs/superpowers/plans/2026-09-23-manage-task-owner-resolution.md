# manage_task Owner Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `manage_task create` infer the task owner from the runtime identity and return a correctable tool error instead of crashing on the `lattice_tasks.owner_id` NOT NULL constraint.

**Architecture:** Fix the upstream `@axiom-lattice/core` task middleware. When `ownerType` is omitted, infer it from the run config (`user_id` → user, else `assistant_id` → agent, else user). Resolve `ownerId` against the resolved type. Guard the create path so an unresolvable owner returns `{success:false, code:"TASK_OWNER_REQUIRED", hint}` as a tool observation rather than throwing. Then update the misleading "ownership defaults" wording and bump core in `fina_demo/agent`.

**Tech Stack:** TypeScript, Zod, Jest (ts-jest), `@axiom-lattice/core`, `@axiom-lattice/pg-stores`.

**Repos:**
- Code change: `~/code/agentic` (GitHub `TZWNCC/agentic`), package `packages/core`.
- Consumer bump: `/Users/simon/code/fina_demo/agent` (this repo).

**Spec:** `docs/superpowers/specs/2026-09-23-manage-task-owner-resolution-design.md`

---

## File Structure

- Modify: `~/code/agentic/packages/core/src/middlewares/taskMiddleware.ts`
  - owner inference + resolution in the `manage_task` handler (around lines 1016-1019 and 1094-1097)
  - corrective guard before `store.create` (after the trusted-project block, ~line 1120)
  - wording: schema `:771`, tool description `:2487-2490`, injected prompt `:2468-2471`
- Test: `~/code/agentic/packages/core/src/middlewares/__tests__/taskMiddleware.test.ts`
  - extend the owner-defaults `describe` block (after the `should respect explicit ownerId` test, ~line 4341)
- Modify (bump only): `/Users/simon/code/fina_demo/agent/package.json`

---

## Task 1: Failing tests for owner inference + corrective error

**Files:**
- Test: `~/code/agentic/packages/core/src/middlewares/__tests__/taskMiddleware.test.ts` (insert after the `should respect explicit ownerId` test, before `describe('structured belief completion'`)

- [ ] **Step 1: Add the three failing tests**

Insert this block right after the existing test that ends with `ownerType: 'agent', ownerId: 'agent-42'` assertions:

```ts
    it('should infer agent owner when only assistant_id is present', async () => {
      const tool = getManageTaskTool(middleware);
      const config = makeConfig({ tenantId: 't-1', assistant_id: 'agent-638cpl' });
      await tool.invoke({ action: 'create', title: 'T' }, config);
      expect(mockStore.create).toHaveBeenCalledWith(
        expect.objectContaining({ ownerType: 'agent', ownerId: 'agent-638cpl' })
      );
    });

    it('should return a corrective error when no runtime identity is available', async () => {
      const tool = getManageTaskTool(middleware);
      const config = makeConfig({ tenantId: 't-1' });
      const result = JSON.parse(await tool.invoke({ action: 'create', title: 'T' }, config) as string);
      expect(result).toMatchObject({ success: false, code: 'TASK_OWNER_REQUIRED' });
      expect(result.hint).toContain('ownerType');
      expect(mockStore.create).not.toHaveBeenCalled();
    });

    it('should respect explicit ownerType and ownerId even when assistant_id is present', async () => {
      const tool = getManageTaskTool(middleware);
      const config = makeConfig({ tenantId: 't-1', assistant_id: 'agent-1' });
      await tool.invoke({ action: 'create', title: 'T', ownerType: 'user', ownerId: 'user-7' }, config);
      expect(mockStore.create).toHaveBeenCalledWith(
        expect.objectContaining({ ownerType: 'user', ownerId: 'user-7' })
      );
    });
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (from `~/code/agentic`):
```bash
pnpm --filter @axiom-lattice/core test -- taskMiddleware
```
Expected: the first test FAILS (`ownerType: 'user'` / `ownerId: undefined` instead of agent), the second FAILS (create was called / DB-shape assertion), the third may pass. Existing tests still pass.

- [ ] **Step 3: Commit the failing tests**

```bash
cd ~/code/agentic
git add packages/core/src/middlewares/__tests__/taskMiddleware.test.ts
git commit -m "test(core): cover manage_task owner inference and corrective error"
```

---

## Task 2: Infer ownerType and resolve ownerId from the runtime actor

**Files:**
- Modify: `~/code/agentic/packages/core/src/middlewares/taskMiddleware.ts:1016-1019`

- [ ] **Step 1: Replace the ownerId fallback with runtime-aware resolution**

Replace:
```ts
          const ownerId = input.ownerId
            || trustedScope?.assistantId
            || (input.ownerType === "agent" ? (rc.assistant_id as string) : null)
            || (rc.user_id as string);
```
with:
```ts
          // Resolve the effective owner type from the runtime actor when the
          // caller omits it: a user-invoked run owns "user", an agent-invoked run
          // (eval, dispatch, project work) owns "agent".
          const runtimeOwnerType: 'user' | 'agent' =
            (rc.user_id as string | undefined) ? 'user'
              : (rc.assistant_id as string | undefined) ? 'agent'
                : 'user';
          const requestedOwnerType: 'user' | 'agent' =
            (input.ownerType as 'user' | 'agent' | undefined) ?? runtimeOwnerType;

          const ownerId = input.ownerId
            || trustedScope?.assistantId
            || (requestedOwnerType === 'agent' ? (rc.assistant_id as string) : null)
            || (rc.user_id as string);
```

- [ ] **Step 2: Use `requestedOwnerType` for the effective owner type**

In the create branch (around line 1094), replace:
```ts
              const effectiveOwnerType = delegatedTaskId
                ? "agent"
                : (trustedProject ? "agent" : (input.ownerType || "user")) as 'user' | 'agent';
```
with:
```ts
              const effectiveOwnerType = delegatedTaskId
                ? "agent"
                : (trustedProject ? "agent" : requestedOwnerType) as 'user' | 'agent';
```
Leave the next line (`const effectiveOwnerId = delegatedTaskId ? (rc.assistant_id as string | undefined) : ownerId;`) unchanged.

- [ ] **Step 3: Run the owner tests**

Run (from `~/code/agentic`):
```bash
pnpm --filter @axiom-lattice/core test -- taskMiddleware
```
Expected: "should infer agent owner when only assistant_id is present" PASSES; "should default ownerType to user and ownerId from runConfig" and "should use assistant_id for agent-owned tasks" still PASS.

- [ ] **Step 4: Commit**

```bash
cd ~/code/agentic
git add packages/core/src/middlewares/taskMiddleware.ts
git commit -m "fix(core): infer manage_task owner from runtime actor when ownerType omitted"
```

---

## Task 3: Corrective guard instead of a DB constraint crash

**Files:**
- Modify: `~/code/agentic/packages/core/src/middlewares/taskMiddleware.ts` (insert after the trusted-project block, before the lifecycle-evidence check at ~line 1121)

- [ ] **Step 1: Add the guard**

Immediately after the trusted-project block (the `if (trustedProject) { ... }` that ends before the comment `// Lifecycle evidence (result/failureReason) is governed for agent tasks`), insert:

```ts
              // Never let an unresolvable owner reach the store: the DB enforces
              // owner_id NOT NULL and would throw, aborting the whole run. Return a
              // tool observation the model can correct on its next call instead.
              if (!delegatedTaskId && (typeof effectiveOwnerId !== 'string' || effectiveOwnerId.length === 0)) {
                return JSON.stringify({
                  success: false,
                  code: 'TASK_OWNER_REQUIRED',
                  error: 'Cannot resolve task owner: no user or agent identity in this run.',
                  hint: 'Retry with ownerType:"agent" (or provide an explicit ownerId).',
                });
              }
```

- [ ] **Step 2: Run the owner tests**

Run (from `~/code/agentic`):
```bash
pnpm --filter @axiom-lattice/core test -- taskMiddleware
```
Expected: "should return a corrective error when no runtime identity is available" PASSES; `mockStore.create` is not called.

- [ ] **Step 3: Commit**

```bash
cd ~/code/agentic
git add packages/core/src/middlewares/taskMiddleware.ts
git commit -m "fix(core): return TASK_OWNER_REQUIRED instead of crashing on null owner"
```

---

## Task 4: Fix the misleading ownership-defaults wording

**Files:**
- Modify: `~/code/agentic/packages/core/src/middlewares/taskMiddleware.ts:771`, `:2468-2471`, `:2487-2490`

- [ ] **Step 1: Update the zod schema description (line 771)**

Replace:
```ts
  ownerType: z.enum(["user", "agent"]).optional().describe("Owner type. Defaults to 'user' if omitted"),
```
with:
```ts
  ownerType: z.enum(["user", "agent"]).optional().describe("Owner type. Defaults to the runtime actor: agent runtime (no user identity) -> 'agent', user runtime -> 'user'."),
```

- [ ] **Step 2: Update the injected prompt block (lines 2468-2471)**

Replace:
```ts
### Ownership defaults
- No params: ownerType defaults to "user" with current user's ID
- ownerType="agent": auto-fills ownerId from current agent (subtask for yourself)
- Explicit ownerId: assign to a specific agent or user`;
```
with:
```ts
### Ownership defaults
- No params: ownerType defaults to the runtime actor (user runtime -> current user's ID; agent runtime -> current agent's ID)
- ownerType="agent": auto-fills ownerId from the current agent (subtask for yourself)
- Explicit ownerId: assign to a specific agent or user`;
```

- [ ] **Step 3: Update the tool description block (lines 2487-2490)**

Replace:
```ts
## Owner defaults
- No ownerType/ownerId: auto-assigned to current user
- ownerType="agent" without ownerId: auto-assigned to current agent
- Explicit ownerId: assign to a specific agent (cross-agent delegation)
```
with:
```ts
## Owner defaults
- No ownerType/ownerId: auto-assigned to the runtime actor (user runtime -> current user; agent runtime -> current agent)
- ownerType="agent" without ownerId: auto-assigned to current agent
- Explicit ownerId: assign to a specific agent (cross-agent delegation)
```

- [ ] **Step 4: Run the tests**

Run (from `~/code/agentic`):
```bash
pnpm --filter @axiom-lattice/core test -- taskMiddleware
```
Expected: PASS (no test asserts the old wording).

- [ ] **Step 5: Commit**

```bash
cd ~/code/agentic
git add packages/core/src/middlewares/taskMiddleware.ts
git commit -m "docs(core): describe runtime-based owner defaults for manage_task"
```

---

## Task 5: Full core verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full core test suite**

Run (from `~/code/agentic`):
```bash
pnpm --filter @axiom-lattice/core test
```
Expected: all suites pass. If any pre-existing test asserted the old "assistant_id-only → user" default, update it to the new semantics and re-run.

- [ ] **Step 2: Lint and typecheck/build**

Run (from `~/code/agentic`):
```bash
pnpm --filter @axiom-lattice/core lint
pnpm --filter @axiom-lattice/core build
```
Expected: no lint errors; `tsup` build succeeds.

- [ ] **Step 3: Commit any fixes from Step 1/2**

```bash
cd ~/code/agentic
git add -A packages/core
git commit -m "chore(core): satisfy lint/build after owner resolution fix"
```

---

## Task 6: Consume the fix in fina_demo/agent

**Files:**
- Modify: `/Users/simon/code/fina_demo/agent/package.json` (`@axiom-lattice/core` version)
- Possibly regenerate `/Users/simon/code/fina_demo/agent/pnpm-lock.yaml`

- [ ] **Step 1: Bump core to the released version**

Only after the fixed core is published to the registry used by this repo (or linked via workspace). Run:
```bash
cd /Users/simon/code/fina_demo/agent
pnpm up @axiom-lattice/core@<released-version>
```
Expected: `package.json` and `pnpm-lock.yaml` update to the fixed version.

- [ ] **Step 2: Verify the fixed code is installed**

Run:
```bash
grep -n "TASK_OWNER_REQUIRED" /Users/simon/code/fina_demo/agent/node_modules/@axiom-lattice/core/dist/index.mjs
```
Expected: a match (confirms the installed dist contains the fix).

- [ ] **Step 3: Build the agent**

Run:
```bash
cd /Users/simon/code/fina_demo/agent
pnpm build
```
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
cd /Users/simon/code/fina_demo
git add agent/package.json agent/pnpm-lock.yaml
git commit -m "chore(agent): bump @axiom-lattice/core for manage_task owner fix"
```

- [ ] **Step 5: Deploy and regression-check**

Deploy via the repo's normal flow (e.g. `deploy-to-214.sh` / docker compose). Then re-run the previously failing 深智元 eval cases for `agent-638cpl` (Nova), e.g. case `f56bf227-1ada-41d9-9340-de843b7853d8`, and confirm no `owner_id` NOT NULL errors:
```sql
SELECT count(*) FROM lattice_eval_run_results res
JOIN lattice_eval_runs r ON r.id = res.run_id
WHERE r.tenant_id = 'shenzhiyuan' AND res.error LIKE '%owner_id%not-null%'
  AND res.created_at > now() - interval '1 day';
```
Expected: `0`.

---

## Self-Review

**1. Spec coverage:**
- 运行时推断 ownerType → Task 2.
- ownerId 按 type 解析 → Task 2.
- 可纠正工具错误（不抛异常）→ Task 3.
- 文案修正 → Task 4.
- 单测 4 项 → Task 1 (3 new) + existing `should default ownerType to user and ownerId from runConfig` covers the user_id case.
- 交付/部署 → Task 6.

**2. Placeholder scan:** Task 6 Step 1 uses `<released-version>` because the publish step is external to these repos; this is an intentional external dependency, not an implementation placeholder.

**3. Type consistency:** `requestedOwnerType` / `runtimeOwnerType` typed `'user' | 'agent'`; `effectiveOwnerType` reuses `requestedOwnerType`; `effectiveOwnerId` unchanged (`string | undefined`); guard checks `typeof effectiveOwnerId !== 'string'`.
