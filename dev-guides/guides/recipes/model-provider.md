# Recipe: Adding a Model Provider

Add a new LLM provider (e.g., Anthropic, Groq, Together AI).

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | `packages/protocols/src/ModelLatticeProtocol.ts` | Add provider enum value |
| 2 | `packages/core/src/model_lattice/ModelLattice.ts` | Add provider initialization case |
| 3 | `.env` | Add API key env var |

## Step 1: Add Provider to LLMConfig

File: `packages/protocols/src/ModelLatticeProtocol.ts`

```typescript
export interface LLMConfig {
  provider: "azure" | "openai" | "deepseek" | "siliconcloud" | "volcengine" | "anthropic"; // Add yours
  // ...
}
```

## Step 2: Add Initialization Case

File: `packages/core/src/model_lattice/ModelLattice.ts`

Find `initChatModel()` (not async, takes `config: LLMConfig` parameter, uses if/else if chain):

```typescript
private initChatModel(config: LLMConfig): BaseChatModel {
  if (config.provider === "azure") {
    // ... Azure setup
  } else if (config.provider === "openai") {
    // ... OpenAI setup
  } else if (config.provider === "deepseek") {
    // ... DeepSeek setup
  }
  // Add your provider:
  else if (config.provider === "anthropic") {
    const { ChatAnthropic } = require("@langchain/anthropic");
    return new ChatAnthropic({
      anthropicApiKey: process.env.ANTHROPIC_API_KEY || config.apiKey,
      model: config.model || "claude-sonnet-4-20250514",
      temperature: config.temperature ?? 0.7,
      maxTokens: config.maxTokens ?? 4096,
    });
  }
  // The final else falls back to ChatOpenAI:
  else {
    return new ChatOpenAI({ /* ... */ });
  }
}
```

## Step 3: Configure Environment

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

## Step 4: Use in Agent Config

```typescript
// Reference the model key in agent config:
const config: ReactAgentConfig = {
  type: AgentType.REACT,
  key: "my-agent",
  modelKey: "anthropic-claude",  // Matches registered model lattice key
  // ...
};
registerAgentLattice(config);
```

Register the model lattice (standalone function, not instance method):

```typescript
import { registerModelLattice } from "@axiom-lattice/core";

registerModelLattice("anthropic-claude", {
  provider: "anthropic",
  model: "claude-sonnet-4-20250514",
  maxTokens: 4096,
  temperature: 0.7,
});
// Or use the manager instance: ModelLatticeManager.getInstance().registerLattice(key, config)
```

## Gotchas

- **`ModelLattice.initChatModel()` is not async** and takes a `config` parameter. It uses **if/else if** chains (not a `switch` statement).
- **The final `else` falls back to ChatOpenAI** with `OPENAI_API_KEY` — any unrecognized provider becomes OpenAI-compatible
- **`registerModelLattice(key, config)`** is a standalone convenience function — the instance method is `registerLattice(key, config)`
- Adding a new provider requires modifying `ModelLattice.ts` source — there is no plugin system
- You can use `LLM_BASE_URL` with `provider: "openai"` for OpenAI-compatible APIs without modifying code
