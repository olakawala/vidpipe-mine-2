import type { LLMProvider } from './types.js';
import type { ProviderName } from './types.js';
import { CopilotProvider } from './CopilotProvider.js';
import { OpenAIProvider } from './OpenAIProvider.js';
import { ClaudeProvider } from './ClaudeProvider.js';
import { GeminiProvider } from './GeminiProvider.js';
import { OpenRouterProvider } from './OpenRouterProvider.js';
import logger from '../../L1-infra/logger/configLogger.js';
import { getConfig } from '../../L1-infra/config/environment.js';

const providers: Record<ProviderName, () => LLMProvider> = {
  copilot: () => new CopilotProvider(),
  openai: () => new OpenAIProvider(),
  claude: () => new ClaudeProvider(),
  gemini: () => new GeminiProvider(),
  openrouter: () => new OpenRouterProvider(),
};

/** Cached singleton provider instance */
let currentProvider: LLMProvider | null = null;
let currentProviderName: ProviderName | null = null;

/**
 * Get the configured LLM provider.
 * Reads from LLM_PROVIDER env var, defaults to 'gemini'.
 * Falls back to 'openrouter' if gemini is unavailable.
 * Caches the instance for reuse.
 */
export function getProvider(name?: ProviderName): LLMProvider {
  let raw = name ?? getConfig().LLM_PROVIDER.trim().toLowerCase();

  // Default to gemini if not specified
  if (!raw) {
    raw = 'gemini';
  }

  const providerName = raw as ProviderName;

  if (currentProvider && currentProviderName === providerName) {
    return currentProvider;
  }

  // Close old provider if switching to a different one
  currentProvider?.close?.().catch(() => { /* ignore close errors */ });

  const provider = tryCreateProvider(providerName);
  logger.info(`Using LLM provider: ${provider.name} (model: ${provider.getDefaultModel()})`);
  currentProvider = provider;
  currentProviderName = providerName;
  return currentProvider;
}

function tryCreateProvider(providerName: ProviderName): LLMProvider {
  if (!providers[providerName]) {
    throw new Error(
      `Unknown LLM provider: "${providerName}". ` +
      `Valid options: ${Object.keys(providers).join(', ')}`
    );
  }

  const provider = providers[providerName]();

  if (provider.isAvailable()) {
    return provider;
  }

  // Fallback logic: gemini -> openrouter -> copilot
  if (providerName === 'gemini') {
    logger.warn(`Gemini unavailable, falling back to openrouter`);
    const fallback = tryCreateProvider('openrouter');
    if (fallback.isAvailable()) {
      return fallback;
    }
    logger.warn(`OpenRouter unavailable, falling back to copilot`);
  }

  // Try copilot as final fallback
  const copilot = providers.copilot();
  if (copilot.isAvailable()) {
    return copilot;
  }

  // No providers available
  throw new Error(
    `No LLM providers available. ` +
    `Please configure at least one of: GEMINI_API_KEY, OPENROUTER_API_KEY, or COPILOT_TOKEN`
  );
}

/** Reset the cached provider (for testing) */
export async function resetProvider(): Promise<void> {
  try { await currentProvider?.close?.(); } catch { /* ignore close errors */ }
  currentProvider = null;
  currentProviderName = null;
}

/** Get the name of the current provider */
export function getProviderName(): ProviderName {
  const raw = getConfig().LLM_PROVIDER.trim().toLowerCase();
  const valid: ProviderName[] = ['copilot', 'openai', 'claude', 'gemini', 'openrouter'];
  return currentProviderName ?? (valid.includes(raw as ProviderName) ? (raw as ProviderName) : 'gemini');
}

// Re-export types and providers
export type { LLMProvider, LLMSession, LLMResponse, SessionConfig, ToolWithHandler, TokenUsage, CostInfo, QuotaSnapshot, ProviderEvent, ProviderEventType } from './types.js';
export type { ProviderName } from './types.js';
export { CopilotProvider } from './CopilotProvider.js';
export { OpenAIProvider } from './OpenAIProvider.js';
export { ClaudeProvider } from './ClaudeProvider.js';
export { GeminiProvider } from './GeminiProvider.js';
export { OpenRouterProvider } from './OpenRouterProvider.js';