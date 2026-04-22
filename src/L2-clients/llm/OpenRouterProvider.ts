import { OpenRouter } from '@openrouter/sdk'
import type {
  LLMProvider,
  LLMSession,
  LLMResponse,
  SessionConfig,
  ToolWithHandler,
  TokenUsage,
  ProviderEventType,
  ProviderEvent,
  ToolCall,
} from './types.js'
import logger from '../../L1-infra/logger/configLogger.js'
import { getConfig } from '../../L1-infra/config/environment.js'

const MAX_TOOL_ROUNDS = 50

function convertTools(tools: ToolWithHandler[]): Array<{ type: 'function'; function: { name: string; description: string; parameters: Record<string, unknown> } }> {
  return tools.map((t) => ({
    type: 'function' as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    },
  }))
}

function buildHandlerMap(tools: ToolWithHandler[]): Map<string, ToolWithHandler['handler']> {
  return new Map(tools.map((t) => [t.name, t.handler]))
}

class OpenRouterSession implements LLMSession {
  private client: OpenRouter
  private model: string
  private messages: Array<{ role: string; content: string }>
  private tools: Array<{ type: 'function'; function: { name: string; description: string; parameters: Record<string, unknown> } }>
  private handlers: Map<string, ToolWithHandler['handler']>
  private listeners = new Map<ProviderEventType, ((e: ProviderEvent) => void)[]>()
  private timeoutMs?: number

  constructor(client: OpenRouter, config: SessionConfig, model: string) {
    this.client = client
    this.model = model
    this.tools = convertTools(config.tools)
    this.handlers = buildHandlerMap(config.tools)
    this.timeoutMs = config.timeoutMs

    this.messages = [{ role: 'system', content: config.systemPrompt }]
  }

  async sendAndWait(message: string): Promise<LLMResponse> {
    this.messages.push({ role: 'user', content: message })

    let cumulative: TokenUsage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
    const start = Date.now()

    let toolRound = 0
    while (true) {
      if (++toolRound > MAX_TOOL_ROUNDS) {
        logger.warn(`OpenRouter agent exceeded ${MAX_TOOL_ROUNDS} tool rounds — aborting to prevent runaway`)
        throw new Error(`Max tool rounds (${MAX_TOOL_ROUNDS}) exceeded — possible infinite loop`)
      }

      const controller = new AbortController()
      const timeoutId = this.timeoutMs
        ? setTimeout(() => controller.abort(), this.timeoutMs)
        : undefined

      try {
        const response = await this.client.chat.send({
          model: this.model,
          messages: this.messages,
          ...(this.tools.length > 0 ? { tools: this.tools } : {}),
        } as any)

        if (timeoutId) clearTimeout(timeoutId)

        const choice = response.choices?.[0]
        const assistantMsg = choice?.message
        const content = assistantMsg?.content || ''

        const toolCalls = (assistantMsg as any)?.tool_calls
        if (!toolCalls || toolCalls.length === 0) {
          return {
            content,
            toolCalls: [],
            usage: cumulative,
            durationMs: Date.now() - start,
          }
        }

        for (const tc of toolCalls) {
          if (tc.type !== 'function') continue

          const fnName = tc.function?.name
          const handler = this.handlers.get(fnName)

          let result: unknown
          if (!handler) {
            logger.warn(`OpenRouter requested unknown tool: ${fnName}`)
            result = { error: `Unknown tool: ${fnName}` }
          } else {
            this.emit('tool_start', { name: fnName, arguments: tc.function?.arguments })
            try {
              const args = JSON.parse(tc.function?.arguments || '{}') as Record<string, unknown>
              result = await handler(args)
            } catch (err) {
              logger.error(`Tool ${fnName} failed: ${err}`)
              result = { error: String(err) }
            }
            this.emit('tool_end', { name: fnName, result })
          }

          this.messages.push({
            role: 'tool',
            content: typeof result === 'string' ? result : JSON.stringify(result),
          } as any)
        }

        continue
      } catch (error) {
        if (timeoutId) clearTimeout(timeoutId)
        throw error
      }
    }
  }

  on(event: ProviderEventType, handler: (e: ProviderEvent) => void): void {
    const list = this.listeners.get(event) ?? []
    list.push(handler)
    this.listeners.set(event, list)
  }

  async close(): Promise<void> {
    this.messages = []
    this.listeners.clear()
  }

  private emit(type: ProviderEventType, data: unknown): void {
    for (const handler of this.listeners.get(type) ?? []) {
      try {
        handler({ type, data })
      } catch {
        // Don't let listener errors break the agent loop
      }
    }
  }
}

export class OpenRouterProvider implements LLMProvider {
  readonly name = 'openrouter' as const
  private apiKeys: string[] = []
  private currentKeyIndex = 0

  constructor() {
    const config = getConfig()
    this.apiKeys = [config.OPENROUTER_API_KEY, ...config.OPENROUTER_API_KEYS].filter(Boolean)
  }

  isAvailable(): boolean {
    return !!this.apiKeys[0]
  }

  getDefaultModel(): string {
    return getConfig().OPENROUTER_MODEL || 'nvidia/nemotron-3-super-120b-a12b:free'
  }

  async createSession(config: SessionConfig): Promise<LLMSession> {
    const key = this.getNextKey()
    const client = new OpenRouter({ apiKey: key })
    const model = config.model ?? this.getDefaultModel()

    logger.info(`OpenRouter session created (model=${model}, tools=${config.tools.length})`)
    return new OpenRouterSession(client, config, model)
  }

  private getNextKey(): string {
    const key = this.apiKeys[this.currentKeyIndex]
    this.currentKeyIndex = (this.currentKeyIndex + 1) % Math.max(1, this.apiKeys.length)
    return key
  }
}