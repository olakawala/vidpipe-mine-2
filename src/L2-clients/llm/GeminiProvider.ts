import { GoogleGenAI, FunctionDeclaration, Type } from '@google/genai'
import type {
  LLMProvider,
  LLMSession,
  LLMResponse,
  SessionConfig,
  ToolWithHandler,
  TokenUsage,
  ProviderEventType,
  ProviderEvent,
} from './types.js'
import logger from '../../L1-infra/logger/configLogger.js'
import { getConfig } from '../../L1-infra/config/environment.js'

const MAX_TOOL_ROUNDS = 50

function convertTools(tools: ToolWithHandler[]): FunctionDeclaration[] {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    parameters: t.parameters as Record<string, unknown>,
  }))
}

function buildHandlerMap(tools: ToolWithHandler[]): Map<string, ToolWithHandler['handler']> {
  return new Map(tools.map((t) => [t.name, t.handler]))
}

function addUsage(a: TokenUsage, b: TokenUsage): TokenUsage {
  return {
    inputTokens: a.inputTokens + b.inputTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    totalTokens: a.totalTokens + b.totalTokens,
  }
}

interface ContentPart {
  role: string
  parts: Array<{ text?: string; functionCall?: { name: string; args: Record<string, unknown> }; functionResponse?: { name: string; response: unknown } }>
}

class GeminiSession implements LLMSession {
  private client: GoogleGenAI
  private model: string
  private messages: ContentPart[]
  private tools: FunctionDeclaration[]
  private handlers: Map<string, ToolWithHandler['handler']>
  private listeners = new Map<ProviderEventType, ((e: ProviderEvent) => void)[]>()
  private timeoutMs?: number

  constructor(client: GoogleGenAI, config: SessionConfig, model: string) {
    this.client = client
    this.model = model
    this.tools = convertTools(config.tools)
    this.handlers = buildHandlerMap(config.tools)
    this.timeoutMs = config.timeoutMs

    this.messages = [{
      role: 'model',
      parts: [{ text: config.systemPrompt }]
    }]
  }

  async sendAndWait(message: string): Promise<LLMResponse> {
    this.messages.push({ role: 'user', parts: [{ text: message }] })

    let cumulative: TokenUsage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
    const start = Date.now()

    let toolRound = 0
    while (true) {
      if (++toolRound > MAX_TOOL_ROUNDS) {
        logger.warn(`Gemini agent exceeded ${MAX_TOOL_ROUNDS} tool rounds — aborting to prevent runaway`)
        throw new Error(`Max tool rounds (${MAX_TOOL_ROUNDS}) exceeded — possible infinite loop`)
      }

      const controller = new AbortController()
      const timeoutId = this.timeoutMs
        ? setTimeout(() => controller.abort(), this.timeoutMs)
        : undefined

      try {
        const response = await this.client.models.generateContent({
          model: this.model,
          contents: this.messages as any,
          config: {
            tools: this.tools.length > 0 ? { functionDeclarations: this.tools } as any : undefined,
          } as any,
        })

        if (timeoutId) clearTimeout(timeoutId)

        const text = response.text || ''

        if (response.functionCalls && response.functionCalls.length > 0) {
          for (const fc of response.functionCalls) {
            const fnName = fc.name || ''
            const handler = this.handlers.get(fnName)

            let result: unknown
            if (!handler) {
              logger.warn(`Gemini requested unknown tool: ${fnName}`)
              result = { error: `Unknown tool: ${fnName}` }
            } else {
              this.emit('tool_start', { name: fnName, arguments: fc.args })
              try {
                result = await handler(fc.args || {})
              } catch (err) {
                logger.error(`Tool ${fnName} failed: ${err}`)
                result = { error: String(err) }
              }
              this.emit('tool_end', { name: fnName, result })
            }

            this.messages.push({
              role: 'model',
              parts: [{
                functionCall: {
                  name: fnName,
                  args: (fc.args as Record<string, unknown>) || {},
                }
              }]
            })

            this.messages.push({
              role: 'user',
              parts: [{
                functionResponse: {
                  name: fnName,
                  response: result,
                }
              }]
            })
          }

          continue
        }

        return {
          content: text,
          toolCalls: [],
          usage: cumulative,
          durationMs: Date.now() - start,
        }
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

export class GeminiProvider implements LLMProvider {
  readonly name = 'gemini' as const
  private apiKeys: string[] = []
  private currentKeyIndex = 0

  constructor() {
    const config = getConfig()
    this.apiKeys = [config.GEMINI_API_KEY, ...config.GEMINI_API_KEYS].filter(Boolean)
  }

  isAvailable(): boolean {
    return !!this.apiKeys[0]
  }

  getDefaultModel(): string {
    return getConfig().GEMINI_MODEL || 'gemini-2.5-flash'
  }

  async createSession(config: SessionConfig): Promise<LLMSession> {
    const key = this.getNextKey()
    const client = new GoogleGenAI({ apiKey: key })
    const model = config.model ?? this.getDefaultModel()

    logger.info(`Gemini session created (model=${model}, tools=${config.tools.length})`)
    return new GeminiSession(client, config, model)
  }

  private getNextKey(): string {
    const key = this.apiKeys[this.currentKeyIndex]
    this.currentKeyIndex = (this.currentKeyIndex + 1) % Math.max(1, this.apiKeys.length)
    return key
  }
}
