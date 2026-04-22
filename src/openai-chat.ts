/**
 * Mock OpenAI-Compatible Chat Completions API Server (port 18500).
 *
 * Drop-in replacement for any client hitting POST /v1/chat/completions with the
 * OpenAI SDK shape — covers OpenAI, xAI, DeepSeek, OpenRouter, and any other
 * provider exposing the OpenAI chat interface.
 *
 * Point your OpenAI SDK at http://localhost:18500/v1
 *
 *   npx tsx src/openai-chat.ts
 *
 * Endpoints:
 *   POST /v1/chat/completions           Non-streaming chat completion
 *   POST /v1/chat/completions (stream)  SSE streaming when `stream: true`
 *   GET  /v1/models                     List a broad set of known model ids
 *
 * Monitoring:
 *   GET    /_mock/log
 *   DELETE /_mock/log
 *
 * Error injection:
 *   MOCK_ERROR:<code> in last user message text
 *   POST /v1/chat/completions?mock_error=<code>
 *
 * Notes:
 *   - Tool calls are intentionally ignored — `tools`/`tool_choice` pass through
 *     silently and the mock always replies with plain text content.
 *   - Model validation is permissive: any non-empty model id is accepted unless
 *     `mock_error=invalid_model` is set.
 */

import express, { type Request, type Response } from 'express';
import crypto from 'node:crypto';
import {
  RequestLog, mockLogRouter, extractErrorTrigger, handleOpenAIError,
  requireBearerAuth, logConsole, corsMiddleware,
} from './shared.js';

const PORT = 18500;
const STREAM_CHUNK_DELAY_MS_DEFAULT = 5;

// Advertised via GET /v1/models — callers rarely require their model to be in
// this list, but it helps SDK `models.list()` probes look realistic.
const ADVERTISED_MODELS = [
  // OpenAI
  'gpt-5', 'gpt-5-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo',
  // xAI
  'grok-4.20-0309-reasoning', 'grok-4-1-fast-reasoning', 'grok-4-fast-reasoning',
  'grok-4-0709', 'grok-code-fast-1', 'grok-3', 'grok-3-mini',
  // DeepSeek
  'deepseek-chat', 'deepseek-reasoner',
  // OpenRouter (a few popular aliases)
  'anthropic/claude-sonnet-4', 'meta-llama/llama-3.3-70b-instruct',
];

// ── Types ────────────────────────────────────────────────────────────

type Role = 'system' | 'user' | 'assistant' | 'tool' | 'developer';

interface ContentPart {
  type: string;
  text?: string;
}

interface ChatMessage {
  role: Role;
  content?: string | ContentPart[];
  name?: string;
  tool_call_id?: string;
}

interface ChatRequestBody {
  model?: string;
  messages?: ChatMessage[];
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  top_p?: number;
  n?: number;
  tools?: unknown[];
  tool_choice?: unknown;
  response_format?: unknown;
  user?: string;
}

// ── Helpers ──────────────────────────────────────────────────────────

function extractText(content: ChatMessage['content']): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map(p => (p.type === 'text' || p.type === 'input_text') ? (p.text ?? '') : '')
      .join('');
  }
  return '';
}

function lastUserText(messages: ChatMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]!;
    if (m.role === 'user') return extractText(m.content);
  }
  const last = messages[messages.length - 1];
  return last ? extractText(last.content) : '';
}

function approxTokens(text: string): number {
  if (!text) return 0;
  // Rough heuristic: ~4 characters per token.
  return Math.max(1, Math.ceil(text.length / 4));
}

function countPromptTokens(messages: ChatMessage[]): number {
  let total = 0;
  for (const m of messages) {
    total += 4; // per-message overhead (role + separators)
    total += approxTokens(extractText(m.content));
  }
  return total + 2;
}

function buildResponseText(messages: ChatMessage[]): string {
  const text = lastUserText(messages).trim();
  if (!text) return 'Hello from the mock.';
  const echo = text.length > 120 ? text.slice(0, 120) + '…' : text;
  return `Mock response to: ${echo}`;
}

function truncateToTokenBudget(text: string, maxTokens?: number): { text: string; truncated: boolean } {
  if (!maxTokens || maxTokens <= 0) return { text, truncated: false };
  const maxChars = maxTokens * 4;
  if (text.length <= maxChars) return { text, truncated: false };
  return { text: text.slice(0, maxChars), truncated: true };
}

function tokenizeForStream(text: string): string[] {
  // Word-ish chunks preserving trailing whitespace so concatenation round-trips.
  const out: string[] = [];
  let current = '';
  for (const ch of text) {
    current += ch;
    if (ch === ' ' || ch === '\n' || ch === '\t') {
      out.push(current);
      current = '';
    }
  }
  if (current) out.push(current);
  return out;
}

function makeCompletionId(): string {
  return `chatcmpl-mock-${crypto.randomBytes(12).toString('hex')}`;
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function getChunkDelay(req: Request): number {
  const raw = req.query.chunk_delay;
  if (typeof raw === 'string') {
    const parsed = Number.parseInt(raw, 10);
    if (Number.isFinite(parsed) && parsed >= 0) return parsed;
  }
  return STREAM_CHUNK_DELAY_MS_DEFAULT;
}

// ── Response builders ────────────────────────────────────────────────

function buildNonStreamResponse(
  id: string, model: string, content: string,
  promptTokens: number, completionTokens: number,
  finishReason: 'stop' | 'length',
): Record<string, unknown> {
  return {
    id,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message: { role: 'assistant', content, refusal: null },
      finish_reason: finishReason,
      logprobs: null,
    }],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
      prompt_tokens_details: {
        cached_tokens: 0,
        audio_tokens: 0,
      },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    system_fingerprint: 'fp_mock',
  };
}

function buildStreamChunk(
  id: string, model: string,
  delta: Record<string, unknown>,
  finishReason: 'stop' | 'length' | null,
  usage?: Record<string, unknown>,
): Record<string, unknown> {
  const chunk: Record<string, unknown> = {
    id,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      delta,
      finish_reason: finishReason,
      logprobs: null,
    }],
    system_fingerprint: 'fp_mock',
  };
  if (usage) chunk.usage = usage;
  return chunk;
}

function writeSse(res: Response, obj: Record<string, unknown>): void {
  res.write(`data: ${JSON.stringify(obj)}\n\n`);
}

// ── App ──────────────────────────────────────────────────────────────

const app = express();
app.use(corsMiddleware);
app.use(express.json({ limit: '50mb' }));

const log = new RequestLog();
app.use(mockLogRouter(log));

// ── POST /v1/chat/completions ───────────────────────────────────────

app.post('/v1/chat/completions', async (req: Request, res: Response) => {
  let trigger = req.query.mock_error as string | undefined;

  const body: ChatRequestBody = (req.body ?? {}) as ChatRequestBody;
  const model = body.model ?? '';
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const stream = Boolean(body.stream);
  const includeUsageInStream = Boolean(body.stream_options?.include_usage);
  const maxTokens = body.max_completion_tokens ?? body.max_tokens;

  const params: Record<string, unknown> = {
    model,
    message_count: messages.length,
    stream,
    max_tokens: maxTokens ?? null,
    temperature: body.temperature ?? null,
    tool_count: Array.isArray(body.tools) ? body.tools.length : 0,
  };
  log.log('/v1/chat/completions', 'POST', params, model);
  logConsole('POST', '/v1/chat/completions', params);

  if (!requireBearerAuth(req, res, 'openai')) return;

  if (!trigger) {
    trigger = extractErrorTrigger(lastUserText(messages), req.query.mock_error as string) ?? undefined;
  }
  if (trigger) {
    if (handleOpenAIError(res, trigger)) return;
  }

  // -- Validation --
  if (!model) {
    res.status(400).json({
      error: {
        message: "'model' is required.",
        type: 'invalid_request_error',
        code: 'missing_required_parameter',
      },
    });
    return;
  }

  if (messages.length === 0) {
    res.status(400).json({
      error: {
        message: "'messages' must be a non-empty array.",
        type: 'invalid_request_error',
        code: 'invalid_parameter',
      },
    });
    return;
  }

  // -- Build content --
  const rawContent = buildResponseText(messages);
  const { text: content, truncated } = truncateToTokenBudget(rawContent, maxTokens);
  const promptTokens = countPromptTokens(messages);
  const completionTokens = approxTokens(content);
  const finishReason: 'stop' | 'length' = truncated ? 'length' : 'stop';
  const completionId = makeCompletionId();

  if (!stream) {
    res.status(200).json(
      buildNonStreamResponse(completionId, model, content, promptTokens, completionTokens, finishReason),
    );
    return;
  }

  // -- Streaming (SSE) --
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders?.();

  const chunkDelayMs = getChunkDelay(req);

  // Initial chunk establishes the assistant role.
  writeSse(res, buildStreamChunk(completionId, model, { role: 'assistant', content: '' }, null));

  const pieces = tokenizeForStream(content);
  for (const piece of pieces) {
    if (chunkDelayMs > 0) await delay(chunkDelayMs);
    writeSse(res, buildStreamChunk(completionId, model, { content: piece }, null));
  }

  // Final chunk: empty delta + finish_reason, plus usage if requested.
  const finalUsage = includeUsageInStream
    ? {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    }
    : undefined;

  // When usage is included, OpenAI sends an additional chunk with an empty
  // choices array carrying the usage payload. Emit both the finish chunk and
  // the usage chunk for broader client compatibility.
  writeSse(res, buildStreamChunk(completionId, model, {}, finishReason));
  if (finalUsage) {
    writeSse(res, {
      id: completionId,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [],
      usage: finalUsage,
      system_fingerprint: 'fp_mock',
    });
  }
  res.write('data: [DONE]\n\n');
  res.end();
});

// ── GET /v1/models ──────────────────────────────────────────────────

app.get('/v1/models', (_req: Request, res: Response) => {
  log.log('/v1/models', 'GET', {});
  const now = Math.floor(Date.now() / 1000);
  res.json({
    object: 'list',
    data: ADVERTISED_MODELS.map(m => ({
      id: m,
      object: 'model',
      created: now,
      owned_by: 'mock',
    })),
  });
});

// ── Start ───────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('='.repeat(64));
  console.log('  Mock OpenAI-Compatible Chat Completions API');
  console.log('  (covers OpenAI, xAI, DeepSeek, OpenRouter)');
  console.log(`  Base URL: http://localhost:${PORT}/v1`);
  console.log('');
  console.log('  Endpoints:');
  console.log('    POST /v1/chat/completions        -- non-stream + SSE streaming');
  console.log('    GET  /v1/models                  -- advertised model list');
  console.log('');
  console.log('  Testing:');
  console.log('    GET    /_mock/log                -- inspect received params');
  console.log('    DELETE /_mock/log                -- clear log');
  console.log('    ?mock_error=<code>               -- trigger error response');
  console.log('    MOCK_ERROR:<code> in user text   -- trigger error response');
  console.log('    ?chunk_delay=<ms>                -- override streaming delay');
  console.log('');
  console.log('  Notes: tool calls are ignored; content is an echo of the last user message.');
  console.log('='.repeat(64));
});
