/**
 * Mock Cloudflare Workers AI Text Generation API Server.
 *
 * Drop-in replacement for testing against the real Cloudflare API.
 * Point your Cloudflare client at http://localhost:8350 instead of https://api.cloudflare.com
 *
 *   npx tsx src/cloudflare-text.ts
 *
 * Then in your config (cloudflare-text-generation):
 *   apiToken: "mock-token"
 *   accountId: "mock-account"
 *   apiBaseUrl: "http://localhost:8350/client/v4"
 *
 * Endpoints:
 *   POST /client/v4/accounts/:accountId/ai/run/:modelId
 *     - Chat format: { "messages": [{"role": "user", "content": "..."}], "stream": false }
 *     - Simple format: { "prompt": "...", "stream": false }
 *     - Supports: max_tokens, temperature, top_p, top_k, repetition_penalty, seed
 *
 * Testing:
 *   GET    /_mock/log              — inspect received params
 *   DELETE /_mock/log              — clear log
 *   ?mock_error=<code>             — trigger error response
 *   MOCK_ERROR:<code> in prompt    — trigger error response
 *   fail=<code> in prompt          — legacy trigger (still works)
 *   Codes: auth, bad_model, no_capacity, quota, timeout,
 *          network, garbage, success_false,
 *          server_error, forbidden
 */

import express, { type Request, type Response } from 'express';
import {
  RequestLog, mockLogRouter, extractErrorTriggerCf, handleCloudflareError,
  requireBearerAuth, cfError, logConsole, corsMiddleware,
} from './shared.js';

const PORT = 18350;

const VALID_MODELS = new Set([
  '@cf/meta/llama-3.1-8b-instruct',
  '@cf/meta/llama-3.1-70b-instruct',
  '@cf/meta/llama-3.2-1b-instruct',
  '@cf/meta/llama-3.2-3b-instruct',
  '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
  '@cf/mistral/mistral-7b-instruct-v0.2',
  '@cf/qwen/qwen1.5-14b-chat-awq',
  '@cf/google/gemma-7b-it-lora',
  '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b',
  '@hf/google/gemma-7b-it',
]);

// ── Helpers ─────────────────────────────────────────────────────────

interface ChatMessage {
  role: string;
  content: string;
}

function extractPromptText(params: Record<string, unknown>): string {
  if ('messages' in params) {
    const messages = params.messages;
    if (Array.isArray(messages)) {
      for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i] as ChatMessage;
        if (msg && typeof msg === 'object' && msg.role === 'user') {
          return msg.content ?? '';
        }
      }
      if (messages.length > 0) {
        const last = messages[messages.length - 1] as ChatMessage;
        if (last && typeof last === 'object') {
          return last.content ?? '';
        }
      }
    }
  }
  return (params.prompt as string) ?? '';
}

function generateMockResponse(promptText: string, modelId: string): string {
  const modelShort = modelId.includes('/') ? modelId.split('/').pop()! : modelId;
  if (!promptText) {
    return `This is a mock response from ${modelShort}.`;
  }

  const promptPreview = promptText.slice(0, 80).replace(/\n/g, ' ');
  return (
    `This is a mock response from ${modelShort}. ` +
    `You asked: "${promptPreview}". ` +
    `In a real deployment, the model would generate a contextual answer here. ` +
    `The Workers AI platform processed your request successfully.`
  );
}

function* streamResponse(text: string): Generator<string> {
  const words = text.split(' ');
  for (let i = 0; i < words.length; i++) {
    const chunk = words[i] + (i < words.length - 1 ? ' ' : '');
    const payload = JSON.stringify({ response: chunk, p: `abcdefgh${String(i).padStart(4, '0')}` });
    yield `data: ${payload}\n\n`;
  }
  yield 'data: [DONE]\n\n';
}

// ── Express app ─────────────────────────────────────────────────────

const app = express();
app.use(corsMiddleware);
app.use(express.json());

const log = new RequestLog();
app.use(mockLogRouter(log));

// ── Main AI run endpoint ────────────────────────────────────────────

app.post('/client/v4/accounts/:accountId/ai/run/:modelId(*)', (req: Request, res: Response): void => {
  const modelId = String(req.params.modelId);
  const accountId = String(req.params.accountId);

  const queryTrigger = req.query.mock_error as string | undefined;
  if (queryTrigger) {
    log.log(`ai/run/${modelId}`, 'POST', { mock_error: queryTrigger }, modelId);
    logConsole('POST', `/client/v4/accounts/${accountId}/ai/run/${modelId}`, { mock_error: queryTrigger });
    if (handleCloudflareError(res, queryTrigger, modelId)) return;
  }

  if (!requireBearerAuth(req, res, 'cloudflare')) return;

  let params: Record<string, unknown>;
  try {
    params = req.body ?? {};
  } catch (e) {
    cfError(res, 400, 3003, `Failed to parse request body: ${e}`);
    return;
  }

  const promptText = extractPromptText(params);
  const stream = params.stream === true;

  let trigger = queryTrigger ?? null;
  if (!trigger) {
    trigger = extractErrorTriggerCf(promptText);
  }

  log.log(`ai/run/${modelId}`, 'POST', params, modelId);
  logConsole('POST', `/client/v4/accounts/${accountId}/ai/run/${modelId}`, {
    stream,
    error_trigger: trigger ?? '(none)',
  });

  if (trigger) {
    if (handleCloudflareError(res, trigger, modelId)) return;
  }

  if (!VALID_MODELS.has(modelId)) {
    cfError(res, 400, 5007, `No such model ${modelId} or task`);
    return;
  }

  if (!promptText) {
    cfError(res, 400, 5004, 'Missing required parameter: prompt or messages');
    return;
  }

  let responseText = generateMockResponse(promptText, modelId);

  const maxTokens = params.max_tokens;
  if (typeof maxTokens === 'number' && maxTokens > 0) {
    const words = responseText.split(' ');
    if (words.length > maxTokens) {
      responseText = words.slice(0, maxTokens).join(' ');
    }
  }

  if (stream) {
    res.set({
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    });

    const chunks = [...streamResponse(responseText)];
    let i = 0;

    function sendNext(): void {
      if (i < chunks.length) {
        res.write(chunks[i]);
        i++;
        setTimeout(sendNext, 20);
      } else {
        res.end();
      }
    }

    sendNext();
    return;
  }

  res.json({
    result: { response: responseText },
    success: true,
    errors: [],
    messages: [],
  });
});

// ── Health / smoke endpoint ─────────────────────────────────────────

app.get('/', (_req: Request, res: Response): void => {
  res.json({
    name: 'mock_cloudflare_text_server',
    valid_models: [...VALID_MODELS].sort(),
    error_triggers: [
      'auth', 'bad_model', 'no_capacity', 'quota', 'timeout',
      'network', 'garbage', 'success_false',
      'server_error', 'forbidden',
    ],
  });
});

// ── Start server ────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('  Mock Cloudflare Workers AI Text Generation API');
  console.log(`  Base URL: http://localhost:${PORT}`);
  console.log('  Endpoint: POST /client/v4/accounts/{id}/ai/run/{model}');
  console.log('');
  console.log('  Models: ' + [...VALID_MODELS].sort().join(', '));
  console.log('');
  console.log('  Formats:');
  console.log('    Chat:   { "messages": [{"role":"user","content":"..."}] }');
  console.log('    Simple: { "prompt": "..." }');
  console.log('    Stream: add "stream": true to either format');
  console.log('');
  console.log('  Params: max_tokens, temperature, top_p, top_k,');
  console.log('          repetition_penalty, seed (accepted, not functional)');
  console.log('');
  console.log('  Testing:');
  console.log('    GET    /_mock/log              — inspect received params');
  console.log('    DELETE /_mock/log              — clear log');
  console.log('    ?mock_error=<code>             — trigger error response');
  console.log('    MOCK_ERROR:<code> in prompt    — trigger error response');
  console.log('    fail=<code> in prompt          — legacy trigger (still works)');
  console.log('    Codes: auth, bad_model, no_capacity, quota, timeout,');
  console.log('           network, garbage, success_false,');
  console.log('           server_error, forbidden');
  console.log('='.repeat(60));
});
