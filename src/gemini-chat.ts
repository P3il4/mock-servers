/**
 * Mock Gemini Chat API Server (port 8180)
 *
 * Implements the Google Gemini native GenerateContent API for chat,
 * with multi-turn conversation, system instructions, function calling,
 * thinking/reasoning tokens, and safety ratings.
 *
 * Endpoints:
 *   POST /v1beta/models/:model:generateContent         Non-streaming
 *   POST /v1beta/models/:model:generateContent?alt=sse  Streaming SSE
 *   POST /v1beta/models/:model:streamGenerateContent    SDK streaming
 *   POST /v1beta/models/:model:countTokens              Token counting
 *   GET  /v1beta/models                                 List models
 *
 * Error injection:
 *   ?mock_error=<code>  or  MOCK_ERROR:<code> in prompt
 *
 * Usage:
 *   npx tsx src/gemini-chat.ts
 */

import express, { type Request, type Response } from 'express';
import crypto from 'node:crypto';
import {
  RequestLog, mockLogRouter, extractErrorTrigger, handleGoogleError,
  logConsole, GOOGLE_ERRORS, corsMiddleware,
} from './shared.js';

/** Extract a single string from an Express query/param value. */
function qs(v: unknown): string | undefined {
  if (typeof v === 'string') return v;
  if (Array.isArray(v) && typeof v[0] === 'string') return v[0];
  return undefined;
}

// ── Valid models ────────────────────────────────────────────────────

const VALID_MODELS = new Set([
  'gemini-2.5-flash',
  'gemini-2.5-pro',
  'gemini-2.0-flash',
  'gemini-2.0-flash-lite',
  'gemini-2.5-flash-thinking',
  'gemini-2.5-pro-thinking',
]);

const MODEL_INFO = [...VALID_MODELS].sort().map(m => ({
  name: `models/${m}`,
  version: '001',
  displayName: m,
  description: `Mock ${m} model`,
  inputTokenLimit: 1048576,
  outputTokenLimit: m.includes('pro') ? 65536 : 32768,
  supportedGenerationMethods: ['generateContent', 'streamGenerateContent', 'countTokens'],
}));

// ── Safety ratings ──────────────────────────────────────────────────

const HARM_CATEGORIES = [
  'HARM_CATEGORY_HARASSMENT',
  'HARM_CATEGORY_HATE_SPEECH',
  'HARM_CATEGORY_SEXUALLY_EXPLICIT',
  'HARM_CATEGORY_DANGEROUS_CONTENT',
  'HARM_CATEGORY_CIVIC_INTEGRITY',
];

function defaultSafetyRatings(): Array<Record<string, unknown>> {
  return HARM_CATEGORIES.map(cat => ({
    category: cat, probability: 'NEGLIGIBLE', blocked: false,
  }));
}

function blockedSafetyRatings(): Array<Record<string, unknown>> {
  const ratings = defaultSafetyRatings();
  ratings[0] = { category: HARM_CATEGORIES[0], probability: 'HIGH', blocked: true };
  return ratings;
}

// ── Token counting helpers ──────────────────────────────────────────

function estimateTokens(text: string | null | undefined): number {
  if (!text) return 0;
  return Math.max(1, text.split(/\s+/).filter(Boolean).length);
}

function countContentsTokens(contents: unknown): number {
  if (!Array.isArray(contents)) return 0;
  let total = 0;
  for (const content of contents) {
    if (typeof content !== 'object' || content === null) continue;
    const parts = (content as Record<string, unknown>).parts;
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      if (typeof part !== 'object' || part === null) continue;
      const p = part as Record<string, unknown>;
      total += estimateTokens(p.text as string | undefined);
      if (p.functionCall) total += estimateTokens(JSON.stringify(p.functionCall));
      if (p.functionResponse) total += estimateTokens(JSON.stringify(p.functionResponse));
      if (p.inlineData) total += 258;
    }
  }
  return total;
}

function countSystemTokens(systemInstruction: unknown): number {
  if (!systemInstruction) return 0;
  if (typeof systemInstruction === 'string') return estimateTokens(systemInstruction);
  const si = systemInstruction as Record<string, unknown>;
  const parts = si.parts;
  if (!Array.isArray(parts)) return 0;
  let total = 0;
  for (const part of parts) {
    if (typeof part === 'string') total += estimateTokens(part);
    else if (typeof part === 'object' && part !== null) total += estimateTokens((part as Record<string, unknown>).text as string | undefined);
  }
  return total;
}

// ── Prompt text extraction ──────────────────────────────────────────

function getPromptText(body: Record<string, unknown>): string {
  const contents = body.contents;
  if (!Array.isArray(contents)) return '';
  for (let i = contents.length - 1; i >= 0; i--) {
    const content = contents[i] as Record<string, unknown> | undefined;
    if (!content || typeof content !== 'object') continue;
    if ((content.role ?? 'user') !== 'user') continue;
    const parts = content.parts;
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      if (typeof part === 'object' && part !== null && typeof (part as Record<string, unknown>).text === 'string') {
        return (part as Record<string, unknown>).text as string;
      }
    }
  }
  return '';
}

// ── Function calling helpers ────────────────────────────────────────

interface FuncDecl {
  name: string;
  parameters?: Record<string, unknown>;
}

function extractFunctionDeclarations(body: Record<string, unknown>): FuncDecl[] {
  const tools = body.tools;
  if (!Array.isArray(tools)) return [];
  const declarations: FuncDecl[] = [];
  for (const tool of tools) {
    if (typeof tool !== 'object' || tool === null) continue;
    const fds = (tool as Record<string, unknown>).functionDeclarations;
    if (!Array.isArray(fds)) continue;
    declarations.push(...(fds as FuncDecl[]));
  }
  return declarations;
}

function buildFunctionCallArgs(
  funcDecl: FuncDecl,
  promptText: string,
): Record<string, unknown> {
  const paramsSchema = funcDecl.parameters ?? {};
  const properties = (paramsSchema as Record<string, unknown>).properties as Record<string, Record<string, unknown>> | undefined;
  if (!properties) return {};

  const args: Record<string, unknown> = {};
  for (const [paramName, paramSchema] of Object.entries(properties)) {
    const paramType = ((paramSchema?.type as string) ?? 'STRING').toUpperCase();
    switch (paramType) {
      case 'STRING':
        args[paramName] = promptText ? promptText.slice(0, 100) : 'mock_value'; break;
      case 'NUMBER': case 'FLOAT':
        args[paramName] = 42.0; break;
      case 'INTEGER': case 'INT':
        args[paramName] = 42; break;
      case 'BOOLEAN': case 'BOOL':
        args[paramName] = true; break;
      case 'ARRAY':
        args[paramName] = ['mock_item']; break;
      case 'OBJECT':
        args[paramName] = { mock_key: 'mock_value' }; break;
      default:
        args[paramName] = 'mock_value';
    }
  }
  return args;
}

// ── Thinking mode detection ─────────────────────────────────────────

function isThinkingMode(model: string, body: Record<string, unknown>): boolean {
  if (model.includes('thinking')) return true;
  const genConfig = body.generationConfig as Record<string, unknown> | undefined;
  if (typeof genConfig === 'object' && genConfig !== null) {
    const thinkingConfig = genConfig.thinkingConfig as Record<string, unknown> | undefined;
    if (typeof thinkingConfig === 'object' && thinkingConfig !== null) {
      if (thinkingConfig.includeThoughts) return true;
      const budget = thinkingConfig.thinkingBudget;
      if (typeof budget === 'number' && budget !== 0) return true;
    }
  }
  return false;
}

// ── Usage metadata builder ──────────────────────────────────────────

function buildUsageMetadata(
  promptTokens: number, candidateTokens: number,
  thoughtsTokens = 0, cachedTokens = 0,
): Record<string, unknown> {
  const meta: Record<string, unknown> = {
    promptTokenCount: promptTokens,
    candidatesTokenCount: candidateTokens,
    totalTokenCount: promptTokens + candidateTokens + thoughtsTokens,
  };
  if (thoughtsTokens > 0) meta.thoughtsTokenCount = thoughtsTokens;
  if (cachedTokens > 0) meta.cachedContentTokenCount = cachedTokens;
  return meta;
}

// ── System instruction extraction ───────────────────────────────────

function extractSystemText(body: Record<string, unknown>): string {
  const si = body.systemInstruction;
  if (!si) return '';
  if (typeof si === 'string') return si;
  const parts = (si as Record<string, unknown>).parts;
  if (!Array.isArray(parts)) return '';
  const pieces: string[] = [];
  for (const p of parts) {
    if (typeof p === 'string') pieces.push(p);
    else if (typeof p === 'object' && p !== null && typeof (p as Record<string, unknown>).text === 'string') {
      pieces.push((p as Record<string, unknown>).text as string);
    }
  }
  return pieces.join(' ').trim();
}

// ── Content error responses ─────────────────────────────────────────

function buildSafetyErrorResponse(model: string): Record<string, unknown> {
  return {
    candidates: [{
      content: { parts: [], role: 'model' },
      finishReason: 'SAFETY',
      index: 0,
      safetyRatings: blockedSafetyRatings(),
      tokenCount: 0,
    }],
    usageMetadata: buildUsageMetadata(5, 0),
    modelVersion: model,
    responseId: crypto.randomUUID(),
  };
}

function buildContentFilterResponse(model: string): Record<string, unknown> {
  return {
    candidates: [],
    promptFeedback: {
      blockReason: 'SAFETY',
      safetyRatings: blockedSafetyRatings(),
    },
    usageMetadata: buildUsageMetadata(5, 0),
    modelVersion: model,
    responseId: crypto.randomUUID(),
  };
}

function buildRecitationResponse(model: string): Record<string, unknown> {
  return {
    candidates: [{
      content: { parts: [{ text: '' }], role: 'model' },
      finishReason: 'RECITATION',
      index: 0,
      safetyRatings: defaultSafetyRatings(),
      citationMetadata: {
        citationSources: [{ startIndex: 0, endIndex: 50, uri: 'https://example.com/source' }],
      },
      tokenCount: 0,
    }],
    usageMetadata: buildUsageMetadata(5, 0),
    modelVersion: model,
    responseId: crypto.randomUUID(),
  };
}

// ── generateContent (non-streaming) ─────────────────────────────────

function buildGenerateResponse(body: Record<string, unknown>, model: string): Record<string, unknown> {
  const promptText = getPromptText(body);
  const funcDecls = extractFunctionDeclarations(body);
  const thinking = isThinkingMode(model, body);

  const contents = body.contents;
  let promptTokens = countContentsTokens(contents);
  promptTokens += countSystemTokens(body.systemInstruction);

  const parts: Array<Record<string, unknown>> = [];
  let thoughtsTokens = 0;

  if (thinking) {
    const thoughtText = `Let me think about this step by step.\n\nThe user asked: "${promptText.slice(0, 80)}"\n\nI need to consider the context and provide a helpful response.`;
    parts.push({ text: thoughtText, thought: true });
    thoughtsTokens = estimateTokens(thoughtText);
  }

  let candidateTokens: number;
  const finishReason = 'STOP';

  if (funcDecls.length > 0) {
    const fd = funcDecls[0]!;
    const funcName = fd.name ?? 'mock_function';
    const funcArgs = buildFunctionCallArgs(fd, promptText);
    parts.push({ functionCall: { name: funcName, args: funcArgs } });
    candidateTokens = estimateTokens(funcName) + estimateTokens(JSON.stringify(funcArgs));
  } else {
    const systemText = extractSystemText(body);
    const turnCount = Array.isArray(contents) ? contents.length : 0;
    let contextNote = '';
    if (systemText) contextNote = ` (system: ${systemText.slice(0, 60)})`;
    if (turnCount > 1) contextNote += ` [turn ${turnCount}]`;

    const replyText = `Mock response from ${model}${contextNote}: This is a simulated reply to "${promptText.slice(0, 120)}".`;
    parts.push({ text: replyText });
    candidateTokens = estimateTokens(replyText);
  }

  const allText = parts
    .filter(p => typeof p.text === 'string')
    .map(p => p.text as string)
    .join(' ');

  return {
    candidates: [{
      content: { parts, role: 'model' },
      finishReason,
      index: 0,
      safetyRatings: defaultSafetyRatings(),
      tokenCount: estimateTokens(allText),
    }],
    usageMetadata: buildUsageMetadata(promptTokens, candidateTokens, thoughtsTokens),
    modelVersion: model,
    responseId: crypto.randomUUID(),
  };
}

// ── streamGenerateContent (SSE) ─────────────────────────────────────

function generateStreamChunks(body: Record<string, unknown>, model: string): Array<Record<string, unknown>> {
  const promptText = getPromptText(body);
  const funcDecls = extractFunctionDeclarations(body);
  const thinking = isThinkingMode(model, body);

  const contents = body.contents;
  let promptTokens = countContentsTokens(contents);
  promptTokens += countSystemTokens(body.systemInstruction);

  const responseId = crypto.randomUUID();
  const chunks: Array<Record<string, unknown>> = [];
  let totalCandidateTokens = 0;
  let thoughtsTokens = 0;

  if (thinking) {
    const thoughtText = `Let me think about this step by step.\n\nThe user asked: "${promptText.slice(0, 80)}"\n\nI need to consider the context and provide a helpful response.`;
    const thoughtWords = thoughtText.split(/\s+/).filter(Boolean);
    thoughtsTokens = thoughtWords.length;
    for (let i = 0; i < thoughtWords.length; i++) {
      const textPiece = i === 0 ? thoughtWords[i]! : ' ' + thoughtWords[i]!;
      chunks.push({
        candidates: [{
          content: { parts: [{ text: textPiece, thought: true }], role: 'model' },
          index: 0,
          safetyRatings: defaultSafetyRatings(),
        }],
        modelVersion: model,
        responseId,
      });
    }
  }

  if (funcDecls.length > 0) {
    const fd = funcDecls[0]!;
    const funcName = fd.name ?? 'mock_function';
    const funcArgs = buildFunctionCallArgs(fd, promptText);
    const candidateTokens = estimateTokens(funcName) + estimateTokens(JSON.stringify(funcArgs));
    totalCandidateTokens = candidateTokens;
    chunks.push({
      candidates: [{
        content: {
          parts: [{ functionCall: { name: funcName, args: funcArgs } }],
          role: 'model',
        },
        finishReason: 'STOP',
        index: 0,
        safetyRatings: defaultSafetyRatings(),
      }],
      usageMetadata: buildUsageMetadata(promptTokens, candidateTokens, thoughtsTokens),
      modelVersion: model,
      responseId,
    });
  } else {
    const systemText = extractSystemText(body);
    const turnCount = Array.isArray(contents) ? contents.length : 0;
    let contextNote = '';
    if (systemText) contextNote = ` (system: ${systemText.slice(0, 60)})`;
    if (turnCount > 1) contextNote += ` [turn ${turnCount}]`;

    const replyText = `Mock response from ${model}${contextNote}: This is a simulated reply to "${promptText.slice(0, 120)}".`;
    const words = replyText.split(/\s+/).filter(Boolean);
    totalCandidateTokens = words.length;

    for (let i = 0; i < words.length; i++) {
      const textPiece = i === 0 ? words[i]! : ' ' + words[i]!;
      const isLast = i === words.length - 1;
      const chunk: Record<string, unknown> = {
        candidates: [{
          content: { parts: [{ text: textPiece }], role: 'model' },
          index: 0,
          safetyRatings: defaultSafetyRatings(),
          ...(isLast ? { finishReason: 'STOP' } : {}),
        }],
        modelVersion: model,
        responseId,
      };
      if (isLast) {
        chunk.usageMetadata = buildUsageMetadata(promptTokens, totalCandidateTokens, thoughtsTokens);
      }
      chunks.push(chunk);
    }
  }

  return chunks;
}

// ── Console logging helper ──────────────────────────────────────────

function buildConsoleParams(body: Record<string, unknown>): Record<string, unknown> {
  const logData: Record<string, unknown> = {};
  const genConfig = body.generationConfig;
  if (genConfig) logData.generationConfig = genConfig;
  const si = body.systemInstruction;
  if (si) {
    logData.systemInstruction = typeof si === 'string' ? (si as string).slice(0, 100) : '[Content object]';
  }
  const tools = body.tools;
  if (Array.isArray(tools)) logData.tools = `[${tools.length} tool(s)]`;
  const prompt = getPromptText(body);
  if (prompt) logData.prompt = prompt.slice(0, 200);
  const contents = body.contents;
  if (Array.isArray(contents)) logData.turns = contents.length;
  return logData;
}

// ── Express App ─────────────────────────────────────────────────────

const app = express();
app.use(corsMiddleware);
app.use(express.json({ limit: '50mb' }));

const log = new RequestLog();
app.use(mockLogRouter(log));

// CORS
app.use((_req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-goog-api-client, x-goog-api-key');
  next();
});

app.options('*', (_req, res) => res.sendStatus(200));

// List models
app.get('/v1beta/models', (req: Request, res: Response) => {
  logConsole('GET', req.path);
  log.log('/v1beta/models', 'GET', {});
  res.json({ models: MODEL_INFO });
});

// Model actions: generateContent, streamGenerateContent, countTokens
app.post('/v1beta/models/:model\\::action', (req: Request, res: Response) => {
  const body = req.body as Record<string, unknown>;
  const model = qs(req.params.model) ?? '';
  const action = qs(req.params.action) ?? '';

  if (!VALID_MODELS.has(model)) {
    const supported = [...VALID_MODELS].sort().join(', ');
    res.status(400).json({
      error: { code: 400, message: `Model not found: ${model}. Supported: ${supported}`, status: 'INVALID_ARGUMENT' },
    });
    return;
  }

  const endpoint = `/v1beta/models/${model}:${action}`;
  const params = buildConsoleParams(body);
  logConsole('POST', endpoint, params);
  log.log(endpoint, 'POST', body, model);

  const promptText = getPromptText(body);
  const trigger = extractErrorTrigger(promptText, qs(req.query.mock_error));

  if (action === 'countTokens') {
    if (trigger && trigger in GOOGLE_ERRORS) {
      const err = GOOGLE_ERRORS[trigger]!;
      res.status(err[0]).json({ error: { code: err[0], message: err[2], status: err[1] } });
      return;
    }
    const contents = body.contents;
    let total = countContentsTokens(contents);
    total += countSystemTokens(body.systemInstruction);
    res.json({ totalTokens: total });
    return;
  }

  if (action === 'generateContent') {
    if (trigger) {
      if (trigger in GOOGLE_ERRORS) {
        const err = GOOGLE_ERRORS[trigger]!;
        res.status(err[0]).json({ error: { code: err[0], message: err[2], status: err[1] } });
        return;
      }
      if (trigger === 'safety') { res.json(buildSafetyErrorResponse(model)); return; }
      if (trigger === 'content_filter') { res.json(buildContentFilterResponse(model)); return; }
      if (trigger === 'recitation') { res.json(buildRecitationResponse(model)); return; }
    }

    const alt = qs(req.query.alt);
    if (alt === 'sse') {
      sendStreamResponse(res, body, model, trigger);
    } else {
      res.json(buildGenerateResponse(body, model));
    }
    return;
  }

  if (action === 'streamGenerateContent') {
    sendStreamResponse(res, body, model, trigger);
    return;
  }

  res.status(400).json({
    error: { code: 400, message: `Unknown action: ${action}`, status: 'INVALID_ARGUMENT' },
  });
});

// ── Stream response helper ──────────────────────────────────────────

function sendStreamResponse(
  res: Response,
  body: Record<string, unknown>,
  model: string,
  trigger: string | null,
): void {
  if (trigger) {
    if (trigger in GOOGLE_ERRORS) {
      const err = GOOGLE_ERRORS[trigger]!;
      res.status(err[0]).json({ error: { code: err[0], message: err[2], status: err[1] } });
      return;
    }
    if (trigger === 'safety') {
      res.setHeader('Content-Type', 'text/event-stream');
      res.write(`data: ${JSON.stringify(buildSafetyErrorResponse(model))}\n\n`);
      res.end();
      return;
    }
    if (trigger === 'content_filter') {
      res.setHeader('Content-Type', 'text/event-stream');
      res.write(`data: ${JSON.stringify(buildContentFilterResponse(model))}\n\n`);
      res.end();
      return;
    }
    if (trigger === 'recitation') {
      res.setHeader('Content-Type', 'text/event-stream');
      res.write(`data: ${JSON.stringify(buildRecitationResponse(model))}\n\n`);
      res.end();
      return;
    }
  }

  const chunks = generateStreamChunks(body, model);

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  for (const chunk of chunks) {
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  }
  res.end();
}

// ── Start ───────────────────────────────────────────────────────────

const PORT = 18180;

app.listen(PORT, () => {
  console.log(`\n${'='.repeat(60)}`);
  console.log('  Mock Gemini Chat API Server');
  console.log(`  Port: ${PORT}`);
  console.log('='.repeat(60));
  console.log('\n  Supported models:');
  for (const m of [...VALID_MODELS].sort()) {
    const tag = m.includes('thinking') ? ' (thinking)' : '';
    console.log(`    - ${m}${tag}`);
  }
  console.log('\n  Endpoints:');
  console.log('    POST /v1beta/models/{model}:generateContent         (non-streaming)');
  console.log('    POST /v1beta/models/{model}:streamGenerateContent    (streaming SSE)');
  console.log('    POST /v1beta/models/{model}:generateContent?alt=sse  (alt streaming)');
  console.log('    POST /v1beta/models/{model}:countTokens              (token counting)');
  console.log('    GET  /v1beta/models                                  (list models)');
  console.log('\n  Monitoring:');
  console.log('    GET    /_mock/log                  (inspect request log)');
  console.log('    DELETE /_mock/log                  (clear request log)');
  console.log('\n  Error injection (via ?mock_error=<code> or MOCK_ERROR:<code> in prompt):');
  console.log('    HTTP errors:');
  for (const [code, [httpCode, status]] of Object.entries(GOOGLE_ERRORS).sort()) {
    console.log(`      ${code.padEnd(16)}  ${httpCode}  ${status}`);
  }
  console.log('    Content errors (200 status):');
  console.log('      safety            finishReason: SAFETY');
  console.log('      content_filter    promptFeedback blockReason: SAFETY');
  console.log('      recitation        finishReason: RECITATION');
  console.log('\n' + '='.repeat(60) + '\n');
});
