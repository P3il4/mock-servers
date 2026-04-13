/**
 * Mock Gemini Image/Video API Server (port 8080)
 *
 * Implements the Google Gemini native API for image and video generation,
 * plus an OpenAI-compatible chat/completions endpoint.
 *
 * Endpoints:
 *   POST /v1beta/models/:model:generateContent       Non-streaming
 *   POST /v1beta/models/:model:generateContent?alt=sse Streaming SSE
 *   POST /v1beta/models/:model:streamGenerateContent   SDK streaming
 *   POST /v1beta/models/:model:predict                 Imagen image gen
 *   POST /v1beta/models/:model:predictLongRunning      Veo video gen
 *   GET  /v1beta/operations/:id                        Poll video op
 *   POST /v1beta/openai/chat/completions               OpenAI compat
 *
 * Error injection:
 *   ?mock_error=<code>  or  MOCK_ERROR:<code> in prompt
 *
 * Usage:
 *   npx tsx src/gemini-image.ts
 */

import express, { type Request, type Response } from 'express';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  RequestLog, mockLogRouter, extractErrorTrigger, handleGoogleError,
  logConsole, MOCK_PNG_B64, MOCK_MP4_B64, corsMiddleware,
} from './shared.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** Extract a single string from an Express query/param value. */
function qs(v: unknown): string | undefined {
  if (typeof v === 'string') return v;
  if (Array.isArray(v) && typeof v[0] === 'string') return v[0];
  return undefined;
}

// ── Load image.png or fall back to MOCK_PNG_B64 ───────────────

let MOCK_IMAGE_B64 = MOCK_PNG_B64;
try {
  const imgPath = path.resolve(__dirname, '../image.png');
  MOCK_IMAGE_B64 = fs.readFileSync(imgPath).toString('base64');
} catch {
  // fallback already set
}

const MOCK_VIDEO_URI = 'https://generativelanguage.googleapis.com/v1beta/files/mock-video-abc123/content';

// ── In-memory store for long-running video operations ───────────────

interface VideoOp {
  model: string;
  numVideos: number;
  created: string;
  pollCount: number;
}

const videoOperations = new Map<string, VideoOp>();

// ── Image token tables ──────────────────────────────────────────────

const IMAGE_SIZE_TO_CANDIDATE_TOKENS: Record<string, number> = {
  '1K': 1120,
  '2K': 1120,
  '4K': 2000,
};

const GEMINI_3_1_FLASH_IMAGE_TOKENS: Record<string, number> = {
  '512': 747,
  '0.5K': 747,
  '1K': 1120,
  '2K': 1680,
  '4K': 2520,
};

const IMAGE_CHAT_MODELS = new Set([
  'gemini-2.5-flash-image',
  'gemini-3-pro-image-preview',
  'gemini-3.1-flash-image-preview',
]);

// ── Helpers ─────────────────────────────────────────────────────────

function countInlineImages(body: Record<string, unknown>): number {
  let count = 0;
  const contents = body.contents as Array<Record<string, unknown>> | undefined;
  if (!Array.isArray(contents)) return 0;
  for (const content of contents) {
    const parts = content.parts as Array<Record<string, unknown>> | undefined;
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      if (part.inlineData) count++;
    }
  }
  return count;
}

function getImageSize(body: Record<string, unknown>): string | null {
  const genConfig = (body.generationConfig ?? {}) as Record<string, unknown>;
  const ic1 = (genConfig.imageConfig ?? {}) as Record<string, unknown>;
  if (ic1.imageSize) return ic1.imageSize as string;
  const config = (body.config ?? {}) as Record<string, unknown>;
  const ic2 = (config.imageConfig ?? {}) as Record<string, unknown>;
  return (ic2.imageSize as string) ?? null;
}

function getPromptText(body: Record<string, unknown>): string {
  const contents = body.contents as Array<Record<string, unknown>> | undefined;
  if (!Array.isArray(contents)) return '';
  for (const content of contents) {
    const parts = content.parts as Array<Record<string, unknown>> | undefined;
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      if (typeof part.text === 'string') return part.text;
    }
  }
  return '';
}

function getPromptFromBody(body: Record<string, unknown>): string {
  let prompt = getPromptText(body);
  if (!prompt) {
    const instances = body.instances as Array<Record<string, unknown>> | undefined;
    if (Array.isArray(instances) && instances.length > 0) {
      prompt = (instances[0]!.prompt as string) ?? '';
    }
  }
  if (!prompt) {
    const messages = body.messages as Array<Record<string, unknown>> | undefined;
    if (Array.isArray(messages)) {
      for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i]!;
        if (msg.role !== 'user') continue;
        const content = msg.content;
        if (typeof content === 'string') { prompt = content; break; }
        if (Array.isArray(content)) {
          for (const part of content) {
            if (typeof part === 'object' && part !== null && (part as Record<string, unknown>).type === 'text') {
              prompt = ((part as Record<string, unknown>).text as string) ?? '';
              break;
            }
          }
        }
        break;
      }
    }
  }
  return prompt;
}

function summarizeMedia(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    out[k] = typeof v === 'string' && v.length > 200 ? `[${v.length} chars]` : v;
  }
  return out;
}

function buildConsoleParams(body: Record<string, unknown>): Record<string, unknown> {
  const logData: Record<string, unknown> = {};
  for (const key of ['generationConfig', 'parameters']) {
    if (body[key]) logData[key] = body[key];
  }
  const contents = body.contents as Array<Record<string, unknown>> | undefined;
  if (Array.isArray(contents)) {
    for (const content of contents) {
      const parts = content.parts as Array<Record<string, unknown>> | undefined;
      if (!Array.isArray(parts)) continue;
      for (const part of parts) {
        if (typeof part.text === 'string') logData.prompt = part.text;
        if (part.inlineData) logData.inlineImages = ((logData.inlineImages as number) ?? 0) + 1;
      }
    }
  }
  const instances = body.instances as Array<Record<string, unknown>> | undefined;
  if (Array.isArray(instances) && instances.length > 0) {
    const inst = instances[0]!;
    if (inst.prompt) logData.prompt = inst.prompt;
    if (inst.image) logData.image = summarizeMedia(inst.image as Record<string, unknown>);
    if (inst.lastFrame) logData.lastFrame = summarizeMedia(inst.lastFrame as Record<string, unknown>);
    if (inst.video) logData.video = summarizeMedia(inst.video as Record<string, unknown>);
    if (inst.referenceImages) {
      logData.referenceImages = (inst.referenceImages as Array<Record<string, unknown>>).map(ref => {
        const mapped: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(ref)) {
          mapped[k] = k === 'image' ? summarizeMedia(v as Record<string, unknown>) : v;
        }
        return mapped;
      });
    }
    if (inst.mask) logData.mask = inst.mask;
  }
  return logData;
}

function extractModel(path: string): string {
  if (!path.includes('/models/')) return '';
  const afterModels = path.split('/models/')[1]!;
  return afterModels.split(':')[0]!.split('?')[0]!;
}

function isoNow(): string {
  return new Date().toISOString().replace(/\.\d{3}Z/, '.000Z');
}

// ── generateContent (non-streaming) ─────────────────────────────────

function handleGenerateContent(body: Record<string, unknown>, model: string): Record<string, unknown> {
  const config = (body.generationConfig ?? {}) as Record<string, unknown>;
  const modalities = (config.responseModalities ?? []) as string[];

  const basePromptTokens = 10;
  const imageCount = countInlineImages(body);
  const promptText = getPromptText(body);
  const textTokens = promptText ? Math.max(1, Math.floor(promptText.length / 4)) : 0;
  const promptTokenCount = Math.max(basePromptTokens, textTokens) + imageCount * 550;

  const promptTokensDetails = [{ modality: 'TEXT', tokenCount: promptTokenCount }];

  const imageSize = getImageSize(body);
  const candidatesTokensDetails: Array<Record<string, unknown>> = [];
  const candidatesTextTokens = 28;
  let candidatesImageTokens = 0;

  if (model === 'gemini-2.5-flash-image') {
    candidatesImageTokens = 1290;
    candidatesTokensDetails.push({ modality: 'IMAGE', tokenCount: 1290 });
  } else if (model === 'gemini-3.1-flash-image-preview') {
    candidatesImageTokens = imageSize ? (GEMINI_3_1_FLASH_IMAGE_TOKENS[imageSize] ?? 1120) : 1120;
    candidatesTokensDetails.push({ modality: 'IMAGE', tokenCount: candidatesImageTokens });
  } else if (modalities.includes('IMAGE')) {
    candidatesImageTokens = imageSize ? (IMAGE_SIZE_TO_CANDIDATE_TOKENS[imageSize] ?? 1120) : 1120;
    candidatesTokensDetails.push({ modality: 'IMAGE', tokenCount: candidatesImageTokens });
  }

  const candidatesTokenCount = candidatesImageTokens + candidatesTextTokens;
  const totalTokenCount = promptTokenCount + candidatesTokenCount;

  console.log(`  -> Model: ${model}, Images in: ${imageCount}, imageSize: ${imageSize}`);
  console.log(`  -> prompt: ${promptTokenCount}, candidates: ${candidatesTokenCount}, total: ${totalTokenCount}`);

  const parts: Array<Record<string, unknown>> = [
    { text: `Mock response for model ${model}. Your prompt was: ${promptText.slice(0, 100)}` },
  ];

  if (modalities.includes('IMAGE')) {
    parts.push({ inlineData: { mimeType: 'image/png', data: MOCK_IMAGE_B64 } });
  }

  return {
    candidates: [{
      content: { role: 'model', parts },
      finishReason: 'STOP',
    }],
    usageMetadata: {
      promptTokenCount,
      promptTokensDetails,
      candidatesTokenCount,
      candidatesTokensDetails,
      totalTokenCount,
    },
  };
}

// ── generateContent streaming (SSE) ─────────────────────────────────

function handleGenerateContentStream(
  res: Response,
  body: Record<string, unknown>,
  model: string,
): void {
  const promptText = getPromptText(body);
  const config = (body.generationConfig ?? body.config ?? {}) as Record<string, unknown>;
  const modalities = (config.responseModalities ?? []) as string[];

  const imageCount = countInlineImages(body);
  const textTokens = promptText ? Math.max(1, Math.floor(promptText.length / 4)) : 0;
  const promptTokenCount = Math.max(10, textTokens) + imageCount * 550;

  const imageSize = getImageSize(body);

  const responseText = `Mock streaming response for ${model}.`;
  const words = responseText.split(' ');

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  for (let i = 0; i < words.length; i++) {
    const chunkText = words[i]! + (i < words.length - 1 ? ' ' : '');
    const event: Record<string, unknown> = {
      candidates: [{
        content: { role: 'model', parts: [{ text: chunkText }] },
      }],
    };

    if (i === words.length - 1) {
      const candidate = (event.candidates as Array<Record<string, unknown>>)[0]!;
      candidate.finishReason = 'STOP';

      const candidatesTextTokens = words.length * 2;
      let candidatesImageTokens = 0;
      const candidatesTokensDetails: Array<Record<string, unknown>> = [];

      if (model === 'gemini-2.5-flash-image') {
        candidatesImageTokens = 1290;
      } else if (model === 'gemini-3.1-flash-image-preview') {
        candidatesImageTokens = imageSize ? (GEMINI_3_1_FLASH_IMAGE_TOKENS[imageSize] ?? 1120) : 1120;
      } else if (modalities.includes('IMAGE')) {
        candidatesImageTokens = imageSize ? (IMAGE_SIZE_TO_CANDIDATE_TOKENS[imageSize] ?? 1120) : 1120;
      }

      if (candidatesImageTokens > 0) {
        const parts = (candidate.content as Record<string, unknown>).parts as Array<Record<string, unknown>>;
        parts.push({ inlineData: { mimeType: 'image/png', data: MOCK_IMAGE_B64 } });
        candidatesTokensDetails.push({ modality: 'IMAGE', tokenCount: candidatesImageTokens });
      }

      const candidatesTokenCount = candidatesTextTokens + candidatesImageTokens;
      event.usageMetadata = {
        promptTokenCount,
        candidatesTokenCount,
        candidatesTokensDetails,
        totalTokenCount: promptTokenCount + candidatesTokenCount,
      };
    }

    res.write(`data: ${JSON.stringify(event)}\n\n`);
  }

  res.end();
}

// ── predict (Imagen image generation) ───────────────────────────────

function handlePredict(res: Response, body: Record<string, unknown>, model: string): void {
  const instances = (body.instances ?? []) as Array<Record<string, unknown>>;
  const parameters = (body.parameters ?? {}) as Record<string, unknown>;
  const sampleCount = (parameters.sampleCount as number) ?? 1;
  const prompt = instances.length > 0 ? (instances[0]!.prompt as string ?? '') : '';

  console.log(`  -> Imagen predict: model=${model}, prompt=${prompt.slice(0, 80)}, count=${sampleCount}`);

  const outputOpts = (parameters.outputOptions ?? {}) as Record<string, unknown>;
  const mimeType = (outputOpts.mimeType as string) ?? 'image/png';

  const predictions: Array<Record<string, unknown>> = [];
  for (let i = 0; i < sampleCount; i++) {
    predictions.push({ bytesBase64Encoded: MOCK_IMAGE_B64, mimeType });
  }

  const response: Record<string, unknown> = { predictions };

  if (parameters.includeSafetyAttributes) {
    response.positivePromptSafetyAttributes = {
      categories: [],
      scores: [],
      contentType: 'Positive Prompt',
    };
  }

  res.json(response);
}

// ── predictLongRunning (Veo video generation) ───────────────────────

function handlePredictLongRunning(res: Response, body: Record<string, unknown>, model: string): void {
  const instances = (body.instances ?? []) as Array<Record<string, unknown>>;
  const parameters = (body.parameters ?? {}) as Record<string, unknown>;
  const prompt = instances.length > 0 ? (instances[0]!.prompt as string ?? '') : '';
  const numVideos = (parameters.sampleCount as number)
    ?? (parameters.numberOfVideos as number)
    ?? 1;

  const opId = crypto.randomUUID();
  const now = isoNow();

  console.log(`  -> Veo predictLongRunning: model=${model}, prompt=${prompt.slice(0, 80)}, videos=${numVideos}`);
  console.log(`  -> Operation created: ${opId}`);

  videoOperations.set(opId, { model, numVideos, created: now, pollCount: 0 });

  res.json({
    name: `operations/${opId}`,
    metadata: { createTime: now, updateTime: now },
    done: false,
  });
}

function handlePollOperation(res: Response, opId: string): void {
  const op = videoOperations.get(opId);
  if (!op) {
    res.status(404).json({ error: { message: `Operation not found: ${opId}` } });
    return;
  }

  op.pollCount++;
  const now = isoNow();

  if (op.pollCount >= 1) {
    const samples: Array<Record<string, unknown>> = [];
    for (let i = 0; i < op.numVideos; i++) {
      if (Math.random() < 0.5) {
        samples.push({ video: { uri: MOCK_VIDEO_URI, encoding: 'video/mp4' } });
      } else {
        samples.push({ video: { encodedVideo: MOCK_MP4_B64, encoding: 'video/mp4' } });
      }
    }

    videoOperations.delete(opId);
    console.log(`  -> Operation ${opId} resolved (${op.numVideos} videos)`);

    res.json({
      name: `operations/${opId}`,
      metadata: { createTime: op.created, updateTime: now },
      done: true,
      response: {
        generateVideoResponse: {
          generatedSamples: samples,
          raiMediaFilteredCount: 0,
          raiMediaFilteredReasons: [],
        },
      },
    });
  } else {
    console.log(`  -> Operation ${opId} still pending (poll #${op.pollCount})`);
    res.json({
      name: `operations/${opId}`,
      metadata: { createTime: op.created, updateTime: now },
      done: false,
    });
  }
}

// ── OpenAI-compatible chat completions ──────────────────────────────

function handleOpenAIChatCompletions(
  res: Response,
  body: Record<string, unknown>,
  queryMockError: string | undefined,
): void {
  const model = (body.model as string) ?? 'gemini-2.5-flash';
  const messages = (body.messages ?? []) as Array<Record<string, unknown>>;
  const stream = body.stream as boolean ?? false;
  const tools = (body.tools ?? []) as Array<Record<string, unknown>>;

  let lastUserMsg = '';
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!;
    if (msg.role !== 'user') continue;
    const content = msg.content;
    if (typeof content === 'string') { lastUserMsg = content; break; }
    if (Array.isArray(content)) {
      for (const part of content) {
        if (typeof part === 'object' && part !== null && (part as Record<string, unknown>).type === 'text') {
          lastUserMsg = ((part as Record<string, unknown>).text as string) ?? '';
          break;
        }
      }
    }
    break;
  }

  const promptTokens = Math.max(1, messages.reduce(
    (sum, m) => sum + Math.floor(JSON.stringify(m.content ?? '').length / 4), 0,
  ));

  const responseText = `Mock Gemini (${model}) response. You said: ${lastUserMsg.slice(0, 200)}`;
  const completionId = `chatcmpl-${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`;
  const created = Math.floor(Date.now() / 1000);
  const completionTokens = Math.max(1, Math.floor(responseText.length / 4));

  console.log(`  -> OpenAI-compat chat: model=${model}, stream=${stream}, msgs=${messages.length}`);

  let toolCallsResponse: Array<Record<string, unknown>> | null = null;
  if (tools.length > 0 && lastUserMsg) {
    const firstTool = tools[0]! as Record<string, unknown>;
    const funcDef = (firstTool.function ?? {}) as Record<string, unknown>;
    const funcName = (funcDef.name as string) ?? 'unknown';
    toolCallsResponse = [{
      id: `call_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`,
      type: 'function',
      function: {
        name: funcName,
        arguments: JSON.stringify({ query: lastUserMsg.slice(0, 50) }),
      },
    }];
  }

  if (stream) {
    openAIStream(res, completionId, created, model, responseText, promptTokens, completionTokens, toolCallsResponse);
  } else {
    openAINonStream(res, completionId, created, model, responseText, promptTokens, completionTokens, toolCallsResponse);
  }
}

function openAINonStream(
  res: Response,
  completionId: string, created: number, model: string,
  responseText: string, promptTokens: number, completionTokens: number,
  toolCalls: Array<Record<string, unknown>> | null,
): void {
  const message: Record<string, unknown> = { role: 'assistant' };
  let finishReason = 'stop';

  if (toolCalls) {
    message.content = null;
    message.tool_calls = toolCalls;
    finishReason = 'tool_calls';
  } else if (IMAGE_CHAT_MODELS.has(model)) {
    message.content = [
      { type: 'text', text: responseText },
      { type: 'image_url', image_url: { url: `data:image/png;base64,${MOCK_IMAGE_B64}` } },
    ];
  } else {
    message.content = responseText;
  }

  res.json({
    id: completionId,
    object: 'chat.completion',
    created,
    model,
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
      prompt_tokens_details: { cached_tokens: 0 },
    },
  });
}

function openAIStream(
  res: Response,
  completionId: string, created: number, model: string,
  responseText: string, promptTokens: number, completionTokens: number,
  toolCalls: Array<Record<string, unknown>> | null,
): void {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const writeEvent = (data: Record<string, unknown>) => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  writeEvent({
    id: completionId, object: 'chat.completion.chunk',
    created, model,
    choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
  });

  let finishReason = 'stop';

  if (toolCalls) {
    const tc = toolCalls[0]!;
    writeEvent({
      id: completionId, object: 'chat.completion.chunk',
      created, model,
      choices: [{ index: 0, delta: { tool_calls: [{
        index: 0, id: tc.id, type: 'function',
        function: (tc as Record<string, unknown>).function,
      }] }, finish_reason: null }],
    });
    finishReason = 'tool_calls';
  } else {
    const words = responseText.split(' ');
    for (let i = 0; i < words.length; i++) {
      const chunkText = words[i]! + (i < words.length - 1 ? ' ' : '');
      writeEvent({
        id: completionId, object: 'chat.completion.chunk',
        created, model,
        choices: [{ index: 0, delta: { content: chunkText }, finish_reason: null }],
      });
    }
  }

  writeEvent({
    id: completionId, object: 'chat.completion.chunk',
    created, model,
    choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
  });

  writeEvent({
    id: completionId, object: 'chat.completion.chunk',
    created, model,
    choices: [],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
      prompt_tokens_details: { cached_tokens: 0 },
    },
  });

  res.write('data: [DONE]\n\n');
  res.end();
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
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-goog-api-client, x-goog-api-key');
  next();
});

app.options('*', (_req, res) => res.sendStatus(200));

// Poll long-running video operation
app.get('/v1beta/operations/:opId', (req: Request, res: Response) => {
  logConsole('GET', req.path);
  handlePollOperation(res, qs(req.params.opId) ?? '');
});

// Native Gemini routes  (model param includes action after colon)
app.post('/v1beta/models/:model\\::action', (req: Request, res: Response) => {
  const body = req.body as Record<string, unknown>;
  const model = qs(req.params.model) ?? '';
  const action = qs(req.params.action) ?? '';
  const params = buildConsoleParams(body);

  logConsole('POST', req.path, params);

  const prompt = getPromptFromBody(body);
  const trigger = extractErrorTrigger(prompt, qs(req.query.mock_error));
  if (trigger) {
    if (handleGoogleError(res, trigger)) {
      log.log('error_trigger', 'POST', { trigger }, model);
      return;
    }
  }

  switch (action) {
    case 'streamGenerateContent':
      log.log('generateContent', 'POST', body, model);
      handleGenerateContentStream(res, body, model);
      break;

    case 'generateContent': {
      log.log('generateContent', 'POST', body, model);
      const alt = qs(req.query.alt);
      if (alt === 'sse') {
        handleGenerateContentStream(res, body, model);
      } else {
        res.json(handleGenerateContent(body, model));
      }
      break;
    }

    case 'predictLongRunning':
      log.log('predictLongRunning', 'POST', body, model);
      handlePredictLongRunning(res, body, model);
      break;

    case 'predict':
      log.log('predict', 'POST', body, model);
      handlePredict(res, body, model);
      break;

    default:
      res.status(404).json({ error: 'Not Found' });
  }
});

// OpenAI-compatible chat completions
app.post('/v1beta/openai/chat/completions', (req: Request, res: Response) => {
  const body = req.body as Record<string, unknown>;
  const model = (body.model as string) ?? '';
  const params = buildConsoleParams(body);

  logConsole('POST', req.path, params);

  const prompt = getPromptFromBody(body);
  const mockError = qs(req.query.mock_error);
  const trigger = extractErrorTrigger(prompt, mockError);
  if (trigger) {
    if (handleGoogleError(res, trigger)) {
      log.log('error_trigger', 'POST', { trigger }, model);
      return;
    }
  }

  log.log('chat/completions', 'POST', body, model);
  handleOpenAIChatCompletions(res, body, mockError);
});

// ── Start ───────────────────────────────────────────────────────────

const PORT = 18080;

app.listen(PORT, () => {
  console.log(`\n${'='.repeat(60)}`);
  console.log('  Mock Gemini Image/Video API Server');
  console.log(`  Port: ${PORT}`);
  console.log('='.repeat(60));
  console.log('\n  Endpoints:');
  console.log('    Native Gemini API:');
  console.log('      POST /v1beta/models/{model}:generateContent       (image gen / chat)');
  console.log('      POST /v1beta/models/{model}:generateContent?alt=sse (streaming)');
  console.log('      POST /v1beta/models/{model}:streamGenerateContent   (SDK streaming)');
  console.log('      POST /v1beta/models/{model}:predict                 (Imagen)');
  console.log('      POST /v1beta/models/{model}:predictLongRunning      (Veo video)');
  console.log('      GET  /v1beta/operations/{id}                        (poll video op)');
  console.log('    OpenAI-compatible:');
  console.log('      POST /v1beta/openai/chat/completions                (chat)');
  console.log('    Testing:');
  console.log('      GET    /_mock/log                                    (inspect params)');
  console.log('      DELETE /_mock/log                                    (clear log)');
  console.log('      ?mock_error=<code>         (trigger error via query)');
  console.log('      MOCK_ERROR:<code> in prompt (trigger error via prompt)');
  console.log('      Codes: auth, not_found, rate_limit, quota, invalid_model,');
  console.log('             server_error, timeout, permission, unavailable,');
  console.log('             invalid_arg, safety, content_filter, recitation');
  console.log(`\n  Use baseUrl:  'http://localhost:${PORT}' for @google/genai SDK`);
  console.log(`  Use baseURL: 'http://localhost:${PORT}/v1beta/openai/' for OpenAI SDK`);
  console.log('='.repeat(60) + '\n');
});
