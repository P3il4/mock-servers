/**
 * Mock OpenAI Image Generation API Server (DALL-E 2 / DALL-E 3 / gpt-image-1).
 *
 * Drop-in replacement for testing against the real API.
 * Point your OpenAI client at http://localhost:18150/v1
 *
 *   npx tsx src/openai-image.ts
 *
 * Monitoring:
 *   GET  /_mock/log                              -- view recent request log
 *   GET  /_mock/log?endpoint=/v1/images/generations
 *   DELETE /_mock/log                            -- clear log
 *
 * Error injection (in prompt text or query param):
 *   prompt="A cat MOCK_ERROR:rate_limit"  -- triggers 429 rate limit error
 *   POST /v1/images/generations?mock_error=auth  -- triggers 401 auth error
 */

import express, { type Request, type Response } from 'express';
import crypto from 'node:crypto';
import {
  RequestLog, mockLogRouter, extractErrorTrigger, handleOpenAIError,
  requireBearerAuth, logConsole, generatePng, corsMiddleware,
} from './shared.js';

// ── Constants ───────────────────────────────────────────────────────

const PORT = 18150;

const VALID_MODELS = new Set([
  'dall-e-2', 'dall-e-3', 'gpt-image-1', 'gpt-image-1-mini', 'gpt-image-1.5',
]);

const VALID_SIZES_DALLE2 = new Set(['256x256', '512x512', '1024x1024']);
const VALID_SIZES_DALLE3 = new Set(['1024x1024', '1792x1024', '1024x1792']);
const VALID_SIZES_GPT_IMAGE = new Set(['auto', '1024x1024', '1536x1024', '1024x1536']);

const VALID_QUALITY = new Set(['standard', 'hd', 'low', 'medium', 'high', 'auto']);
const VALID_STYLES = new Set(['vivid', 'natural']);
const VALID_RESPONSE_FORMATS = new Set(['url', 'b64_json']);
const VALID_OUTPUT_FORMATS = new Set(['png', 'jpeg', 'webp']);

// ── In-memory image store for CDN ───────────────────────────────────

const imageStore = new Map<string, Buffer>();

// ── Color palette for multiple images ───────────────────────────────

const COLORS: [number, number, number][] = [
  [100, 149, 237], // cornflower blue
  [255, 127, 80],  // coral
  [50, 205, 50],   // lime green
  [255, 215, 0],   // gold
  [186, 85, 211],  // medium orchid
  [0, 206, 209],   // dark turquoise
  [255, 99, 71],   // tomato
  [60, 179, 113],  // medium sea green
  [238, 130, 238], // violet
  [30, 144, 255],  // dodger blue
];

// ── Helpers ─────────────────────────────────────────────────────────

function isGptImageModel(model: string): boolean {
  return model.startsWith('gpt-image-');
}

function parseSize(sizeStr: string): [number, number] {
  const [w, h] = sizeStr.split('x').map(Number);
  return [w!, h!];
}

function mapQualityForResponse(quality: string): string {
  const mapping: Record<string, string> = {
    standard: 'medium', hd: 'high', auto: 'medium',
    low: 'low', medium: 'medium', high: 'high',
  };
  return mapping[quality] ?? 'medium';
}

function buildImageData(
  n: number, size: string, responseFormat: string,
  model: string, prompt: string, host: string,
): Record<string, unknown>[] {
  const [w, h] = parseSize(size);
  const images: Record<string, unknown>[] = [];

  for (let i = 0; i < n; i++) {
    const [r, g, b] = COLORS[i % COLORS.length]!;
    const pngBytes = generatePng(w, h, r, g, b);
    const img: Record<string, unknown> = {};

    if (responseFormat === 'b64_json') {
      img.b64_json = pngBytes.toString('base64');
    } else {
      const imageId = crypto.randomUUID().replace(/-/g, '');
      imageStore.set(imageId, pngBytes);
      img.url = `http://${host}/v1/mock_cdn/${imageId}.png`;
    }

    if (model === 'dall-e-3') {
      img.revised_prompt = `A high-quality, detailed image of: ${prompt}`;
    }

    images.push(img);
  }
  return images;
}

function buildUsage(n: number, prompt: string): Record<string, unknown> {
  const textTokens = Math.max(1, prompt.split(/\s+/).length);
  const imageTokens = 1024 * n;
  return {
    input_tokens: textTokens,
    input_tokens_details: { image_tokens: 0, text_tokens: textTokens },
    output_tokens: imageTokens,
    total_tokens: textTokens + imageTokens,
    output_tokens_details: { image_tokens: imageTokens, text_tokens: 0 },
  };
}

function buildEditUsage(n: number, prompt: string, hasImage: boolean): Record<string, unknown> {
  const textTokens = Math.max(1, prompt.split(/\s+/).length);
  const inputImageTokens = hasImage ? 512 : 0;
  const imageTokens = 1024 * n;
  return {
    input_tokens: textTokens + inputImageTokens,
    input_tokens_details: { image_tokens: inputImageTokens, text_tokens: textTokens },
    output_tokens: imageTokens,
    total_tokens: textTokens + inputImageTokens + imageTokens,
    output_tokens_details: { image_tokens: imageTokens, text_tokens: 0 },
  };
}

// ── Minimal multipart parser ────────────────────────────────────────

interface MultipartResult {
  fields: Record<string, string>;
  fileFields: Set<string>;
}

function parseMultipart(req: Request): Promise<MultipartResult> {
  return new Promise((resolve, reject) => {
    const contentType = req.headers['content-type'] ?? '';
    const boundaryMatch = contentType.match(/boundary=(?:"([^"]+)"|([^\s;]+))/);
    if (!boundaryMatch) {
      reject(new Error('No multipart boundary found'));
      return;
    }
    const boundary = boundaryMatch[1] ?? boundaryMatch[2]!;

    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('error', reject);
    req.on('end', () => {
      const body = Buffer.concat(chunks);
      const fields: Record<string, string> = {};
      const fileFields = new Set<string>();

      const boundaryBuf = Buffer.from(`--${boundary}`);
      const parts: Buffer[] = [];
      let start = 0;

      while (true) {
        const idx = body.indexOf(boundaryBuf, start);
        if (idx === -1) break;
        if (start > 0) {
          // Strip leading \r\n and trailing \r\n before boundary
          let partStart = start;
          let partEnd = idx;
          if (body[partStart] === 0x0d && body[partStart + 1] === 0x0a) partStart += 2;
          if (partEnd >= 2 && body[partEnd - 2] === 0x0d && body[partEnd - 1] === 0x0a) partEnd -= 2;
          if (partEnd > partStart) parts.push(body.subarray(partStart, partEnd));
        }
        start = idx + boundaryBuf.length;
        // Skip past -- terminator
        if (body[start] === 0x2d && body[start + 1] === 0x2d) break;
      }

      for (const part of parts) {
        const headerEnd = part.indexOf('\r\n\r\n');
        if (headerEnd === -1) continue;

        const headerStr = part.subarray(0, headerEnd).toString('utf-8');
        const valueBytes = part.subarray(headerEnd + 4);

        const nameMatch = headerStr.match(/name="([^"]+)"/);
        if (!nameMatch) continue;
        const name = nameMatch[1]!;

        const isFile = /filename=/.test(headerStr);
        if (isFile) {
          fileFields.add(name);
        } else {
          fields[name] = valueBytes.toString('utf-8');
        }
      }

      resolve({ fields, fileFields });
    });
  });
}

// ── App ─────────────────────────────────────────────────────────────

const app = express();
app.use(corsMiddleware);
app.use((req, _res, next) => {
  const ct = req.headers['content-type'] ?? '';
  if (ct.includes('multipart/form-data')) {
    next();
  } else {
    express.json()(req, _res, next);
  }
});

const log = new RequestLog();
app.use(mockLogRouter(log));

// ── Mock CDN endpoint ───────────────────────────────────────────────

app.get('/v1/mock_cdn/:imageFile', (req: Request, res: Response) => {
  const imageId = (req.params.imageFile as string).replace(/\.png$/, '');
  const pngBytes = imageStore.get(imageId);
  if (!pngBytes) {
    res.status(404).json({
      error: {
        message: `Image '${imageId}' not found or has expired.`,
        type: 'invalid_request_error',
        code: 'not_found',
      },
    });
    return;
  }
  res.setHeader('Content-Type', 'image/png');
  res.setHeader('Content-Disposition', `inline; filename="${imageId}.png"`);
  res.send(pngBytes);
});

// ── POST /v1/images/generations ─────────────────────────────────────

app.post('/v1/images/generations', (req: Request, res: Response) => {
  let trigger = req.query.mock_error as string | undefined;
  if (trigger) {
    log.log('/v1/images/generations', 'POST', { mock_error: trigger });
    if (handleOpenAIError(res, trigger)) return;
  }

  if (!requireBearerAuth(req, res, 'openai')) return;

  const data = req.body ?? {};
  const prompt: string = data.prompt ?? '';
  const model: string = data.model ?? 'dall-e-2';
  const n: number = data.n ?? 1;
  let size: string | undefined = data.size;
  const quality: string = data.quality ?? 'standard';
  const style: string | undefined = data.style;
  let responseFormat: string = data.response_format ?? 'url';
  const background: string | undefined = data.background;
  const outputFormat: string | undefined = data.output_format;

  const params: Record<string, unknown> = {
    prompt, model, n, size: size ?? null,
    quality, style: style ?? null, response_format: responseFormat,
    background: background ?? null, output_format: outputFormat ?? null,
  };
  log.log('/v1/images/generations', 'POST', params, model);
  logConsole('POST', '/v1/images/generations', params);

  if (!trigger) {
    trigger = extractErrorTrigger(prompt, req.query.mock_error as string) ?? undefined;
  }
  if (trigger) {
    if (handleOpenAIError(res, trigger)) return;
  }

  // -- Validation --
  if (!prompt) {
    res.status(400).json({
      error: { message: "'prompt' is required.", type: 'invalid_request_error', code: 'missing_required_parameter' },
    });
    return;
  }

  if (!VALID_MODELS.has(model)) {
    res.status(400).json({
      error: { message: `The model '${model}' does not exist or you do not have access to it.`, type: 'invalid_request_error', code: 'model_not_found' },
    });
    return;
  }

  const isGpt = isGptImageModel(model);

  if (size === undefined || size === null) {
    if (model === 'dall-e-2') size = '1024x1024';
    else if (model === 'dall-e-3') size = '1024x1024';
    else size = 'auto';
  }

  if (size === 'auto' && isGpt) {
    size = '1024x1024';
  }

  if (model === 'dall-e-2' && !VALID_SIZES_DALLE2.has(size)) {
    const valid = [...VALID_SIZES_DALLE2].sort().join(', ');
    res.status(400).json({
      error: { message: `Invalid size '${size}' for dall-e-2. Must be one of: ${valid}.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (model === 'dall-e-3' && !VALID_SIZES_DALLE3.has(size)) {
    const valid = [...VALID_SIZES_DALLE3].sort().join(', ');
    res.status(400).json({
      error: { message: `Invalid size '${size}' for dall-e-3. Must be one of: ${valid}.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (isGpt && !VALID_SIZES_GPT_IMAGE.has(size) && size !== 'auto') {
    const valid = [...VALID_SIZES_GPT_IMAGE].filter(s => s !== 'auto').sort().join(', ');
    res.status(400).json({
      error: { message: `Invalid size '${size}' for ${model}. Must be one of: ${valid}.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (typeof n !== 'number' || n < 1 || n > 10) {
    res.status(400).json({
      error: { message: "'n' must be between 1 and 10.", type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (model === 'dall-e-3' && n !== 1) {
    res.status(400).json({
      error: { message: 'For dall-e-3, only n=1 is supported.', type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (quality && !VALID_QUALITY.has(quality)) {
    res.status(400).json({
      error: { message: `Invalid quality '${quality}'. Must be one of: ${[...VALID_QUALITY].sort().join(', ')}.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (style && model !== 'dall-e-3') {
    res.status(400).json({
      error: { message: "'style' is only supported for dall-e-3.", type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (style && !VALID_STYLES.has(style)) {
    res.status(400).json({
      error: { message: `Invalid style '${style}'. Must be one of: ${[...VALID_STYLES].sort().join(', ')}.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (isGpt) {
    responseFormat = 'b64_json';
  } else if (!VALID_RESPONSE_FORMATS.has(responseFormat)) {
    res.status(400).json({
      error: { message: `Invalid response_format '${responseFormat}'. Must be 'url' or 'b64_json'.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (outputFormat && !isGpt) {
    res.status(400).json({
      error: { message: "'output_format' is only supported for GPT image models.", type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (outputFormat && !VALID_OUTPUT_FORMATS.has(outputFormat)) {
    res.status(400).json({
      error: { message: `Invalid output_format '${outputFormat}'. Must be one of: ${[...VALID_OUTPUT_FORMATS].sort().join(', ')}.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  // -- Build response --
  const imageData = buildImageData(n, size, responseFormat, model, prompt, req.headers.host ?? 'localhost:18150');
  const result: Record<string, unknown> = {
    created: Math.floor(Date.now() / 1000),
    data: imageData,
  };

  if (isGpt) {
    result.background = background ?? 'auto';
    result.output_format = outputFormat ?? 'png';
    result.quality = mapQualityForResponse(quality);
    result.size = size;
    result.usage = buildUsage(n, prompt);
  }

  res.status(200).json(result);
});

// ── POST /v1/images/edits ───────────────────────────────────────────

app.post('/v1/images/edits', async (req: Request, res: Response) => {
  let trigger = req.query.mock_error as string | undefined;
  if (trigger) {
    log.log('/v1/images/edits', 'POST', { mock_error: trigger });
    if (handleOpenAIError(res, trigger)) return;
  }

  if (!requireBearerAuth(req, res, 'openai')) return;

  let parsed: MultipartResult;
  try {
    parsed = await parseMultipart(req);
  } catch {
    res.status(400).json({
      error: { message: 'Failed to parse multipart form data.', type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  const { fields, fileFields } = parsed;
  const prompt = fields.prompt ?? '';
  const model = fields.model ?? 'gpt-image-1.5';
  const n = parseInt(fields.n ?? '1');
  let size: string | undefined = fields.size;
  const quality = fields.quality ?? 'auto';
  let responseFormat = fields.response_format ?? 'url';
  const background: string | undefined = fields.background;
  const outputFormat: string | undefined = fields.output_format;
  const hasImage = fileFields.has('image');
  const hasMask = fileFields.has('mask');

  const params: Record<string, unknown> = {
    prompt, model, n, size: size ?? null,
    quality, response_format: responseFormat,
    has_image: hasImage, has_mask: hasMask,
    background: background ?? null, output_format: outputFormat ?? null,
  };
  log.log('/v1/images/edits', 'POST', params, model);
  logConsole('POST', '/v1/images/edits', params);

  if (!trigger) {
    trigger = extractErrorTrigger(prompt, req.query.mock_error as string) ?? undefined;
  }
  if (trigger) {
    if (handleOpenAIError(res, trigger)) return;
  }

  // -- Validation --
  if (!prompt) {
    res.status(400).json({
      error: { message: "'prompt' is required.", type: 'invalid_request_error', code: 'missing_required_parameter' },
    });
    return;
  }

  if (!hasImage) {
    res.status(400).json({
      error: { message: "'image' is required.", type: 'invalid_request_error', code: 'missing_required_parameter' },
    });
    return;
  }

  if (!VALID_MODELS.has(model)) {
    res.status(400).json({
      error: { message: `The model '${model}' does not exist or you do not have access to it.`, type: 'invalid_request_error', code: 'model_not_found' },
    });
    return;
  }

  if (model === 'dall-e-3') {
    res.status(400).json({
      error: { message: 'Image edits are not supported for dall-e-3.', type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  const isGpt = isGptImageModel(model);

  if (size === undefined || size === null || size === '') {
    if (model === 'dall-e-2') size = '1024x1024';
    else size = 'auto';
  }

  if (size === 'auto' && isGpt) {
    size = '1024x1024';
  }

  const validEditSizesDalle2 = new Set(['256x256', '512x512', '1024x1024']);
  const validEditSizesGpt = new Set(['256x256', '512x512', '1024x1024', '1536x1024', '1024x1536', 'auto']);

  if (model === 'dall-e-2' && !validEditSizesDalle2.has(size)) {
    const valid = [...validEditSizesDalle2].sort().join(', ');
    res.status(400).json({
      error: { message: `Invalid size '${size}' for dall-e-2 edits. Must be one of: ${valid}.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (isGpt && !validEditSizesGpt.has(size) && size !== 'auto') {
    const valid = [...validEditSizesGpt].filter(s => s !== 'auto').sort().join(', ');
    res.status(400).json({
      error: { message: `Invalid size '${size}' for ${model} edits. Must be one of: ${valid}.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (isNaN(n) || n < 1 || n > 10) {
    res.status(400).json({
      error: { message: "'n' must be between 1 and 10.", type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (isGpt) {
    responseFormat = 'b64_json';
  } else if (!VALID_RESPONSE_FORMATS.has(responseFormat)) {
    res.status(400).json({
      error: { message: `Invalid response_format '${responseFormat}'. Must be 'url' or 'b64_json'.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  // -- Build response --
  const imageData = buildImageData(n, size, responseFormat, model, prompt, req.headers.host ?? 'localhost:18150');
  const result: Record<string, unknown> = {
    created: Math.floor(Date.now() / 1000),
    data: imageData,
  };

  if (isGpt) {
    result.background = background ?? 'auto';
    result.output_format = outputFormat ?? 'png';
    result.quality = mapQualityForResponse(quality);
    result.size = size;
    result.usage = buildEditUsage(n, prompt, hasImage);
  }

  res.status(200).json(result);
});

// ── POST /v1/images/variations ──────────────────────────────────────

app.post('/v1/images/variations', async (req: Request, res: Response) => {
  let trigger = req.query.mock_error as string | undefined;
  if (trigger) {
    log.log('/v1/images/variations', 'POST', { mock_error: trigger });
    if (handleOpenAIError(res, trigger)) return;
  }

  if (!requireBearerAuth(req, res, 'openai')) return;

  let parsed: MultipartResult;
  try {
    parsed = await parseMultipart(req);
  } catch {
    res.status(400).json({
      error: { message: 'Failed to parse multipart form data.', type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  const { fields, fileFields } = parsed;
  const model = fields.model ?? 'dall-e-2';
  const n = parseInt(fields.n ?? '1');
  const size = fields.size ?? '1024x1024';
  const responseFormat = fields.response_format ?? 'url';
  const hasImage = fileFields.has('image');

  const params: Record<string, unknown> = {
    model, n, size, response_format: responseFormat, has_image: hasImage,
  };
  log.log('/v1/images/variations', 'POST', params, model);
  logConsole('POST', '/v1/images/variations', params);

  if (!trigger) {
    trigger = extractErrorTrigger(null, req.query.mock_error as string) ?? undefined;
  }
  if (trigger) {
    if (handleOpenAIError(res, trigger)) return;
  }

  // -- Validation --
  if (!hasImage) {
    res.status(400).json({
      error: { message: "'image' is required.", type: 'invalid_request_error', code: 'missing_required_parameter' },
    });
    return;
  }

  if (model !== 'dall-e-2') {
    res.status(400).json({
      error: { message: 'Image variations are only supported for dall-e-2.', type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (!VALID_SIZES_DALLE2.has(size)) {
    const valid = [...VALID_SIZES_DALLE2].sort().join(', ');
    res.status(400).json({
      error: { message: `Invalid size '${size}' for dall-e-2. Must be one of: ${valid}.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (isNaN(n) || n < 1 || n > 10) {
    res.status(400).json({
      error: { message: "'n' must be between 1 and 10.", type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (!VALID_RESPONSE_FORMATS.has(responseFormat)) {
    res.status(400).json({
      error: { message: `Invalid response_format '${responseFormat}'. Must be 'url' or 'b64_json'.`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  // -- Build response --
  const imageData = buildImageData(n, size, responseFormat, 'dall-e-2', 'variation', req.headers.host ?? 'localhost:18150');
  const result: Record<string, unknown> = {
    created: Math.floor(Date.now() / 1000),
    data: imageData,
  };

  res.status(200).json(result);
});

// ── GET /v1/models ──────────────────────────────────────────────────

app.get('/v1/models', (_req: Request, res: Response) => {
  const now = Math.floor(Date.now() / 1000);
  log.log('/v1/models', 'GET', {});

  res.json({
    object: 'list',
    data: [...VALID_MODELS].sort().map(m => ({
      id: m,
      object: 'model',
      created: now,
      owned_by: 'openai-mock',
    })),
  });
});

// ── Start ───────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('='.repeat(64));
  console.log('  Mock OpenAI Image Generation API');
  console.log('  (DALL-E 2 / DALL-E 3 / gpt-image-1)');
  console.log(`  Base URL: http://localhost:${PORT}/v1`);
  console.log('');
  console.log('  Endpoints:');
  console.log('    POST /v1/images/generations      -- text-to-image');
  console.log('    POST /v1/images/edits            -- image editing (inpainting)');
  console.log('    POST /v1/images/variations       -- image variations (DALL-E 2)');
  console.log('    GET  /v1/models                  -- list available models');
  console.log('    GET  /v1/mock_cdn/<id>.png       -- serve generated images');
  console.log('');
  console.log('  Testing:');
  console.log('    GET    /_mock/log                -- inspect received params');
  console.log('    DELETE /_mock/log                -- clear log');
  console.log('    ?mock_error=<code>               -- trigger error response');
  console.log('    MOCK_ERROR:<code> in prompt      -- trigger error response');
  console.log('');
  console.log('  Models: ' + [...VALID_MODELS].sort().join(', '));
  console.log('='.repeat(64));
});
