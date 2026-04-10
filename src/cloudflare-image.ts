/**
 * Mock Cloudflare Workers AI Image Generation API Server.
 *
 * Drop-in replacement for testing against the real Cloudflare API.
 * Point your Cloudflare client at http://localhost:8300 instead of https://api.cloudflare.com
 *
 *   npx tsx src/cloudflare-image.ts
 *
 * Then in your config (cloudflare-image-generation):
 *   apiToken: "mock-token"
 *   accountId: "mock-account"
 *   apiBaseUrl: "http://localhost:8300/client/v4"
 *
 * Endpoints:
 *   POST /client/v4/accounts/:accountId/ai/run/:modelId
 *     - JSON body for flux-1-schnell (and other JSON models)
 *     - multipart/form-data for flux-2-dev/klein
 *
 * Testing:
 *   GET    /_mock/log              — inspect received params
 *   DELETE /_mock/log              — clear log
 *   ?mock_error=<code>             — trigger error response
 *   MOCK_ERROR:<code> in prompt    — trigger error response
 *   fail=<code> in prompt          — legacy trigger (still works)
 *   Codes: auth, bad_model, no_capacity, quota, timeout,
 *          network, garbage, success_false, no_image,
 *          server_error, forbidden
 */

import express, { type Request, type Response, type NextFunction } from 'express';
import {
  RequestLog, mockLogRouter, extractErrorTriggerCf, handleCloudflareError,
  requireBearerAuth, cfError, logConsole, generatePng,
} from './shared.js';

const PORT = 18300;

const VALID_MODELS = new Set([
  '@cf/black-forest-labs/flux-1-schnell',
  '@cf/black-forest-labs/flux-2-dev',
  '@cf/black-forest-labs/flux-2-klein-4b',
  '@cf/black-forest-labs/flux-2-klein-9b',
  '@cf/leonardo/lucid-origin',
  '@cf/leonardo/phoenix-1.0',
]);

const MULTIPART_MODELS = new Set([
  '@cf/black-forest-labs/flux-2-dev',
  '@cf/black-forest-labs/flux-2-klein-4b',
  '@cf/black-forest-labs/flux-2-klein-9b',
]);

// ── Request body parsing ────────────────────────────────────────────

function parseRequestBody(req: Request, modelId: string): Record<string, unknown> {
  const contentType = req.headers['content-type'] ?? '';

  if (MULTIPART_MODELS.has(modelId) && contentType.includes('multipart/form-data')) {
    // For multipart, fields come through req.body from urlencoded or raw parsing.
    // Express doesn't natively parse multipart, but the fields we need (prompt,
    // width, height, steps, seed, guidance) may arrive as urlencoded fallback
    // or we parse from the raw body if available.
    const params: Record<string, unknown> = { ...req.body };
    for (const k of ['width', 'height', 'steps', 'seed']) {
      if (k in params && typeof params[k] === 'string') {
        const n = parseInt(params[k] as string, 10);
        if (!isNaN(n)) params[k] = n;
      }
    }
    for (const k of ['guidance']) {
      if (k in params && typeof params[k] === 'string') {
        const n = parseFloat(params[k] as string);
        if (!isNaN(n)) params[k] = n;
      }
    }
    return params;
  }

  return req.body ?? {};
}

// ── Multipart field extraction from raw body ────────────────────────

function extractMultipartFields(buf: Buffer, boundary: string): Record<string, string> {
  const fields: Record<string, string> = {};
  const sep = `--${boundary}`;
  const parts = buf.toString('latin1').split(sep);
  for (const part of parts) {
    if (part === '--\r\n' || part === '--' || part.trim() === '') continue;
    const headerEnd = part.indexOf('\r\n\r\n');
    if (headerEnd === -1) continue;
    const headers = part.slice(0, headerEnd);
    const body = part.slice(headerEnd + 4).replace(/\r\n$/, '');
    const nameMatch = headers.match(/name="([^"]+)"/);
    if (!nameMatch) continue;
    const filenameMatch = headers.match(/filename="/);
    if (filenameMatch) continue; // skip file fields
    fields[nameMatch[1]] = body;
  }
  return fields;
}

// ── Express app ─────────────────────────────────────────────────────

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Capture raw body for multipart parsing
app.use((req: Request, _res: Response, next: NextFunction) => {
  const contentType = req.headers['content-type'] ?? '';
  if (contentType.includes('multipart/form-data')) {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      const rawBody = Buffer.concat(chunks);
      const boundaryMatch = contentType.match(/boundary=(.+)/);
      if (boundaryMatch) {
        const fields = extractMultipartFields(rawBody, boundaryMatch[1]);
        req.body = fields;
      }
      next();
    });
  } else {
    next();
  }
});

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
    params = parseRequestBody(req, modelId);
  } catch (e) {
    cfError(res, 400, 3003, `Failed to parse request body: ${e}`);
    return;
  }

  const prompt = (params.prompt as string) ?? '';

  let trigger = queryTrigger ?? null;
  if (!trigger) {
    trigger = extractErrorTriggerCf(prompt);
  }

  log.log(`ai/run/${modelId}`, 'POST', params, modelId);
  logConsole('POST', `/client/v4/accounts/${accountId}/ai/run/${modelId}`, {
    ...params,
    error_trigger: trigger ?? '(none)',
  });

  if (trigger) {
    if (handleCloudflareError(res, trigger, modelId)) return;
  }

  if (!VALID_MODELS.has(modelId)) {
    cfError(res, 400, 5007, `No such model ${modelId} or task`);
    return;
  }

  if (!prompt) {
    cfError(res, 400, 5004, 'Missing required parameter: prompt');
    return;
  }

  const width = parseInt(String(params.width ?? '1024'), 10);
  const height = parseInt(String(params.height ?? '1024'), 10);

  const png = generatePng(Math.min(width, 64), Math.min(height, 64));
  const b64 = png.toString('base64');

  res.json({
    result: { image: b64 },
    success: true,
    errors: [],
    messages: [],
  });
});

// ── Health / smoke endpoint ─────────────────────────────────────────

app.get('/', (_req: Request, res: Response): void => {
  res.json({
    name: 'mock_cloudflare_image_server',
    valid_models: [...VALID_MODELS].sort(),
    error_triggers: [
      'auth', 'bad_model', 'no_capacity', 'quota', 'timeout',
      'network', 'garbage', 'success_false', 'no_image',
      'server_error', 'forbidden',
    ],
  });
});

// ── Start server ────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('  Mock Cloudflare Workers AI Image Generation API');
  console.log(`  Base URL: http://localhost:${PORT}`);
  console.log('  Endpoint: POST /client/v4/accounts/{id}/ai/run/{model}');
  console.log('');
  console.log('  Models: ' + [...VALID_MODELS].sort().join(', '));
  console.log('');
  console.log('  Testing:');
  console.log('    GET    /_mock/log              — inspect received params');
  console.log('    DELETE /_mock/log              — clear log');
  console.log('    ?mock_error=<code>             — trigger error response');
  console.log('    MOCK_ERROR:<code> in prompt    — trigger error response');
  console.log('    fail=<code> in prompt          — legacy trigger (still works)');
  console.log('    Codes: auth, bad_model, no_capacity, quota, timeout,');
  console.log('           network, garbage, success_false, no_image,');
  console.log('           server_error, forbidden');
  console.log('='.repeat(60));
});
