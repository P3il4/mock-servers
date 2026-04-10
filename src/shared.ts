/**
 * Shared infrastructure for all mock API servers.
 *
 * Provides: request logging, /_mock/log router, error injection,
 * auth checks, console logging, and media generation (PNG / MP4).
 */

import { type Request, type Response, Router } from 'express';
import zlib from 'node:zlib';

// ── Request Logging ──────────────────────────────────────────────────

export interface LogEntry {
  timestamp: number;
  endpoint: string;
  method: string;
  model: string | null;
  params: unknown;
}

export class RequestLog {
  private entries: LogEntry[] = [];
  private readonly max: number;

  constructor(max = 200) {
    this.max = max;
  }

  log(endpoint: string, method: string, params: Record<string, unknown>, model?: string): void {
    this.entries.push({
      timestamp: Date.now() / 1000,
      endpoint,
      method,
      model: model ?? null,
      params: sanitize(params),
    });
    if (this.entries.length > this.max) this.entries.shift();
  }

  query(filters?: { endpoint?: string; model?: string; last?: number }): LogEntry[] {
    let r = [...this.entries];
    if (filters?.endpoint) {
      const ep = filters.endpoint;
      r = r.filter(e => e.endpoint.includes(ep));
    }
    if (filters?.model) {
      const m = filters.model;
      r = r.filter(e => (e.model ?? '').includes(m));
    }
    if (filters?.last) r = r.slice(-filters.last);
    return r;
  }

  clear(): void {
    this.entries = [];
  }
}

// ── Sanitization ─────────────────────────────────────────────────────

export function sanitize(obj: unknown): unknown {
  if (obj === null || obj === undefined) return obj;
  if (typeof obj === 'string') return obj.length > 200 ? `[${obj.length} chars truncated]` : obj;
  if (Array.isArray(obj)) return obj.map(sanitize);
  if (typeof obj === 'object') {
    return Object.fromEntries(
      Object.entries(obj as Record<string, unknown>).map(([k, v]) => [k, sanitize(v)]),
    );
  }
  return obj;
}

// ── Mock Log Router ──────────────────────────────────────────────────

export function mockLogRouter(log: RequestLog): Router {
  const router = Router();

  router.get('/_mock/log', (req: Request, res: Response) => {
    res.json(
      log.query({
        endpoint: req.query.endpoint as string | undefined,
        model: req.query.model as string | undefined,
        last: req.query.last ? parseInt(req.query.last as string) : undefined,
      }),
    );
  });

  router.delete('/_mock/log', (_req: Request, res: Response) => {
    log.clear();
    res.json({ cleared: true });
  });

  return router;
}

// ── Error Trigger Extraction ─────────────────────────────────────────

export function extractErrorTrigger(
  promptText: string | null | undefined,
  queryMockError?: string,
): string | null {
  if (queryMockError) return queryMockError;
  if (!promptText) return null;
  const m = promptText.match(/MOCK_ERROR:(\w+)/);
  return m ? m[1] : null;
}

/** Also checks legacy Cloudflare `fail=<code>` in prompt text. */
export function extractErrorTriggerCf(
  promptText: string | null | undefined,
  queryMockError?: string,
): string | null {
  const t = extractErrorTrigger(promptText, queryMockError);
  if (t) return t;
  if (!promptText) return null;
  const lower = promptText.toLowerCase();
  for (const mode of [
    'auth', 'bad_model', 'no_capacity', 'quota', 'timeout',
    'network', 'garbage', 'success_false', 'no_image',
    'server_error', 'forbidden',
  ]) {
    if (lower.includes(`fail=${mode}`)) return mode;
  }
  return null;
}

// ── Google API Errors ────────────────────────────────────────────────

export const GOOGLE_ERRORS: Record<string, [number, string, string]> = {
  auth:          [401, 'UNAUTHENTICATED',   'Request had invalid authentication credentials.'],
  not_found:     [404, 'NOT_FOUND',          'The requested resource was not found.'],
  rate_limit:    [429, 'RESOURCE_EXHAUSTED', 'Resource has been exhausted (e.g. check quota).'],
  quota:         [429, 'RESOURCE_EXHAUSTED', 'Quota exceeded for project.'],
  invalid_model: [400, 'INVALID_ARGUMENT',   'Model not found or not supported.'],
  server_error:  [500, 'INTERNAL',           'An internal error has occurred.'],
  timeout:       [504, 'DEADLINE_EXCEEDED',  'The request deadline was exceeded.'],
  permission:    [403, 'PERMISSION_DENIED',  'Permission denied on the resource.'],
  unavailable:   [503, 'UNAVAILABLE',        'The service is currently unavailable.'],
  invalid_arg:   [400, 'INVALID_ARGUMENT',   'Invalid argument provided.'],
};

export function handleGoogleError(res: Response, trigger: string): boolean {
  if (trigger === 'safety') {
    res.json({
      candidates: [{
        finishReason: 'SAFETY',
        safetyRatings: [
          { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', probability: 'HIGH', blocked: true },
          { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', probability: 'NEGLIGIBLE' },
          { category: 'HARM_CATEGORY_HARASSMENT', probability: 'NEGLIGIBLE' },
          { category: 'HARM_CATEGORY_HATE_SPEECH', probability: 'NEGLIGIBLE' },
        ],
      }],
      usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 0, totalTokenCount: 10 },
    });
    return true;
  }
  if (trigger === 'content_filter') {
    res.json({
      promptFeedback: {
        blockReason: 'SAFETY',
        safetyRatings: [{ category: 'HARM_CATEGORY_DANGEROUS_CONTENT', probability: 'HIGH', blocked: true }],
      },
    });
    return true;
  }
  if (trigger === 'recitation') {
    res.json({
      candidates: [{
        finishReason: 'RECITATION',
        citationMetadata: {
          citationSources: [{ startIndex: 0, endIndex: 100, uri: 'https://example.com/source' }],
        },
      }],
      usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 0, totalTokenCount: 10 },
    });
    return true;
  }
  const err = GOOGLE_ERRORS[trigger];
  if (err) {
    res.status(err[0]).json({ error: { code: err[0], message: err[2], status: err[1] } });
    return true;
  }
  return false;
}

// ── OpenAI API Errors ────────────────────────────────────────────────

export const OPENAI_ERRORS: Record<string, [number, string, string, string]> = {
  auth:           [401, 'invalid_request_error', 'invalid_api_key', 'Incorrect API key provided.'],
  rate_limit:     [429, 'rate_limit_error', 'rate_limit_exceeded', 'Rate limit reached. Please retry after a brief wait.'],
  quota:          [429, 'insufficient_quota', 'insufficient_quota', 'You exceeded your current usage quota.'],
  server_error:   [500, 'server_error', 'server_error', 'The server had an error while processing your request.'],
  content_policy: [400, 'invalid_request_error', 'content_policy_violation', 'Your request was rejected as a result of our safety system.'],
  billing:        [402, 'billing_error', 'billing_not_active', 'Your billing is not active.'],
  timeout:        [408, 'server_error', 'timeout', 'Request timed out.'],
  invalid_model:  [400, 'invalid_request_error', 'model_not_found', 'The model does not exist or you do not have access to it.'],
  overloaded:     [503, 'server_error', 'overloaded', 'The engine is currently overloaded. Please try again later.'],
};

export function handleOpenAIError(res: Response, trigger: string): boolean {
  const err = OPENAI_ERRORS[trigger];
  if (!err) return false;
  res.status(err[0]).json({ error: { message: err[3], type: err[1], code: err[2] } });
  return true;
}

// ── Cloudflare API Errors ────────────────────────────────────────────

export function cfError(res: Response, httpStatus: number, code: number, message: string): void {
  res.status(httpStatus).json({ result: null, success: false, errors: [{ code, message }], messages: [] });
}

export function handleCloudflareError(res: Response, trigger: string, modelId = ''): boolean {
  const map: Record<string, () => void> = {
    auth:          () => cfError(res, 401, 10000, 'Authentication error: invalid token'),
    bad_model:     () => cfError(res, 400, 5007, `No such model ${modelId} or task`),
    no_capacity:   () => cfError(res, 429, 3040, 'No available capacity for this model right now'),
    quota:         () => cfError(res, 429, 3036, 'You have exhausted your daily free allocation of 10,000 neurons'),
    timeout:       () => cfError(res, 408, 3007, 'Request timed out while processing'),
    network:       () => res.status(500).end(),
    garbage:       () => res.type('html').send('<html><body>oops not json</body></html>'),
    success_false: () => res.json({ result: null, success: false, errors: [{ code: 7003, message: 'Internal error during inference' }], messages: [] }),
    no_image:      () => res.json({ result: { foo: 'bar' }, success: true, errors: [], messages: [] }),
    server_error:  () => cfError(res, 500, 1000, 'An internal server error occurred'),
    forbidden:     () => cfError(res, 403, 10001, 'Account access restricted'),
  };
  const handler = map[trigger];
  if (!handler) return false;
  handler();
  return true;
}

// ── Auth Checks ──────────────────────────────────────────────────────

export function requireBearerAuth(req: Request, res: Response, style: 'openai' | 'cloudflare'): boolean {
  const auth = req.headers.authorization ?? '';
  if (!auth.startsWith('Bearer ') || !auth.slice(7).trim()) {
    if (style === 'openai') {
      res.status(401).json({ error: { message: 'Missing API key', type: 'invalid_request_error', code: 'missing_api_key' } });
    } else {
      cfError(res, 401, 10000, 'Authentication error: missing bearer token');
    }
    return false;
  }
  return true;
}

// ── Console Logging ──────────────────────────────────────────────────

export function logConsole(method: string, path: string, params?: Record<string, unknown>): void {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`  ${method} ${path}`);
  if (params && Object.keys(params).length > 0) {
    console.log(`  params: ${JSON.stringify(params, null, 2)}`);
  }
  console.log('='.repeat(60));
}

// ── PNG Generation ───────────────────────────────────────────────────

function crc32(buf: Buffer): number {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) {
    crc ^= buf[i]!;
    for (let j = 0; j < 8; j++) crc = (crc >>> 1) ^ (crc & 1 ? 0xEDB88320 : 0);
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

function pngChunk(type: string, data: Buffer): Buffer {
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}

export function generatePng(w: number, h: number, r = 51, g = 153, b = 255): Buffer {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 2; // 8-bit RGB

  const pixel = Buffer.from([r, g, b]);
  const rows: Buffer[] = [];
  for (let y = 0; y < h; y++) {
    rows.push(Buffer.from([0])); // filter byte
    for (let x = 0; x < w; x++) rows.push(pixel);
  }

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', zlib.deflateSync(Buffer.concat(rows))),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
}

export const MOCK_PNG_B64 = generatePng(1, 1).toString('base64');

// ── MP4 Generation ───────────────────────────────────────────────────

function mp4Box(type: string, data: Buffer): Buffer {
  const h = Buffer.alloc(8);
  h.writeUInt32BE(data.length + 8, 0);
  h.write(type, 4, 4, 'ascii');
  return Buffer.concat([h, data]);
}

function u32(n: number): Buffer { const b = Buffer.alloc(4); b.writeUInt32BE(n); return b; }
function u16(n: number): Buffer { const b = Buffer.alloc(2); b.writeUInt16BE(n); return b; }
function zeros(n: number): Buffer { return Buffer.alloc(n); }

export function generateMp4(width: number, height: number, durationS: number): Buffer {
  const now = Math.floor(Date.now() / 1000);
  const dur = durationS * 1000;

  const ftypPayload = Buffer.concat([
    Buffer.from('ftypisom'), u32(0x200), Buffer.from('isomiso2mp41'),
  ]);
  const ftyp = Buffer.concat([u32(ftypPayload.length + 4), ftypPayload]);
  const mdat = mp4Box('mdat', Buffer.alloc(1024));

  const mvhd = mp4Box('mvhd', Buffer.concat([
    zeros(4), u32(now), u32(now), u32(1000), u32(dur),
    u16(1), u16(0), // rate 1.0
    Buffer.from([1, 0]), zeros(10), // volume + reserved
    u32(0x00010000), zeros(12), u32(0x00010000), zeros(12), u32(0x40000000),
    zeros(24), u32(2),
  ]));

  const tkhd = mp4Box('tkhd', Buffer.concat([
    u32(3), zeros(8), u32(1), zeros(4), u32(dur),
    zeros(8), u16(0), u16(0),
    u32(0x00010000), zeros(12), u32(0x00010000), zeros(12), u32(0x40000000),
    u16(width), u16(height), zeros(4),
  ]));

  const mdhd = mp4Box('mdhd', Buffer.concat([zeros(4), zeros(8), u32(1000), u32(dur), u16(0x55C4), u16(0)]));
  const hdlr = mp4Box('hdlr', Buffer.concat([zeros(8), Buffer.from('vide'), zeros(12), Buffer.from('VideoHandler\0')]));
  const stbl = mp4Box('stbl', Buffer.concat([
    mp4Box('stsd', Buffer.concat([zeros(4), u32(0)])),
    mp4Box('stts', Buffer.concat([zeros(4), u32(0)])),
    mp4Box('stsc', Buffer.concat([zeros(4), u32(0)])),
    mp4Box('stsz', Buffer.concat([zeros(4), u32(0), u32(0)])),
    mp4Box('stco', Buffer.concat([zeros(4), u32(0)])),
  ]));
  const dinf = mp4Box('dinf', mp4Box('dref', Buffer.concat([zeros(4), u32(1), mp4Box('url ', u32(1))])));
  const minf = mp4Box('minf', Buffer.concat([mp4Box('vmhd', Buffer.concat([zeros(4), zeros(8)])), dinf, stbl]));
  const mdia = mp4Box('mdia', Buffer.concat([mdhd, hdlr, minf]));
  const trak = mp4Box('trak', Buffer.concat([tkhd, mdia]));
  const moov = mp4Box('moov', Buffer.concat([mvhd, trak]));

  return Buffer.concat([ftyp, mdat, moov]);
}

export const MOCK_MP4_B64 = generateMp4(320, 240, 4).toString('base64');
