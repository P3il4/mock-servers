/**
 * Mock xAI Speech-to-Text API Server (port 18700).
 *
 *   POST /v1/stt     multipart/form-data → JSON transcript
 *
 *   npx tsx src/xai-stt.ts
 *
 * Fields accepted (multipart):
 *   file                  audio bytes (required)
 *   language              BCP-47 or 'auto' (optional)
 *   response_format       'json' (default) | 'text' (optional)
 *   timestamp_granularities  'word' | 'segment' | JSON array (optional)
 *
 * Monitoring: GET/DELETE /_mock/log
 * Error injection:
 *   - ?mock_error=<code>  rejects the request with an OpenAI-style error
 */

import express, { type Request, type Response } from 'express';
import {
  RequestLog, mockLogRouter, handleOpenAIError,
  requireBearerAuth, logConsole, corsMiddleware,
} from './shared.js';

const PORT = 18700;
const MAX_AUDIO_BYTES = 100 * 1024 * 1024; // 100 MB
const VALID_RESPONSE_FORMATS = new Set(['json', 'text']);

const SAMPLE_TRANSCRIPT_TEXT = 'Hello, this is a mock transcription from the xAI STT server.';

// ── Minimal multipart parser (matches openai-image.ts) ──────────────

interface MultipartResult {
  fields: Record<string, string>;
  files: Record<string, { name: string; data: Buffer; contentType: string | null }>;
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
    let totalSize = 0;
    req.on('data', (chunk: Buffer) => {
      totalSize += chunk.length;
      if (totalSize > MAX_AUDIO_BYTES) {
        reject(new Error('Audio exceeds maximum allowed size'));
        return;
      }
      chunks.push(chunk);
    });
    req.on('error', reject);
    req.on('end', () => {
      const body = Buffer.concat(chunks);
      const fields: Record<string, string> = {};
      const files: Record<string, { name: string; data: Buffer; contentType: string | null }> = {};

      const boundaryBuf = Buffer.from(`--${boundary}`);
      const parts: Buffer[] = [];
      let start = 0;

      while (true) {
        const idx = body.indexOf(boundaryBuf, start);
        if (idx === -1) break;
        if (start > 0) {
          let partStart = start;
          let partEnd = idx;
          if (body[partStart] === 0x0d && body[partStart + 1] === 0x0a) partStart += 2;
          if (partEnd >= 2 && body[partEnd - 2] === 0x0d && body[partEnd - 1] === 0x0a) partEnd -= 2;
          if (partEnd > partStart) parts.push(body.subarray(partStart, partEnd));
        }
        start = idx + boundaryBuf.length;
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

        const filenameMatch = headerStr.match(/filename="([^"]*)"/);
        if (filenameMatch) {
          const contentTypeMatch = headerStr.match(/Content-Type:\s*([^\r\n]+)/i);
          files[name] = {
            name: filenameMatch[1] ?? 'file',
            data: valueBytes,
            contentType: contentTypeMatch ? contentTypeMatch[1]!.trim() : null,
          };
        } else {
          fields[name] = valueBytes.toString('utf-8');
        }
      }

      resolve({ fields, files });
    });
  });
}

// ── App ──────────────────────────────────────────────────────────────

const app = express();
app.use(corsMiddleware);
app.use((req, _res, next) => {
  const ct = req.headers['content-type'] ?? '';
  if (ct.includes('multipart/form-data')) {
    next();
  } else {
    express.json({ limit: '1mb' })(req, _res, next);
  }
});

const log = new RequestLog();
app.use(mockLogRouter(log));

// ── POST /v1/stt ────────────────────────────────────────────────────

app.post('/v1/stt', async (req: Request, res: Response) => {
  const trigger = req.query.mock_error as string | undefined;
  if (trigger) {
    log.log('/v1/stt', 'POST', { mock_error: trigger });
    if (handleOpenAIError(res, trigger)) return;
  }

  if (!requireBearerAuth(req, res, 'openai')) return;

  let parsed: MultipartResult;
  try {
    parsed = await parseMultipart(req);
  } catch (e) {
    const msg = e instanceof Error ? e.message : 'Failed to parse multipart form data.';
    res.status(400).json({
      error: { message: msg, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  const { fields, files } = parsed;
  const language = fields.language;
  const response_format = (fields.response_format ?? 'json').toLowerCase();
  const timestamp_granularities = fields.timestamp_granularities;
  const filePart = files.file;

  const params: Record<string, unknown> = {
    language: language ?? null,
    response_format,
    timestamp_granularities: timestamp_granularities ?? null,
    file_size: filePart?.data.length ?? 0,
    file_content_type: filePart?.contentType ?? null,
    file_name: filePart?.name ?? null,
  };
  log.log('/v1/stt', 'POST', params);
  logConsole('POST', '/v1/stt', params);

  if (!filePart || filePart.data.length === 0) {
    res.status(400).json({
      error: { message: "'file' is required (multipart field).", type: 'invalid_request_error', code: 'missing_required_parameter' },
    });
    return;
  }
  if (!VALID_RESPONSE_FORMATS.has(response_format)) {
    res.status(400).json({
      error: {
        message: `Invalid response_format '${response_format}'. Valid: ${[...VALID_RESPONSE_FORMATS].sort().join(', ')}.`,
        type: 'invalid_request_error',
        code: 'invalid_parameter',
      },
    });
    return;
  }

  // Fake duration: roughly 1 second per 16 KB of audio, minimum 1 second.
  const durationSeconds = Math.max(1, Math.ceil(filePart.data.length / 16_000));
  const transcriptText = SAMPLE_TRANSCRIPT_TEXT;

  if (response_format === 'text') {
    res.type('text/plain').send(transcriptText);
    return;
  }

  const words = transcriptText.split(/\s+/).filter(Boolean);
  const perWord = durationSeconds / Math.max(words.length, 1);
  const wordTimestamps = words.map((w, i) => ({
    start: Number((i * perWord).toFixed(2)),
    end: Number(((i + 1) * perWord).toFixed(2)),
    text: w,
  }));

  res.status(200).json({
    text: transcriptText,
    language: language ?? 'en',
    duration_seconds: durationSeconds,
    words: wordTimestamps,
    segments: [{
      start: 0,
      end: durationSeconds,
      text: transcriptText,
    }],
  });
});

// ── Start ───────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('='.repeat(64));
  console.log('  Mock xAI Speech-to-Text API (Grok STT)');
  console.log(`  Base URL: http://localhost:${PORT}/v1`);
  console.log('');
  console.log('  Endpoints:');
  console.log('    POST /v1/stt                     -- transcribe (multipart/form-data)');
  console.log('');
  console.log('  Testing:');
  console.log('    GET    /_mock/log                -- inspect received params');
  console.log('    DELETE /_mock/log                -- clear log');
  console.log('    ?mock_error=<code>               -- reject request with error');
  console.log('');
  console.log('  Response formats: ' + [...VALID_RESPONSE_FORMATS].sort().join(', '));
  console.log('='.repeat(64));
});
