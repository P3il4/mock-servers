/**
 * Mock xAI Text-to-Speech API Server (port 18650).
 *
 * Implements the REST TTS surface:
 *   POST /v1/tts               JSON body → binary audio
 *   GET  /v1/tts/voices        List voices
 *
 *   npx tsx src/xai-tts.ts
 *
 * The mock always returns a silent PCM WAV payload (length roughly scaled to
 * text length). Content-Type is set per requested codec; the bytes themselves
 * are the same silent WAV regardless — clients generally don't parse them.
 *
 * Monitoring: GET/DELETE /_mock/log
 * Error injection:
 *   - MOCK_ERROR:<code> in `text`, or ?mock_error=<code>
 */

import express, { type Request, type Response } from 'express';
import {
  RequestLog, mockLogRouter, extractErrorTrigger, handleOpenAIError,
  requireBearerAuth, logConsole, corsMiddleware,
} from './shared.js';

const PORT = 18650;
const MAX_UNARY_CHARS = 15_000;
const DEFAULT_SAMPLE_RATE = 24_000;
const DEFAULT_CODEC = 'mp3';

const VALID_VOICES = [
  { voice_id: 'ara', name: 'Ara', language: 'multilingual' },
  { voice_id: 'eve', name: 'Eve', language: 'multilingual' },
  { voice_id: 'leo', name: 'Leo', language: 'multilingual' },
  { voice_id: 'rex', name: 'Rex', language: 'multilingual' },
  { voice_id: 'sal', name: 'Sal', language: 'multilingual' },
  { voice_id: 'una', name: 'Una', language: 'multilingual' },
];
const VALID_VOICE_IDS = new Set(VALID_VOICES.map(v => v.voice_id));

const VALID_CODECS = new Set(['mp3', 'wav', 'pcm', 'ulaw', 'alaw']);
const CODEC_CONTENT_TYPES: Record<string, string> = {
  mp3: 'audio/mpeg',
  wav: 'audio/wav',
  pcm: 'audio/pcm',
  ulaw: 'audio/basic',
  alaw: 'audio/basic',
};

// ── Silent WAV generator ─────────────────────────────────────────────

function silentWav(durationMs: number, sampleRate = DEFAULT_SAMPLE_RATE): Buffer {
  const numSamples = Math.floor((sampleRate * durationMs) / 1000);
  const dataLen = numSamples * 2; // 16-bit mono
  const buf = Buffer.alloc(44 + dataLen);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + dataLen, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);          // fmt chunk size
  buf.writeUInt16LE(1, 20);           // PCM format
  buf.writeUInt16LE(1, 22);           // 1 channel (mono)
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28); // byte rate
  buf.writeUInt16LE(2, 32);           // block align
  buf.writeUInt16LE(16, 34);          // bits per sample
  buf.write('data', 36);
  buf.writeUInt32LE(dataLen, 40);
  // samples are already zero (silence)
  return buf;
}

// ── App ──────────────────────────────────────────────────────────────

const app = express();
app.use(corsMiddleware);
app.use(express.json({ limit: '10mb' }));

const log = new RequestLog();
app.use(mockLogRouter(log));

// ── GET /v1/tts/voices ───────────────────────────────────────────────

app.get('/v1/tts/voices', (_req: Request, res: Response) => {
  log.log('/v1/tts/voices', 'GET', {});
  res.json({ voices: VALID_VOICES });
});

// ── POST /v1/tts ────────────────────────────────────────────────────

app.post('/v1/tts', (req: Request, res: Response) => {
  const data = (req.body ?? {}) as Record<string, unknown>;
  const text = typeof data.text === 'string' ? data.text : '';
  const voice_id = typeof data.voice_id === 'string' ? data.voice_id.toLowerCase() : 'eve';
  const language = typeof data.language === 'string' ? data.language : 'auto';
  const outputFormat = (data.output_format ?? {}) as Record<string, unknown>;
  const codec = typeof outputFormat.codec === 'string'
    ? outputFormat.codec.toLowerCase()
    : DEFAULT_CODEC;

  const params: Record<string, unknown> = {
    text_length: text.length,
    voice_id,
    language,
    codec,
  };
  log.log('/v1/tts', 'POST', params);
  logConsole('POST', '/v1/tts', params);

  if (!requireBearerAuth(req, res, 'openai')) return;

  const trigger = (req.query.mock_error as string | undefined)
    ?? extractErrorTrigger(text, undefined) ?? undefined;
  if (trigger) {
    if (handleOpenAIError(res, trigger)) return;
  }

  if (!text) {
    res.status(400).json({
      error: { message: "'text' is required.", type: 'invalid_request_error', code: 'missing_required_parameter' },
    });
    return;
  }
  if (text.length > MAX_UNARY_CHARS) {
    res.status(400).json({
      error: {
        message: `'text' exceeds the ${MAX_UNARY_CHARS}-character unary cap. Use the WebSocket streaming endpoint for longer input.`,
        type: 'invalid_request_error',
        code: 'invalid_parameter',
      },
    });
    return;
  }
  if (!VALID_VOICE_IDS.has(voice_id)) {
    res.status(400).json({
      error: {
        message: `Unknown voice_id '${voice_id}'. Valid: ${[...VALID_VOICE_IDS].sort().join(', ')}.`,
        type: 'invalid_request_error',
        code: 'invalid_parameter',
      },
    });
    return;
  }
  if (!VALID_CODECS.has(codec)) {
    res.status(400).json({
      error: {
        message: `Unknown codec '${codec}'. Valid: ${[...VALID_CODECS].sort().join(', ')}.`,
        type: 'invalid_request_error',
        code: 'invalid_parameter',
      },
    });
    return;
  }

  // Roughly 60ms of audio per character, capped to a sane max.
  const approxDurationMs = Math.min(60_000, Math.max(200, text.length * 60));
  const audio = silentWav(approxDurationMs);

  res.setHeader('Content-Type', CODEC_CONTENT_TYPES[codec] ?? 'audio/mpeg');
  res.setHeader('Content-Length', String(audio.length));
  res.status(200).end(audio);
});

// ── Start ───────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('='.repeat(64));
  console.log('  Mock xAI Text-to-Speech API (Grok TTS)');
  console.log(`  Base URL: http://localhost:${PORT}/v1`);
  console.log('');
  console.log('  Endpoints:');
  console.log('    POST /v1/tts                     -- synthesize (binary audio)');
  console.log('    GET  /v1/tts/voices              -- list voices');
  console.log('');
  console.log('  Testing:');
  console.log('    GET    /_mock/log                -- inspect received params');
  console.log('    DELETE /_mock/log                -- clear log');
  console.log('    ?mock_error=<code>               -- reject request with error');
  console.log('    MOCK_ERROR:<code> in text        -- reject request with error');
  console.log('');
  console.log('  Voices: ' + [...VALID_VOICE_IDS].sort().join(', '));
  console.log('  Codecs: ' + [...VALID_CODECS].sort().join(', '));
  console.log('='.repeat(64));
});
