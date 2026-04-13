/**
 * Mock Together AI Video & Image Generation API Server.
 *
 * Drop-in replacement for testing against the real Together API.
 * Point your Together client at http://localhost:8200 instead of https://api.together.xyz
 *
 *   npx tsx src/together.ts
 *
 * Then in your code:
 *   client = Together(api_key="mock-key", base_url="http://localhost:8200/v1")
 *
 * Endpoints:
 *   POST /v2/videos                — create video job
 *   GET  /v2/videos/:id            — poll job status
 *   POST /v1/videos                — alias for /v2/videos
 *   GET  /v1/videos/:id            — alias for /v2/videos/:id
 *   POST /v1/images/generations    — generate image
 *   GET  /mock-cdn/:id.mp4         — download video
 *   GET  /mock-cdn/img-:id.png     — download image
 *   GET  /v1/models                — list models
 *
 * Testing:
 *   GET    /_mock/log              — inspect received params
 *   DELETE /_mock/log              — clear log
 *   ?mock_error=<code>             — trigger error response
 *   MOCK_ERROR:<code> in prompt    — trigger error response
 *   Codes: auth, rate_limit, quota, server_error, invalid_model,
 *          content_policy, timeout, overloaded,
 *          generation_failed (async — job fails after polling)
 */

import express, { type Request, type Response } from 'express';
import crypto from 'node:crypto';
import {
  RequestLog, mockLogRouter, extractErrorTrigger, handleOpenAIError,
  requireBearerAuth, logConsole, generatePng, generateMp4, corsMiddleware,
} from './shared.js';

const PORT = 18200;
const MOCK_GENERATION_SECONDS = 6;

const VALID_MODELS = new Set([
  'Wan-AI/Wan2.1-T2V-14B',
  'Wan-AI/Wan2.1-I2V-14B-720P',
  'Wan-AI/Wan2.1-T2V-1.3B',
  'Wan-AI/Wan2.2-T2V-A14B',
  'minimax/video-01-director',
  'google/veo-3.0',
  'openai/sora-2',
]);

// ── In-memory stores ────────────────────────────────────────────────

interface VideoRecord {
  id: string;
  object: 'video';
  model: string;
  status: 'queued' | 'in_progress' | 'completed' | 'failed';
  created_at: number;
  completed_at: number | null;
  seconds: string;
  size: string;
  error: { code: string; message: string } | null;
  outputs: { cost: number; video_url: string } | null;
  _width: number;
  _height: number;
  _fps: number;
  _prompt: string;
}

interface ImageRecord {
  _png: Buffer;
}

const videosDb = new Map<string, VideoRecord>();
const imagesDb = new Map<string, ImageRecord>();

// ── Background progress via setTimeout ──────────────────────────────

function progressWorker(videoId: string, totalMs: number, shouldFail: boolean): void {
  const steps = 10;
  const stepMs = totalMs / steps;

  for (let i = 1; i <= steps; i++) {
    setTimeout(() => {
      const vid = videosDb.get(videoId);
      if (!vid) return;
      vid.status = 'in_progress';
    }, stepMs * i);
  }

  setTimeout(() => {
    const vid = videosDb.get(videoId);
    if (!vid) return;
    if (shouldFail) {
      vid.status = 'failed';
      vid.error = {
        code: 'generation_failed',
        message: 'Video generation failed (MOCK_ERROR:generation_failed)',
      };
    } else {
      vid.status = 'completed';
      vid.completed_at = Math.floor(Date.now() / 1000);
      vid.outputs = {
        cost: 0.0,
        video_url: `http://localhost:${PORT}/mock-cdn/${videoId}.mp4`,
      };
    }
  }, totalMs);
}

// ── Express app ─────────────────────────────────────────────────────

const app = express();
app.use(corsMiddleware);
app.use(express.json());

const log = new RequestLog();
app.use(mockLogRouter(log));

// ── POST /v2/videos ─────────────────────────────────────────────────

function createVideo(req: Request, res: Response): void {
  const queryTrigger = req.query.mock_error as string | undefined;
  if (queryTrigger) {
    log.log('/v2/videos', 'POST', { mock_error: queryTrigger });
    logConsole('POST', '/v2/videos', { mock_error: queryTrigger });
    if (handleOpenAIError(res, queryTrigger)) return;
  }

  if (!requireBearerAuth(req, res, 'openai')) return;

  const data = req.body ?? {};

  const model: string = data.model ?? '';
  const prompt: string = data.prompt ?? '';
  const height: number = data.height ?? 480;
  const width: number = data.width ?? 848;
  const seconds: string = String(data.seconds ?? '5');
  const fps: number = data.fps ?? 24;
  const steps: number = data.steps ?? 20;
  const seed: number | undefined = data.seed;
  const guidanceScale: number = data.guidance_scale ?? 8.0;
  const outputFormat: string = data.output_format ?? 'MP4';
  const negativePrompt: string | undefined = data.negative_prompt;

  const params: Record<string, unknown> = {
    model, prompt, height, width, seconds, fps, steps, seed,
    guidance_scale: guidanceScale, output_format: outputFormat,
    negative_prompt: negativePrompt,
  };

  if (!queryTrigger) {
    log.log('/v2/videos', 'POST', params, model);
    logConsole('POST', '/v2/videos', params);
  }

  let trigger = queryTrigger ?? null;
  if (!trigger) {
    trigger = extractErrorTrigger(prompt);
  }

  let shouldFail = false;
  if (trigger) {
    if (trigger === 'generation_failed') {
      shouldFail = true;
    } else {
      if (handleOpenAIError(res, trigger)) return;
    }
  }

  if (!model) {
    res.status(400).json({
      error: { message: "'model' is required", type: 'invalid_request_error', code: 'missing_required_parameter' },
    });
    return;
  }

  const videoId = `video-${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`;
  const now = Math.floor(Date.now() / 1000);
  const size = `${width}x${height}`;

  const video: VideoRecord = {
    id: videoId,
    object: 'video',
    model,
    status: 'queued',
    created_at: now,
    completed_at: null,
    seconds,
    size,
    error: null,
    outputs: null,
    _width: width,
    _height: height,
    _fps: fps,
    _prompt: prompt,
  };

  videosDb.set(videoId, video);

  progressWorker(videoId, MOCK_GENERATION_SECONDS * 1000, shouldFail);

  res.json({ id: videoId });
}

app.post('/v2/videos', createVideo);

// ── GET /v2/videos/:id ──────────────────────────────────────────────

function retrieveVideo(req: Request, res: Response): void {
  if (!requireBearerAuth(req, res, 'openai')) return;

  const videoId = String(req.params.id ?? req.params.video_id ?? req.params[0]);
  log.log(`/v2/videos/${videoId}`, 'GET', {});
  logConsole('GET', `/v2/videos/${videoId}`);

  const video = videosDb.get(videoId);
  if (!video) {
    res.status(404).json({
      error: { message: `No video found with id '${videoId}'`, type: 'invalid_request_error', code: 'not_found' },
    });
    return;
  }

  const publicFields: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(video)) {
    if (!k.startsWith('_')) publicFields[k] = v;
  }
  res.json(publicFields);
}

app.get('/v2/videos/:id', retrieveVideo);

// ── /v1/ aliases ────────────────────────────────────────────────────

app.post('/v1/videos', createVideo);
app.get('/v1/videos/:id', retrieveVideo);

// ── POST /v1/images/generations ─────────────────────────────────────

app.post('/v1/images/generations', (req: Request, res: Response): void => {
  const queryTrigger = req.query.mock_error as string | undefined;
  if (queryTrigger) {
    log.log('/v1/images/generations', 'POST', { mock_error: queryTrigger });
    logConsole('POST', '/v1/images/generations', { mock_error: queryTrigger });
    if (handleOpenAIError(res, queryTrigger)) return;
  }

  if (!requireBearerAuth(req, res, 'openai')) return;

  const data = req.body ?? {};

  const model: string = data.model ?? '';
  const prompt: string = data.prompt ?? '';
  const width = parseInt(data.width ?? '1024', 10);
  const height = parseInt(data.height ?? '1024', 10);
  const n = parseInt(data.n ?? '1', 10);
  const responseFormat: string = data.response_format ?? 'url';
  const steps: number | undefined = data.steps;
  const seed: number | undefined = data.seed;

  const params: Record<string, unknown> = {
    model, prompt: prompt.slice(0, 80), width, height,
    n, response_format: responseFormat, steps, seed,
  };

  if (!queryTrigger) {
    log.log('/v1/images/generations', 'POST', params, model);
    logConsole('POST', '/v1/images/generations', params);
  }

  let trigger = queryTrigger ?? null;
  if (!trigger) {
    trigger = extractErrorTrigger(prompt);
  }
  if (trigger) {
    if (handleOpenAIError(res, trigger)) return;
  }

  if (!model) {
    res.status(400).json({
      error: { message: "'model' is required", type: 'invalid_request_error', code: 'missing_required_parameter' },
    });
    return;
  }

  const pngBytes = generatePng(Math.min(width, 64), Math.min(height, 64));

  const results: Record<string, unknown>[] = [];
  for (let i = 0; i < n; i++) {
    if (responseFormat === 'b64_json') {
      results.push({
        index: i,
        b64_json: pngBytes.toString('base64'),
        revised_prompt: prompt,
      });
    } else {
      const imgId = crypto.randomUUID().replace(/-/g, '').slice(0, 16);
      imagesDb.set(`img-${imgId}`, { _png: pngBytes });
      results.push({
        index: i,
        url: `http://localhost:${PORT}/mock-cdn/img-${imgId}.png`,
        revised_prompt: prompt,
      });
    }
  }

  res.json({
    id: `img-${crypto.randomUUID().replace(/-/g, '').slice(0, 16)}`,
    model,
    object: 'list',
    created: Math.floor(Date.now() / 1000),
    data: results,
  });
});

// ── Mock CDN — video downloads ──────────────────────────────────────

app.get('/mock-cdn/:videoId.mp4', (req: Request, res: Response): void => {
  const videoId = String(req.params.videoId);
  const video = videosDb.get(videoId);

  if (!video) {
    res.status(404).json({ error: 'not found' });
    return;
  }
  if (video.status !== 'completed') {
    res.status(400).json({ error: 'video not ready' });
    return;
  }

  const w = video._width ?? 848;
  const h = video._height ?? 480;
  const dur = parseInt(video.seconds ?? '5', 10);

  const mp4Bytes = generateMp4(w, h, dur);
  res.set('Content-Type', 'video/mp4');
  res.set('Content-Disposition', `attachment; filename="${videoId}.mp4"`);
  res.send(mp4Bytes);
});

// ── Mock CDN — image downloads ──────────────────────────────────────

app.get('/mock-cdn/img-:imgId.png', (req: Request, res: Response): void => {
  const imgId = String(req.params.imgId);
  const entry = imagesDb.get(`img-${imgId}`);

  if (!entry) {
    res.status(404).json({ error: 'not found' });
    return;
  }

  res.set('Content-Type', 'image/png');
  res.send(entry._png);
});

// ── GET /v1/models ──────────────────────────────────────────────────

app.get('/v1/models', (_req: Request, res: Response): void => {
  res.json({
    object: 'list',
    data: [...VALID_MODELS].sort().map(m => ({
      id: m,
      object: 'model',
      type: 'video',
    })),
  });
});

// ── Start server ────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('  Mock Together AI Video & Image Generation API');
  console.log(`  Base URL: http://localhost:${PORT}`);
  console.log(`  Jobs complete after ${MOCK_GENERATION_SECONDS}s of polling`);
  console.log('');
  console.log('  Endpoints:');
  console.log('    POST /v2/videos                — create video job');
  console.log('    GET  /v2/videos/<id>           — poll job status');
  console.log('    POST /v1/images/generations    — generate image');
  console.log('    GET  /mock-cdn/<id>.mp4        — download video');
  console.log('');
  console.log('  Testing:');
  console.log('    GET    /_mock/log              — inspect received params');
  console.log('    DELETE /_mock/log              — clear log');
  console.log('    ?mock_error=<code>             — trigger error response');
  console.log('    MOCK_ERROR:<code> in prompt    — trigger error response');
  console.log('    Codes: auth, rate_limit, quota, server_error, invalid_model,');
  console.log('           content_policy, timeout, overloaded,');
  console.log('           generation_failed (async — job fails after polling)');
  console.log('');
  console.log('  Models: ' + [...VALID_MODELS].sort().join(', '));
  console.log('='.repeat(60));
});
