/**
 * Mock OpenAI Video Generation (Sora) API Server.
 *
 * Drop-in replacement for testing against the real API.
 * Point your OpenAI client at http://localhost:8100/v1
 *
 *   npx tsx src/openai-video.ts
 *
 * Monitoring:
 *   GET  /_mock/log                       -- view recent request log
 *   GET  /_mock/log?endpoint=/v1/videos   -- filter by endpoint
 *   DELETE /_mock/log                     -- clear log
 *
 * Error injection (in prompt text or query param):
 *   prompt="A cat MOCK_ERROR:rate_limit"  -- triggers 429 rate limit error
 *   POST /v1/videos?mock_error=auth       -- triggers 401 auth error
 *   generation_failed                     -- job fails asynchronously after polling
 */

import express, { type Request, type Response } from 'express';
import {
  RequestLog, mockLogRouter, extractErrorTrigger, handleOpenAIError,
  requireBearerAuth, logConsole, generatePng, generateMp4,
} from './shared.js';

// ── Types ───────────────────────────────────────────────────────────

interface VideoRecord {
  id: string;
  object: 'video';
  created_at: number;
  completed_at: number | null;
  expires_at: number | null;
  error: { code: string; message: string } | null;
  model: string;
  progress: number;
  prompt: string;
  remixed_from_video_id: string | null;
  seconds: string;
  size: string;
  status: 'queued' | 'in_progress' | 'completed' | 'failed';
}

// ── Constants ───────────────────────────────────────────────────────

const PORT = 18100;
const MOCK_GENERATION_SECONDS = 6;

const VALID_SIZES = new Set([
  '720x1280', '1280x720', '1024x1792', '1792x1024',
  '1080x1920', '1920x1080',
]);

const VALID_SECONDS = new Set(['4', '8', '12']);

const VALID_MODELS = new Set([
  'sora-2', 'sora-2-pro',
  'sora-2-2025-10-06', 'sora-2-pro-2025-10-06', 'sora-2-2025-12-08',
]);

// ── In-memory store ─────────────────────────────────────────────────

const videosDb = new Map<string, VideoRecord>();

// ── Background progress worker ──────────────────────────────────────

function startProgressWorker(videoId: string, totalSeconds: number, shouldFail: boolean): void {
  const steps = 10;
  const stepTime = (totalSeconds / steps) * 1000;
  let step = 0;

  const interval = setInterval(() => {
    step++;
    const video = videosDb.get(videoId);
    if (!video) {
      clearInterval(interval);
      return;
    }

    video.status = 'in_progress';
    video.progress = Math.min(step * 10, 100);

    if (step >= steps) {
      clearInterval(interval);
      if (shouldFail) {
        video.status = 'failed';
        video.error = {
          code: 'generation_failed',
          message: 'Video generation failed (MOCK_ERROR:generation_failed)',
        };
      } else {
        video.status = 'completed';
        video.progress = 100;
        video.completed_at = Math.floor(Date.now() / 1000);
        video.expires_at = Math.floor(Date.now() / 1000) + 3600;
      }
    }
  }, stepTime);
}

// ── Helpers ─────────────────────────────────────────────────────────

function randomHex(len: number): string {
  const chars = '0123456789abcdef';
  let result = '';
  for (let i = 0; i < len; i++) result += chars[Math.floor(Math.random() * 16)];
  return result;
}

// ── App ─────────────────────────────────────────────────────────────

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const log = new RequestLog();
app.use(mockLogRouter(log));

// ── POST /v1/videos ─────────────────────────────────────────────────

app.post('/v1/videos', (req: Request, res: Response) => {
  let trigger = req.query.mock_error as string | undefined;
  if (trigger) {
    log.log('/v1/videos', 'POST', { mock_error: trigger });
    if (handleOpenAIError(res, trigger)) return;
  }

  if (!requireBearerAuth(req, res, 'openai')) return;

  let prompt: string;
  let model: string;
  let seconds: string;
  let size: string;

  const contentType = req.headers['content-type'] ?? '';
  if (contentType.includes('multipart') || contentType.includes('urlencoded')) {
    prompt = (req.body?.prompt as string) ?? '';
    model = (req.body?.model as string) ?? 'sora-2';
    seconds = String(req.body?.seconds ?? '4');
    size = (req.body?.size as string) ?? '720x1280';
  } else {
    const data = req.body ?? {};
    prompt = data.prompt ?? '';
    model = data.model ?? 'sora-2';
    seconds = String(data.seconds ?? '4');
    size = data.size ?? '720x1280';
  }

  const params = { prompt, model, seconds, size };
  log.log('/v1/videos', 'POST', params, model);
  logConsole('POST', '/v1/videos', params);

  if (!trigger) {
    trigger = extractErrorTrigger(prompt, req.query.mock_error as string) ?? undefined;
  }

  let shouldFail = false;
  if (trigger) {
    if (trigger === 'generation_failed') {
      shouldFail = true;
    } else {
      if (handleOpenAIError(res, trigger)) return;
    }
  }

  if (!prompt) {
    res.status(400).json({
      error: { message: "'prompt' is required", type: 'invalid_request_error', code: 'missing_required_parameter' },
    });
    return;
  }

  if (!VALID_SIZES.has(size)) {
    const valid = [...VALID_SIZES].sort().join(', ');
    res.status(400).json({
      error: { message: `Invalid size '${size}'. Must be one of: ${valid}`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  if (!VALID_SECONDS.has(seconds)) {
    res.status(400).json({
      error: { message: `Invalid seconds '${seconds}'. Must be one of: 4, 8, 12`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
    return;
  }

  const videoId = `video_${randomHex(24)}`;
  const now = Math.floor(Date.now() / 1000);

  const video: VideoRecord = {
    id: videoId,
    object: 'video',
    created_at: now,
    completed_at: null,
    expires_at: null,
    error: null,
    model,
    progress: 0,
    prompt,
    remixed_from_video_id: null,
    seconds,
    size,
    status: 'queued',
  };

  videosDb.set(videoId, video);
  startProgressWorker(videoId, MOCK_GENERATION_SECONDS, shouldFail);

  res.status(200).json(video);
});

// ── GET /v1/videos/:id ──────────────────────────────────────────────

app.get('/v1/videos/:videoId', (req: Request, res: Response) => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  const videoId = req.params.videoId as string;
  log.log(`/v1/videos/${videoId}`, 'GET', {});
  console.log(`\n  GET /v1/videos/${videoId}`);

  const video = videosDb.get(videoId);
  if (!video) {
    res.status(404).json({
      error: { message: `No video found with id '${videoId}'`, type: 'invalid_request_error', code: 'not_found' },
    });
    return;
  }

  if (video.status === 'queued' || video.status === 'in_progress') {
    res.setHeader('openai-poll-after-ms', '1000');
  }

  res.status(200).json(video);
});

// ── GET /v1/videos ──────────────────────────────────────────────────

app.get('/v1/videos', (req: Request, res: Response) => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  const limit = parseInt(req.query.limit as string ?? '20') || 20;
  const order = (req.query.order as string ?? 'desc');
  const after = req.query.after as string | undefined;

  log.log('/v1/videos', 'GET', { limit, order, after: after ?? null });

  let allVideos = [...videosDb.values()];
  allVideos.sort((a, b) =>
    order === 'desc' ? b.created_at - a.created_at : a.created_at - b.created_at,
  );

  if (after) {
    const idx = allVideos.findIndex(v => v.id === after);
    if (idx !== -1) {
      allVideos = allVideos.slice(idx + 1);
    }
  }

  const page = allVideos.slice(0, limit);
  const hasMore = allVideos.length > limit;

  res.status(200).json({
    object: 'list',
    data: page,
    has_more: hasMore,
    first_id: page.length > 0 ? page[0]!.id : null,
    last_id: page.length > 0 ? page[page.length - 1]!.id : null,
  });
});

// ── DELETE /v1/videos/:id ───────────────────────────────────────────

app.delete('/v1/videos/:videoId', (req: Request, res: Response) => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  const videoId = req.params.videoId as string;
  log.log(`/v1/videos/${videoId}`, 'DELETE', {});

  if (!videosDb.has(videoId)) {
    res.status(404).json({
      error: { message: `No video found with id '${videoId}'`, type: 'invalid_request_error', code: 'not_found' },
    });
    return;
  }

  videosDb.delete(videoId);
  res.status(200).json({
    id: videoId,
    object: 'video.deleted',
    deleted: true,
  });
});

// ── GET /v1/videos/:id/content ──────────────────────────────────────

app.get('/v1/videos/:videoId/content', (req: Request, res: Response) => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  const videoId = req.params.videoId as string;
  const video = videosDb.get(videoId);

  if (!video) {
    res.status(404).json({
      error: { message: `No video found with id '${videoId}'`, type: 'invalid_request_error', code: 'not_found' },
    });
    return;
  }

  if (video.status !== 'completed') {
    res.status(400).json({
      error: { message: 'Video is not yet completed', type: 'invalid_request_error', code: 'video_not_ready' },
    });
    return;
  }

  const variant = (req.query.variant as string) ?? 'video';
  log.log(`/v1/videos/${videoId}/content`, 'GET', { variant });
  console.log(`\n  GET /v1/videos/${videoId}/content  variant=${variant}`);

  const [w, h] = video.size.split('x').map(Number) as [number, number];
  const dur = parseInt(video.seconds);

  if (variant === 'video') {
    const mp4Bytes = generateMp4(w, h, dur);
    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Content-Disposition', `attachment; filename="${videoId}.mp4"`);
    res.send(mp4Bytes);
  } else if (variant === 'thumbnail') {
    const png = generatePng(1, 1);
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Content-Disposition', `attachment; filename="${videoId}_thumb.png"`);
    res.send(png);
  } else if (variant === 'spritesheet') {
    const png = generatePng(1, 1);
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Content-Disposition', `attachment; filename="${videoId}_sprite.png"`);
    res.send(png);
  } else {
    res.status(400).json({
      error: { message: `Invalid variant '${variant}'`, type: 'invalid_request_error', code: 'invalid_parameter' },
    });
  }
});

// ── GET /v1/models ──────────────────────────────────────────────────

app.get('/v1/models', (_req: Request, res: Response) => {
  res.json({
    object: 'list',
    data: [...VALID_MODELS].sort().map(m => ({
      id: m,
      object: 'model',
      owned_by: 'openai-mock',
    })),
  });
});

// ── Start ───────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('  Mock OpenAI Video Generation API (Sora)');
  console.log(`  Base URL: http://localhost:${PORT}/v1`);
  console.log(`  Jobs complete after ${MOCK_GENERATION_SECONDS}s of polling`);
  console.log('');
  console.log('  Endpoints:');
  console.log('    POST   /v1/videos              -- create video job');
  console.log('    GET    /v1/videos/<id>          -- poll job status');
  console.log('    GET    /v1/videos               -- list all jobs');
  console.log('    DELETE /v1/videos/<id>          -- delete job');
  console.log('    GET    /v1/videos/<id>/content  -- download video/thumb/sprite');
  console.log('');
  console.log('  Testing:');
  console.log('    GET    /_mock/log               -- inspect received params');
  console.log('    DELETE /_mock/log               -- clear log');
  console.log('    ?mock_error=<code>              -- trigger error response');
  console.log('    MOCK_ERROR:<code> in prompt     -- trigger error response');
  console.log('    Codes: auth, rate_limit, quota, server_error, content_policy,');
  console.log('           billing, timeout, invalid_model, overloaded,');
  console.log('           generation_failed (async -- job fails after polling)');
  console.log('='.repeat(60));
});
