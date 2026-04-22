/**
 * Mock xAI Video Generation API Server (port 18600).
 *
 * Implements `grok-imagine-video` submit + polling:
 *   POST /v1/videos/generations          Submit a job → { request_id }
 *   GET  /v1/videos/:request_id          Poll → { status, video?, model }
 *   GET  /v1/mock_cdn/:id.mp4            Serve the generated MP4
 *
 *   npx tsx src/xai-video.ts
 *
 * Polling behaviour:
 *   - Default: job completes on the first poll (fast tests).
 *   - `?pending_polls=N` on submit → first N polls return 'pending', the N+1th returns 'done'.
 *
 * Error injection:
 *   - MOCK_ERROR:<code> in prompt, or ?mock_error=<code> on submit — rejects the
 *     submit with an OpenAI-style error.
 *   - `?fail=1` on submit → poll eventually returns status:'failed'.
 *   - `?expire=1` on submit → poll eventually returns status:'expired'.
 *
 * Monitoring: GET/DELETE /_mock/log
 */

import express, { type Request, type Response } from 'express';
import crypto from 'node:crypto';
import {
  RequestLog, mockLogRouter, extractErrorTrigger, handleOpenAIError,
  requireBearerAuth, logConsole, generateMp4, corsMiddleware,
} from './shared.js';

const PORT = 18600;

const VALID_MODELS = new Set(['grok-imagine-video']);
const VALID_ASPECT_RATIOS = new Set(['1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3']);
const VALID_RESOLUTIONS = new Set(['480p', '720p']);

const DEFAULT_ASPECT_RATIO = '16:9';
const DEFAULT_RESOLUTION = '480p';
const DEFAULT_DURATION = 10;
const MIN_DURATION = 1;
const MAX_DURATION = 15;

const RESOLUTION_DIMENSIONS: Record<string, [number, number]> = {
  '480p': [854, 480],
  '720p': [1280, 720],
};

type JobStatus = 'pending' | 'done' | 'expired' | 'failed';

interface Job {
  request_id: string;
  model: string;
  prompt: string;
  duration: number;
  aspect_ratio: string;
  resolution: string;
  pollCount: number;
  pendingPolls: number;          // number of polls that return 'pending' before flipping
  terminalStatus: JobStatus;     // what flip produces — 'done' | 'failed' | 'expired'
  errorMessage?: string;
  videoId?: string;              // set when we mint the mp4
  createdAt: number;
}

const jobs = new Map<string, Job>();
const videoStore = new Map<string, Buffer>();

// ── Helpers ──────────────────────────────────────────────────────────

function parsePositiveInt(value: unknown, fallback: number): number {
  if (value === undefined || value === null) return fallback;
  const n = typeof value === 'number' ? value : Number.parseInt(String(value), 10);
  if (!Number.isFinite(n) || n < 0) return fallback;
  return Math.floor(n);
}

function clampDuration(raw: unknown): number {
  const n = typeof raw === 'number' ? raw : Number.parseInt(String(raw), 10);
  if (!Number.isFinite(n)) return DEFAULT_DURATION;
  return Math.max(MIN_DURATION, Math.min(MAX_DURATION, Math.round(n)));
}

function normalizeAspectRatio(raw: unknown): string {
  if (typeof raw === 'string' && VALID_ASPECT_RATIOS.has(raw)) return raw;
  return DEFAULT_ASPECT_RATIO;
}

function normalizeResolution(raw: unknown): string {
  if (typeof raw === 'string' && VALID_RESOLUTIONS.has(raw.toLowerCase())) {
    return raw.toLowerCase();
  }
  return DEFAULT_RESOLUTION;
}

// ── App ──────────────────────────────────────────────────────────────

const app = express();
app.use(corsMiddleware);
app.use(express.json({ limit: '50mb' }));

const log = new RequestLog();
app.use(mockLogRouter(log));

// ── POST /v1/videos/generations ─────────────────────────────────────

app.post('/v1/videos/generations', (req: Request, res: Response) => {
  const data = (req.body ?? {}) as Record<string, unknown>;
  const prompt = typeof data.prompt === 'string' ? data.prompt : '';
  const model = typeof data.model === 'string' ? data.model : '';
  const duration = clampDuration(data.duration);
  const aspect_ratio = normalizeAspectRatio(data.aspect_ratio);
  const resolution = normalizeResolution(data.resolution);
  const hasImage = typeof data.image === 'string' && (data.image as string).length > 0;
  const hasRefs = Array.isArray(data.reference_images) && (data.reference_images as unknown[]).length > 0;

  const params: Record<string, unknown> = {
    prompt, model, duration, aspect_ratio, resolution,
    has_image: hasImage, has_reference_images: hasRefs,
  };
  log.log('/v1/videos/generations', 'POST', params, model);
  logConsole('POST', '/v1/videos/generations', params);

  if (!requireBearerAuth(req, res, 'openai')) return;

  const trigger = (req.query.mock_error as string | undefined)
    ?? extractErrorTrigger(prompt, undefined) ?? undefined;
  if (trigger) {
    if (handleOpenAIError(res, trigger)) return;
  }

  if (!prompt) {
    res.status(400).json({
      error: { message: "'prompt' is required.", type: 'invalid_request_error', code: 'missing_required_parameter' },
    });
    return;
  }
  if (!model) {
    res.status(400).json({
      error: { message: "'model' is required.", type: 'invalid_request_error', code: 'missing_required_parameter' },
    });
    return;
  }
  if (!VALID_MODELS.has(model)) {
    res.status(400).json({
      error: { message: `The model '${model}' does not exist or you do not have access to it.`, type: 'invalid_request_error', code: 'model_not_found' },
    });
    return;
  }

  const request_id = `vidgen-${crypto.randomBytes(12).toString('hex')}`;
  const pendingPolls = parsePositiveInt(req.query.pending_polls, 0);
  const shouldFail = req.query.fail === '1' || req.query.fail === 'true';
  const shouldExpire = req.query.expire === '1' || req.query.expire === 'true';

  const job: Job = {
    request_id,
    model,
    prompt,
    duration,
    aspect_ratio,
    resolution,
    pollCount: 0,
    pendingPolls,
    terminalStatus: shouldFail ? 'failed' : shouldExpire ? 'expired' : 'done',
    errorMessage: shouldFail ? 'Simulated generation failure' : shouldExpire ? 'Request timed out' : undefined,
    createdAt: Date.now(),
  };
  jobs.set(request_id, job);

  res.status(200).json({ request_id });
});

// ── GET /v1/videos/:request_id ──────────────────────────────────────

app.get('/v1/videos/:request_id', (req: Request, res: Response) => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  const request_id = req.params.request_id!;
  const job = jobs.get(request_id);

  log.log('/v1/videos/:request_id', 'GET', { request_id }, job?.model);

  if (!job) {
    res.status(404).json({
      error: { message: `No video request with id '${request_id}'.`, type: 'invalid_request_error', code: 'not_found' },
    });
    return;
  }

  job.pollCount++;
  const shouldFlip = job.pollCount > job.pendingPolls;

  if (!shouldFlip) {
    res.status(200).json({ status: 'pending', model: job.model });
    return;
  }

  if (job.terminalStatus === 'failed' || job.terminalStatus === 'expired') {
    res.status(200).json({
      status: job.terminalStatus,
      model: job.model,
      error: job.errorMessage,
    });
    return;
  }

  // done — mint mp4 if not already minted
  if (!job.videoId) {
    const [w, h] = RESOLUTION_DIMENSIONS[job.resolution] ?? RESOLUTION_DIMENSIONS['480p']!;
    const mp4 = generateMp4(w, h, job.duration);
    const videoId = crypto.randomBytes(12).toString('hex');
    videoStore.set(videoId, mp4);
    job.videoId = videoId;
  }

  const host = req.headers.host ?? `localhost:${PORT}`;
  const videoUrl = `http://${host}/v1/mock_cdn/${job.videoId}.mp4`;

  res.status(200).json({
    status: 'done',
    model: job.model,
    video: {
      url: videoUrl,
      duration: job.duration,
      respect_moderation: true,
    },
  });
});

// ── GET /v1/mock_cdn/:videoFile ────────────────────────────────────

app.get('/v1/mock_cdn/:videoFile', (req: Request, res: Response) => {
  const videoId = (req.params.videoFile ?? '').replace(/\.mp4$/, '');
  const mp4 = videoStore.get(videoId);
  if (!mp4) {
    res.status(404).json({
      error: { message: `Video '${videoId}' not found or has expired.`, type: 'invalid_request_error', code: 'not_found' },
    });
    return;
  }
  res.setHeader('Content-Type', 'video/mp4');
  res.setHeader('Content-Disposition', `inline; filename="${videoId}.mp4"`);
  res.send(mp4);
});

// ── GET /v1/video-generation-models ─────────────────────────────────

app.get('/v1/video-generation-models', (_req: Request, res: Response) => {
  log.log('/v1/video-generation-models', 'GET', {});
  res.json({
    models: [...VALID_MODELS].sort().map(id => ({
      id,
      created: Math.floor(Date.now() / 1000),
      object: 'model',
      owned_by: 'xai-mock',
      input_modalities: ['text', 'image'],
      output_modalities: ['video'],
    })),
  });
});

// ── Start ───────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('='.repeat(64));
  console.log('  Mock xAI Video Generation API (Grok Imagine Video)');
  console.log(`  Base URL: http://localhost:${PORT}/v1`);
  console.log('');
  console.log('  Endpoints:');
  console.log('    POST /v1/videos/generations      -- submit a job');
  console.log('    GET  /v1/videos/:request_id      -- poll job status');
  console.log('    GET  /v1/mock_cdn/<id>.mp4       -- serve generated video');
  console.log('    GET  /v1/video-generation-models -- list available models');
  console.log('');
  console.log('  Testing:');
  console.log('    GET    /_mock/log                -- inspect received params');
  console.log('    DELETE /_mock/log                -- clear log');
  console.log('    ?mock_error=<code>               -- reject submit with error');
  console.log('    MOCK_ERROR:<code> in prompt      -- reject submit with error');
  console.log('    ?pending_polls=N                 -- simulate N pending polls');
  console.log('    ?fail=1                          -- job ends as status:failed');
  console.log('    ?expire=1                        -- job ends as status:expired');
  console.log('');
  console.log('  Models: ' + [...VALID_MODELS].sort().join(', '));
  console.log('='.repeat(64));
});
