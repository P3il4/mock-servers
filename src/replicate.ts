/**
 * Mock Replicate API Server.
 *
 * Drop-in replacement for testing against the real Replicate API.
 * Point your Replicate client at http://localhost:18400 instead of https://api.replicate.com
 *
 *   npx tsx src/replicate.ts
 *
 * Then in your code:
 *   import Replicate from "replicate";
 *   const replicate = new Replicate({ auth: "mock-key", baseUrl: "http://localhost:18400" });
 *
 * Or in Python:
 *   import replicate
 *   client = replicate.Client(api_token="mock-key", base_url="http://localhost:18400")
 *
 * Endpoints:
 *   POST /v1/predictions                              — create prediction (by version)
 *   GET  /v1/predictions/:id                           — poll prediction status
 *   GET  /v1/predictions                               — list predictions
 *   POST /v1/predictions/:id/cancel                    — cancel prediction
 *   POST /v1/models/:owner/:name/predictions           — create prediction (by model)
 *   GET  /v1/models/:owner/:name                       — get model info
 *   GET  /v1/models                                    — list models
 *   POST /v1/deployments/:owner/:name/predictions      — create prediction via deployment
 *   GET  /v1/deployments/:owner/:name                  — get deployment info
 *   GET  /v1/webhooks/default/secret                   — get webhook signing secret
 *   GET  /mock-cdn/:id.png                             — download generated image
 *
 * Testing:
 *   GET    /_mock/log              — inspect received params
 *   DELETE /_mock/log              — clear log
 *   ?mock_error=<code>             — trigger error response
 *   MOCK_ERROR:<code> in prompt    — trigger error response
 *   Codes: auth, rate_limit, not_found, server_error, invalid_model,
 *          timeout, content_policy, overloaded,
 *          prediction_failed (async — prediction fails after polling)
 */

import express, { type Request, type Response } from 'express';
import crypto from 'node:crypto';
import {
  RequestLog, mockLogRouter, extractErrorTrigger,
  requireBearerAuth, logConsole, generatePng, corsMiddleware,
} from './shared.js';

const PORT = 18400;
const MOCK_PROCESSING_SECONDS = 4;

// ── Replicate Error Format (RFC 7807) ──────────────────────────────

const REPLICATE_ERRORS: Record<string, [number, string, string]> = {
  auth:           [401, 'Unauthorized',              'Invalid API token. Please check your token and try again.'],
  rate_limit:     [429, 'Too Many Requests',         'Rate limit exceeded. Please try again later.'],
  not_found:      [404, 'Not Found',                 'The requested resource was not found.'],
  server_error:   [500, 'Internal Server Error',     'An unexpected error occurred. Please try again later.'],
  invalid_model:  [404, 'Model Not Found',           'The model you specified does not exist or you do not have access to it.'],
  timeout:        [504, 'Gateway Timeout',           'The request timed out. Please try again.'],
  content_policy: [422, 'Content Policy Violation',  'Your request was rejected because it violates our content policy.'],
  overloaded:     [503, 'Service Unavailable',       'The service is temporarily overloaded. Please try again later.'],
  billing:        [402, 'Payment Required',          'Your account does not have an active billing method. Please add one at replicate.com/account/billing.'],
  permission:     [403, 'Forbidden',                 'You do not have permission to access this resource.'],
};

function handleReplicateError(res: Response, trigger: string): boolean {
  const err = REPLICATE_ERRORS[trigger];
  if (!err) return false;
  res.status(err[0]).json({
    type: 'about:blank',
    title: err[1],
    status: err[0],
    detail: err[2],
  });
  return true;
}

// ── Mock Models ────────────────────────────────────────────────────

const MOCK_MODELS: Record<string, { owner: string; name: string; description: string; visibility: string; run_count: number }> = {
  'stability-ai/sdxl': {
    owner: 'stability-ai', name: 'sdxl',
    description: 'A text-to-image generative AI model that creates beautiful images',
    visibility: 'public', run_count: 50000000,
  },
  'meta/llama-2-70b-chat': {
    owner: 'meta', name: 'llama-2-70b-chat',
    description: 'A 70 billion parameter language model from Meta, fine tuned for chat completions',
    visibility: 'public', run_count: 30000000,
  },
  'black-forest-labs/flux-schnell': {
    owner: 'black-forest-labs', name: 'flux-schnell',
    description: 'The fastest image generation model tailored for local development and personal use',
    visibility: 'public', run_count: 40000000,
  },
  'minimax/video-01': {
    owner: 'minimax', name: 'video-01',
    description: 'Generate 6 second videos with prompts or images.',
    visibility: 'public', run_count: 5000000,
  },
};

const MOCK_VERSION = 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2';

// ── In-memory stores ───────────────────────────────────────────────

interface PredictionRecord {
  id: string;
  model: string;
  version: string;
  status: 'starting' | 'processing' | 'succeeded' | 'failed' | 'canceled';
  input: Record<string, unknown>;
  output: unknown;
  error: string | null;
  logs: string;
  metrics: Record<string, unknown> | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  urls: { get: string; cancel: string; stream?: string };
  _isImageModel: boolean;
}

const predictionsDb = new Map<string, PredictionRecord>();
const imageStore = new Map<string, Buffer>();

// ── Background progress ────────────────────────────────────────────

function progressWorker(predictionId: string, totalMs: number, shouldFail: boolean): void {
  const steps = 5;
  const stepMs = totalMs / steps;

  for (let i = 1; i <= steps; i++) {
    setTimeout(() => {
      const pred = predictionsDb.get(predictionId);
      if (!pred || pred.status === 'canceled') return;
      pred.status = 'processing';
      pred.started_at = pred.started_at ?? new Date().toISOString();
      const pct = Math.min(Math.round((i / steps) * 100), 99);
      pred.logs = `  ${pct}%|${'█'.repeat(Math.floor(pct / 5))}${'░'.repeat(20 - Math.floor(pct / 5))}| ${i}/${steps}`;
    }, stepMs * i);
  }

  setTimeout(() => {
    const pred = predictionsDb.get(predictionId);
    if (!pred || pred.status === 'canceled') return;
    if (shouldFail) {
      pred.status = 'failed';
      pred.error = 'Model encountered an error during inference (MOCK_ERROR:prediction_failed)';
      pred.completed_at = new Date().toISOString();
      pred.logs += '\nError: prediction failed';
    } else {
      pred.status = 'succeeded';
      pred.completed_at = new Date().toISOString();
      pred.logs += '\nCompleted';
      pred.metrics = { predict_time: totalMs / 1000 };
      if (pred._isImageModel) {
        const png = generatePng(64, 64);
        const imgId = crypto.randomUUID().replace(/-/g, '').slice(0, 16);
        imageStore.set(imgId, png);
        pred.output = [`http://localhost:${PORT}/mock-cdn/${imgId}.png`];
      } else {
        pred.output = 'This is a mock response from the Replicate API. The model would normally generate real output here.';
      }
    }
  }, totalMs);
}

// ── Express app ────────────────────────────────────────────────────

const app = express();
app.use(corsMiddleware);
app.use(express.json({ limit: '50mb' }));

const log = new RequestLog();
app.use(mockLogRouter(log));

// ── Helper: detect image model ─────────────────────────────────────

function isImageModel(model: string): boolean {
  const lower = model.toLowerCase();
  return lower.includes('sdxl') || lower.includes('flux') || lower.includes('stable-diffusion')
    || lower.includes('dall') || lower.includes('imagen') || lower.includes('kandinsky');
}

// ── Helper: build prediction response ──────────────────────────────

function predictionToJson(pred: PredictionRecord): Record<string, unknown> {
  return {
    id: pred.id,
    model: pred.model,
    version: pred.version,
    status: pred.status,
    input: pred.input,
    output: pred.output,
    error: pred.error,
    logs: pred.logs || '',
    metrics: pred.metrics,
    created_at: pred.created_at,
    started_at: pred.started_at,
    completed_at: pred.completed_at,
    urls: pred.urls,
  };
}

// ── Helper: create prediction from request ─────────────────────────

function createPrediction(
  req: Request,
  res: Response,
  endpointPath: string,
  model: string,
  version: string,
): void {
  const queryTrigger = req.query.mock_error as string | undefined;
  if (queryTrigger) {
    log.log(endpointPath, 'POST', { mock_error: queryTrigger });
    logConsole('POST', endpointPath, { mock_error: queryTrigger });
    if (handleReplicateError(res, queryTrigger)) return;
  }

  if (!requireBearerAuth(req, res, 'openai')) return;

  const data = req.body ?? {};
  const input: Record<string, unknown> = data.input ?? {};
  const prompt = String(input.prompt ?? '');

  const params: Record<string, unknown> = {
    model, version,
    input: { prompt: prompt.slice(0, 80), ...Object.fromEntries(Object.entries(input).filter(([k]) => k !== 'prompt').slice(0, 10)) },
    webhook: data.webhook,
    stream: data.stream,
  };

  if (!queryTrigger) {
    log.log(endpointPath, 'POST', params, model);
    logConsole('POST', endpointPath, params);
  }

  let trigger = queryTrigger ?? null;
  if (!trigger) trigger = extractErrorTrigger(prompt);

  let shouldFail = false;
  if (trigger) {
    if (trigger === 'prediction_failed') {
      shouldFail = true;
    } else {
      if (handleReplicateError(res, trigger)) return;
    }
  }

  const predId = crypto.randomUUID().replace(/-/g, '').slice(0, 24);
  const now = new Date().toISOString();

  const pred: PredictionRecord = {
    id: predId,
    model,
    version,
    status: 'starting',
    input,
    output: null,
    error: null,
    logs: '',
    metrics: null,
    created_at: now,
    started_at: null,
    completed_at: null,
    urls: {
      get: `http://localhost:${PORT}/v1/predictions/${predId}`,
      cancel: `http://localhost:${PORT}/v1/predictions/${predId}/cancel`,
    },
    _isImageModel: isImageModel(model),
  };

  predictionsDb.set(predId, pred);
  progressWorker(predId, MOCK_PROCESSING_SECONDS * 1000, shouldFail);

  res.status(201).json(predictionToJson(pred));
}

// ── POST /v1/predictions ───────────────────────────────────────────

app.post('/v1/predictions', (req: Request, res: Response): void => {
  const data = req.body ?? {};
  const version: string = data.version ?? MOCK_VERSION;
  const model = data.model ?? 'unknown/unknown';
  createPrediction(req, res, '/v1/predictions', model, version);
});

// ── GET /v1/predictions/:id ────────────────────────────────────────

app.get('/v1/predictions/:id', (req: Request, res: Response): void => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  const predId = String(req.params.id);
  log.log(`/v1/predictions/${predId}`, 'GET', {});
  logConsole('GET', `/v1/predictions/${predId}`);

  const pred = predictionsDb.get(predId);
  if (!pred) {
    res.status(404).json({
      type: 'about:blank',
      title: 'Not Found',
      status: 404,
      detail: `No prediction found with id '${predId}'`,
    });
    return;
  }

  res.json(predictionToJson(pred));
});

// ── GET /v1/predictions ────────────────────────────────────────────

app.get('/v1/predictions', (req: Request, res: Response): void => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  log.log('/v1/predictions', 'GET', {});
  logConsole('GET', '/v1/predictions');

  const all = [...predictionsDb.values()].reverse().slice(0, 50);
  res.json({
    previous: null,
    next: null,
    results: all.map(predictionToJson),
  });
});

// ── POST /v1/predictions/:id/cancel ────────────────────────────────

app.post('/v1/predictions/:id/cancel', (req: Request, res: Response): void => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  const predId = String(req.params.id);
  log.log(`/v1/predictions/${predId}/cancel`, 'POST', {});
  logConsole('POST', `/v1/predictions/${predId}/cancel`);

  const pred = predictionsDb.get(predId);
  if (!pred) {
    res.status(404).json({
      type: 'about:blank',
      title: 'Not Found',
      status: 404,
      detail: `No prediction found with id '${predId}'`,
    });
    return;
  }

  if (pred.status === 'starting' || pred.status === 'processing') {
    pred.status = 'canceled';
    pred.completed_at = new Date().toISOString();
  }

  res.json(predictionToJson(pred));
});

// ── POST /v1/models/:owner/:name/predictions ───────────────────────

app.post('/v1/models/:owner/:name/predictions', (req: Request, res: Response): void => {
  const model = `${req.params.owner}/${req.params.name}`;
  createPrediction(req, res, `/v1/models/${model}/predictions`, model, MOCK_VERSION);
});

// ── GET /v1/models/:owner/:name ────────────────────────────────────

app.get('/v1/models/:owner/:name', (req: Request, res: Response): void => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  const key = `${req.params.owner}/${req.params.name}`;
  log.log(`/v1/models/${key}`, 'GET', {});
  logConsole('GET', `/v1/models/${key}`);

  const mock = MOCK_MODELS[key];
  if (mock) {
    res.json({
      url: `https://replicate.com/${key}`,
      owner: mock.owner,
      name: mock.name,
      description: mock.description,
      visibility: mock.visibility,
      github_url: null,
      paper_url: null,
      license_url: null,
      run_count: mock.run_count,
      cover_image_url: null,
      default_example: null,
      latest_version: {
        id: MOCK_VERSION,
        created_at: '2024-01-01T00:00:00.000Z',
        cog_version: '0.9.0',
        openapi_schema: {},
      },
    });
    return;
  }

  // Return generic model info for any owner/name
  res.json({
    url: `https://replicate.com/${key}`,
    owner: req.params.owner,
    name: req.params.name,
    description: `Mock model ${key}`,
    visibility: 'public',
    github_url: null,
    paper_url: null,
    license_url: null,
    run_count: 1000,
    cover_image_url: null,
    default_example: null,
    latest_version: {
      id: MOCK_VERSION,
      created_at: '2024-01-01T00:00:00.000Z',
      cog_version: '0.9.0',
      openapi_schema: {},
    },
  });
});

// ── GET /v1/models ─────────────────────────────────────────────────

app.get('/v1/models', (req: Request, res: Response): void => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  log.log('/v1/models', 'GET', {});
  logConsole('GET', '/v1/models');

  res.json({
    previous: null,
    next: null,
    results: Object.entries(MOCK_MODELS).map(([key, m]) => ({
      url: `https://replicate.com/${key}`,
      owner: m.owner,
      name: m.name,
      description: m.description,
      visibility: m.visibility,
      github_url: null,
      paper_url: null,
      license_url: null,
      run_count: m.run_count,
      cover_image_url: null,
      default_example: null,
      latest_version: null,
    })),
  });
});

// ── POST /v1/deployments/:owner/:name/predictions ──────────────────

app.post('/v1/deployments/:owner/:name/predictions', (req: Request, res: Response): void => {
  const deployment = `${req.params.owner}/${req.params.name}`;
  createPrediction(req, res, `/v1/deployments/${deployment}/predictions`, deployment, MOCK_VERSION);
});

// ── GET /v1/deployments/:owner/:name ───────────────────────────────

app.get('/v1/deployments/:owner/:name', (req: Request, res: Response): void => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  const owner = req.params.owner;
  const name = req.params.name;
  log.log(`/v1/deployments/${owner}/${name}`, 'GET', {});
  logConsole('GET', `/v1/deployments/${owner}/${name}`);

  res.json({
    owner,
    name,
    current_release: {
      number: 1,
      model: `${owner}/model`,
      version: MOCK_VERSION,
      created_at: '2024-01-01T00:00:00.000Z',
      created_by: { type: 'organization', username: owner, name: owner, github_url: null },
      configuration: { hardware: 'gpu-a40-large', min_instances: 1, max_instances: 5 },
    },
  });
});

// ── GET /v1/webhooks/default/secret ────────────────────────────────

app.get('/v1/webhooks/default/secret', (req: Request, res: Response): void => {
  if (!requireBearerAuth(req, res, 'openai')) return;

  log.log('/v1/webhooks/default/secret', 'GET', {});
  logConsole('GET', '/v1/webhooks/default/secret');

  res.json({ key: 'whsec_dGVzdF9zZWNyZXRfa2V5' });
});

// ── Mock CDN — image downloads ─────────────────────────────────────

app.get('/mock-cdn/:imgId.png', (req: Request, res: Response): void => {
  const imgId = String(req.params.imgId);
  const img = imageStore.get(imgId);
  if (!img) {
    res.status(404).json({ detail: 'not found' });
    return;
  }
  res.set('Content-Type', 'image/png');
  res.send(img);
});

// ── Start server ───────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('  Mock Replicate API Server');
  console.log(`  Base URL: http://localhost:${PORT}`);
  console.log(`  Predictions complete after ${MOCK_PROCESSING_SECONDS}s of polling`);
  console.log('');
  console.log('  Endpoints:');
  console.log('    POST /v1/predictions                            — create prediction');
  console.log('    GET  /v1/predictions/<id>                       — poll prediction');
  console.log('    GET  /v1/predictions                            — list predictions');
  console.log('    POST /v1/predictions/<id>/cancel                — cancel prediction');
  console.log('    POST /v1/models/<owner>/<name>/predictions      — create via model');
  console.log('    GET  /v1/models/<owner>/<name>                  — get model info');
  console.log('    POST /v1/deployments/<owner>/<name>/predictions — create via deployment');
  console.log('    GET  /v1/deployments/<owner>/<name>             — get deployment info');
  console.log('    GET  /v1/webhooks/default/secret                — webhook secret');
  console.log('');
  console.log('  Testing:');
  console.log('    GET    /_mock/log              — inspect received params');
  console.log('    DELETE /_mock/log              — clear log');
  console.log('    ?mock_error=<code>             — trigger error response');
  console.log('    MOCK_ERROR:<code> in prompt    — trigger error response');
  console.log('    Codes: auth, rate_limit, not_found, server_error, invalid_model,');
  console.log('           timeout, content_policy, overloaded, billing, permission,');
  console.log('           prediction_failed (async — prediction fails after polling)');
  console.log('');
  console.log('  Models: ' + Object.keys(MOCK_MODELS).join(', '));
  console.log('='.repeat(60));
});
