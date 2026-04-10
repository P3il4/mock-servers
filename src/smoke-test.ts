/**
 * Smoke tests for all mock servers.
 * Verifies each server is running and responds correctly.
 *
 *   npx tsx src/smoke-test.ts
 *
 * Exit code 0 = all passed, 1 = failures.
 */

export {}; // module marker for top-level await

interface TestCase {
  method: string;
  path: string;
  headers?: Record<string, string>;
  body?: Record<string, unknown>;
  expectStatus?: number;
  expect: (r: any) => unknown;
}

const SERVERS: { name: string; port: number; tests: TestCase[] }[] = [
  { name: 'Gemini Image/Video', port: 18080, tests: [
    { method: 'POST', path: '/v1beta/models/gemini-2.5-flash:generateContent',
      body: { contents: [{ role: 'user', parts: [{ text: 'hello' }] }] },
      expect: (r: any) => r.candidates?.[0]?.content?.parts?.[0]?.text },
    { method: 'GET', path: '/_mock/log?last=1',
      expect: (r: any) => Array.isArray(r) },
  ]},
  { name: 'Gemini Chat', port: 18180, tests: [
    { method: 'POST', path: '/v1beta/models/gemini-2.5-flash:generateContent',
      body: { contents: [{ role: 'user', parts: [{ text: 'hello' }] }] },
      expect: (r: any) => r.candidates?.[0]?.content?.parts?.[0]?.text },
    { method: 'GET', path: '/_mock/log?last=1',
      expect: (r: any) => Array.isArray(r) },
  ]},
  { name: 'OpenAI Video', port: 18100, tests: [
    { method: 'POST', path: '/v1/videos',
      headers: { Authorization: 'Bearer mock' },
      body: { prompt: 'a cat', model: 'sora-2', seconds: '4', size: '1280x720' },
      expect: (r: any) => r.id && r.status === 'queued' },
    { method: 'GET', path: '/_mock/log?last=1',
      expect: (r: any) => Array.isArray(r) },
  ]},
  { name: 'OpenAI Image', port: 18150, tests: [
    { method: 'POST', path: '/v1/images/generations',
      headers: { Authorization: 'Bearer mock' },
      body: { prompt: 'a cat', model: 'dall-e-3', size: '1024x1024' },
      expect: (r: any) => r.data?.[0] },
    { method: 'GET', path: '/_mock/log?last=1',
      expect: (r: any) => Array.isArray(r) },
  ]},
  { name: 'Together AI', port: 18200, tests: [
    { method: 'POST', path: '/v2/videos',
      headers: { Authorization: 'Bearer mock' },
      body: { model: 'Wan-AI/Wan2.1-T2V-14B', prompt: 'a cat' },
      expect: (r: any) => r.id },
    { method: 'GET', path: '/_mock/log?last=1',
      expect: (r: any) => Array.isArray(r) },
  ]},
  { name: 'Cloudflare Image', port: 18300, tests: [
    { method: 'POST', path: '/client/v4/accounts/test/ai/run/@cf/black-forest-labs/flux-1-schnell',
      headers: { Authorization: 'Bearer mock' },
      body: { prompt: 'a cat' },
      expect: (r: any) => r.success === true && r.result?.image },
    { method: 'GET', path: '/_mock/log?last=1',
      expect: (r: any) => Array.isArray(r) },
  ]},
  { name: 'Cloudflare Text', port: 18350, tests: [
    { method: 'POST', path: '/client/v4/accounts/test/ai/run/@cf/meta/llama-3.1-8b-instruct',
      headers: { Authorization: 'Bearer mock' },
      body: { messages: [{ role: 'user', content: 'hello' }] },
      expect: (r: any) => r.success === true && r.result?.response },
    { method: 'GET', path: '/_mock/log?last=1',
      expect: (r: any) => Array.isArray(r) },
  ]},
  // Error injection tests
  { name: 'OpenAI Video (error)', port: 18100, tests: [
    { method: 'POST', path: '/v1/videos?mock_error=rate_limit',
      headers: { Authorization: 'Bearer mock' },
      body: { prompt: 'test' },
      expectStatus: 429,
      expect: (r: any) => r.error?.code === 'rate_limit_exceeded' },
  ]},
  { name: 'Cloudflare Image (error)', port: 18300, tests: [
    { method: 'POST', path: '/client/v4/accounts/test/ai/run/@cf/black-forest-labs/flux-1-schnell?mock_error=quota',
      headers: { Authorization: 'Bearer mock' },
      body: { prompt: 'test' },
      expectStatus: 429,
      expect: (r: any) => r.errors?.[0]?.code === 3036 },
  ]},
];

let passed = 0;
let failed = 0;

for (const server of SERVERS) {
  for (const test of server.tests) {
    const url = `http://localhost:${server.port}${test.path}`;
    try {
      const opts: RequestInit = {
        method: test.method,
        headers: {
          'Content-Type': 'application/json',
          ...(test.headers ?? {}),
        },
      };
      if (test.body) opts.body = JSON.stringify(test.body);

      const resp = await fetch(url, opts);
      const expectedStatus = (test as any).expectStatus ?? 200;
      if (resp.status !== expectedStatus) {
        console.log(`  FAIL  ${server.name} ${test.method} ${test.path} — status ${resp.status} (expected ${expectedStatus})`);
        failed++;
        continue;
      }
      const json = await resp.json();
      if (test.expect(json)) {
        console.log(`  PASS  ${server.name} ${test.method} ${test.path}`);
        passed++;
      } else {
        console.log(`  FAIL  ${server.name} ${test.method} ${test.path} — unexpected response`);
        console.log(`        ${JSON.stringify(json).slice(0, 200)}`);
        failed++;
      }
    } catch (e) {
      console.log(`  FAIL  ${server.name} ${test.method} ${test.path} — ${(e as Error).message}`);
      failed++;
    }
  }
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
