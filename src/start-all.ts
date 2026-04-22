/**
 * Start all mock API servers concurrently.
 *
 *   npx tsx src/start-all.ts
 *
 * Each server runs on its own port. Press Ctrl+C to stop all.
 */

import { fork, type ChildProcess } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const servers = [
  { file: 'gemini-image.ts',     port: 18080, name: 'Gemini Image/Video' },
  { file: 'gemini-chat.ts',      port: 18180, name: 'Gemini Chat' },
  { file: 'openai-video.ts',     port: 18100, name: 'OpenAI Video' },
  { file: 'openai-image.ts',     port: 18150, name: 'OpenAI Image' },
  { file: 'together.ts',         port: 18200, name: 'Together AI' },
  { file: 'cloudflare-image.ts', port: 18300, name: 'Cloudflare Image' },
  { file: 'cloudflare-text.ts',  port: 18350, name: 'Cloudflare Text' },
  { file: 'replicate.ts',        port: 18400, name: 'Replicate' },
  { file: 'openai-chat.ts',      port: 18500, name: 'OpenAI Chat (generic)' },
  { file: 'xai-video.ts',        port: 18600, name: 'xAI Video (Grok Imagine)' },
  { file: 'xai-tts.ts',          port: 18650, name: 'xAI TTS (Grok TTS)' },
  { file: 'xai-stt.ts',          port: 18700, name: 'xAI STT (Grok STT)' },
];

const children: ChildProcess[] = [];

console.log('Starting all mock API servers...\n');

for (const s of servers) {
  const child = fork(path.join(__dirname, s.file), [], {
    execArgv: ['--import', 'tsx'],
    stdio: 'inherit',
  });
  children.push(child);
  console.log(`  ${s.name.padEnd(20)} -> http://localhost:${s.port}`);
}

console.log(`\n  /_mock/log available on every server`);
console.log(`  MOCK_ERROR:<code> in prompt for error injection\n`);

process.on('SIGINT', () => {
  console.log('\nShutting down all servers...');
  for (const c of children) c.kill();
  process.exit(0);
});

process.on('SIGTERM', () => {
  for (const c of children) c.kill();
  process.exit(0);
});
