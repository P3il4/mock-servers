"use strict";
/**
 * Smoke tests for all mock servers.
 * Verifies each server is running and responds correctly.
 *
 *   npx tsx src/smoke-test.ts
 *
 * Exit code 0 = all passed, 1 = failures.
 */
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var _a, _b;
Object.defineProperty(exports, "__esModule", { value: true });
var SERVERS = [
    { name: 'Gemini Image/Video', port: 18080, tests: [
            { method: 'POST', path: '/v1beta/models/gemini-2.5-flash:generateContent',
                body: { contents: [{ role: 'user', parts: [{ text: 'hello' }] }] },
                expect: function (r) { var _a, _b, _c, _d, _e; return (_e = (_d = (_c = (_b = (_a = r.candidates) === null || _a === void 0 ? void 0 : _a[0]) === null || _b === void 0 ? void 0 : _b.content) === null || _c === void 0 ? void 0 : _c.parts) === null || _d === void 0 ? void 0 : _d[0]) === null || _e === void 0 ? void 0 : _e.text; } },
            { method: 'GET', path: '/_mock/log?last=1',
                expect: function (r) { return Array.isArray(r); } },
        ] },
    { name: 'Gemini Chat', port: 18180, tests: [
            { method: 'POST', path: '/v1beta/models/gemini-2.5-flash:generateContent',
                body: { contents: [{ role: 'user', parts: [{ text: 'hello' }] }] },
                expect: function (r) { var _a, _b, _c, _d, _e; return (_e = (_d = (_c = (_b = (_a = r.candidates) === null || _a === void 0 ? void 0 : _a[0]) === null || _b === void 0 ? void 0 : _b.content) === null || _c === void 0 ? void 0 : _c.parts) === null || _d === void 0 ? void 0 : _d[0]) === null || _e === void 0 ? void 0 : _e.text; } },
            { method: 'GET', path: '/_mock/log?last=1',
                expect: function (r) { return Array.isArray(r); } },
        ] },
    { name: 'OpenAI Video', port: 18100, tests: [
            { method: 'POST', path: '/v1/videos',
                headers: { Authorization: 'Bearer mock' },
                body: { prompt: 'a cat', model: 'sora-2', seconds: '4', size: '1280x720' },
                expect: function (r) { return r.id && r.status === 'queued'; } },
            { method: 'GET', path: '/_mock/log?last=1',
                expect: function (r) { return Array.isArray(r); } },
        ] },
    { name: 'OpenAI Image', port: 18150, tests: [
            { method: 'POST', path: '/v1/images/generations',
                headers: { Authorization: 'Bearer mock' },
                body: { prompt: 'a cat', model: 'dall-e-3', size: '1024x1024' },
                expect: function (r) { var _a; return (_a = r.data) === null || _a === void 0 ? void 0 : _a[0]; } },
            { method: 'GET', path: '/_mock/log?last=1',
                expect: function (r) { return Array.isArray(r); } },
        ] },
    { name: 'Together AI', port: 18200, tests: [
            { method: 'POST', path: '/v2/videos',
                headers: { Authorization: 'Bearer mock' },
                body: { model: 'Wan-AI/Wan2.1-T2V-14B', prompt: 'a cat' },
                expect: function (r) { return r.id; } },
            { method: 'GET', path: '/_mock/log?last=1',
                expect: function (r) { return Array.isArray(r); } },
        ] },
    { name: 'Cloudflare Image', port: 18300, tests: [
            { method: 'POST', path: '/client/v4/accounts/test/ai/run/@cf/black-forest-labs/flux-1-schnell',
                headers: { Authorization: 'Bearer mock' },
                body: { prompt: 'a cat' },
                expect: function (r) { var _a; return r.success === true && ((_a = r.result) === null || _a === void 0 ? void 0 : _a.image); } },
            { method: 'GET', path: '/_mock/log?last=1',
                expect: function (r) { return Array.isArray(r); } },
        ] },
    { name: 'Cloudflare Text', port: 18350, tests: [
            { method: 'POST', path: '/client/v4/accounts/test/ai/run/@cf/meta/llama-3.1-8b-instruct',
                headers: { Authorization: 'Bearer mock' },
                body: { messages: [{ role: 'user', content: 'hello' }] },
                expect: function (r) { var _a; return r.success === true && ((_a = r.result) === null || _a === void 0 ? void 0 : _a.response); } },
            { method: 'GET', path: '/_mock/log?last=1',
                expect: function (r) { return Array.isArray(r); } },
        ] },
    // Error injection tests
    { name: 'OpenAI Video (error)', port: 18100, tests: [
            { method: 'POST', path: '/v1/videos?mock_error=rate_limit',
                headers: { Authorization: 'Bearer mock' },
                body: { prompt: 'test' },
                expectStatus: 429,
                expect: function (r) { var _a; return ((_a = r.error) === null || _a === void 0 ? void 0 : _a.code) === 'rate_limit_exceeded'; } },
        ] },
    { name: 'Cloudflare Image (error)', port: 18300, tests: [
            { method: 'POST', path: '/client/v4/accounts/test/ai/run/@cf/black-forest-labs/flux-1-schnell?mock_error=quota',
                headers: { Authorization: 'Bearer mock' },
                body: { prompt: 'test' },
                expectStatus: 429,
                expect: function (r) { var _a, _b; return ((_b = (_a = r.errors) === null || _a === void 0 ? void 0 : _a[0]) === null || _b === void 0 ? void 0 : _b.code) === 3036; } },
        ] },
];
var passed = 0;
var failed = 0;
for (var _i = 0, SERVERS_1 = SERVERS; _i < SERVERS_1.length; _i++) {
    var server = SERVERS_1[_i];
    for (var _c = 0, _d = server.tests; _c < _d.length; _c++) {
        var test = _d[_c];
        var url = "http://localhost:".concat(server.port).concat(test.path);
        try {
            var opts = {
                method: test.method,
                headers: __assign({ 'Content-Type': 'application/json' }, ((_a = test.headers) !== null && _a !== void 0 ? _a : {})),
            };
            if (test.body)
                opts.body = JSON.stringify(test.body);
            var resp = await fetch(url, opts);
            var expectedStatus = (_b = test.expectStatus) !== null && _b !== void 0 ? _b : 200;
            if (resp.status !== expectedStatus) {
                console.log("  FAIL  ".concat(server.name, " ").concat(test.method, " ").concat(test.path, " \u2014 status ").concat(resp.status, " (expected ").concat(expectedStatus, ")"));
                failed++;
                continue;
            }
            var json = await resp.json();
            if (test.expect(json)) {
                console.log("  PASS  ".concat(server.name, " ").concat(test.method, " ").concat(test.path));
                passed++;
            }
            else {
                console.log("  FAIL  ".concat(server.name, " ").concat(test.method, " ").concat(test.path, " \u2014 unexpected response"));
                console.log("        ".concat(JSON.stringify(json).slice(0, 200)));
                failed++;
            }
        }
        catch (e) {
            console.log("  FAIL  ".concat(server.name, " ").concat(test.method, " ").concat(test.path, " \u2014 ").concat(e.message));
            failed++;
        }
    }
}
console.log("\n".concat(passed, " passed, ").concat(failed, " failed"));
process.exit(failed > 0 ? 1 : 0);
