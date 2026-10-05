const test = require('node:test');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const { WebSocketServer } = require('ws');
const { createServer } = require('node:http');

process.env.STREAMLIT_APP_URL = 'https://gex-q-dashboard.streamlit.app/';
delete process.env.STREAMLIT_VIEWER_STATE;
// Synthetic checks must never write a production availability summary.
delete process.env.GITHUB_STEP_SUMMARY;
const { visit } = require('./visit.cjs');
const results = '<h1>Forward Test</h1><button>Download Excel results</button>';
const connect = 'new WebSocket("SOCKET_URL");';

// Real Chromium loads a controlled page and real browser WebSocket API.
// Only network responses are stubbed; no login credentials or production data.
async function withPage(html, action) {
  const server = createServer((request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/html' });
    response.end(html);
  });
  const sockets = new WebSocketServer({ server });
  sockets.on('connection', socket => socket.send('ready'));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  html = html.replace('SOCKET_URL', `ws://127.0.0.1:${server.address().port}/_stcore/stream`);
  const browser = await chromium.launch({ headless: true });
  try {
    const configuredBrowser = {
      async newContext(options) {
        const context = await browser.newContext(options);
        await context.route('https://gex-q-dashboard.streamlit.app/**', route =>
          route.fulfill({ status: 302, headers: { location: `http://127.0.0.1:${server.address().port}/` } }));
        return context;
      },
    };
    await action(configuredBrowser);
  } finally {
    await browser.close();
    for (const socket of sockets.clients) socket.terminate();
    await new Promise(resolve => sockets.close(resolve));
    await new Promise(resolve => server.close(resolve));
  }
}

const fast = { timeoutMs: 2000, holdMs: 50, pollMs: 50 };

test('accept rendered results with a receiving Streamlit connection', async () => {
  await withPage(`${results}<script>${connect}</script>`, browser => visit(browser, fast));
});

test('wake a sleeping page and verify the resulting session', async () => {
  const html = `<button onclick='document.body.innerHTML=${JSON.stringify(results)}; ${connect}'>Yes, get this app back up!</button>`;
  await withPage(html, browser => visit(browser, fast));
});

test('reject an HTTP 200 shell without a Streamlit session', async () => {
  await withPage(results, browser => assert.rejects(visit(browser, fast), /Timed out/));
});

test('reject the private-app sign-in screen', async () => {
  await withPage('<h1>You do not have access to this app or it does not exist</h1>',
    browser => assert.rejects(visit(browser, fast), /App is inaccessible/));
});
