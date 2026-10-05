// An HTTP 200 can be the Cloud sleep/login page. Open a real Streamlit
// session and require rendered results plus an active, receiving websocket.
const { chromium } = require('playwright');
const { appendFileSync } = require('node:fs');
const { appSession } = require('./session.cjs');

const configuredUrl = process.env.STREAMLIT_APP_URL;
if (!configuredUrl) throw new Error('Set STREAMLIT_APP_URL to the public Gamma Lens URL.');
const appUrl = new URL(configuredUrl);
if (appUrl.protocol !== 'https:' || !appUrl.hostname.endsWith('.streamlit.app')) {
  throw new Error('STREAMLIT_APP_URL must be an HTTPS Streamlit Community Cloud URL.');
}
// This route only reads stored results; opening GEX can fetch quotes and
// self-heal snapshots. The keep-awake visit must not perform that work.
appUrl.search = '?tab=forward';
appUrl.hash = '';

async function visit(browser, { timeoutMs = 180_000, holdMs = 15_000, pollMs = 1000 } = {}) {
  let storageState;
  if (process.env.STREAMLIT_VIEWER_STATE) {
    let state;
    try {
      state = JSON.parse(process.env.STREAMLIT_VIEWER_STATE);
    } catch {
      // JSON parser messages can quote input. Never log credential contents.
      throw new Error('STREAMLIT_VIEWER_STATE is invalid JSON. Replace the saved session secret.');
    }
    storageState = appSession(state, appUrl.hostname);
  }
  const context = await browser.newContext({ storageState });
  const page = await context.newPage();
  const streams = [];
  page.on('websocket', socket => {
    if (!new URL(socket.url()).pathname.endsWith('/_stcore/stream')) return;
    const stream = { socket, received: false };
    streams.push(stream);
    socket.on('framereceived', () => { stream.received = true; });
  });
  try {
    await page.goto(appUrl.href, { waitUntil: 'domcontentloaded', timeout: 60_000 });
    const deadline = Date.now() + timeoutMs;
    let woke = false;
    while (Date.now() < deadline) {
      for (const frame of page.frames()) {
        if (await frame.getByText(/You do not have access to this app or it does not exist/i).count()) {
          throw new Error('App is inaccessible. Renew STREAMLIT_VIEWER_STATE using the local sign-in helper, or check the URL and viewer access.');
        }
        const wake = frame.getByRole('button', { name: /Yes, get this app back up!/i });
        if (!woke && await wake.isVisible()) {
          await wake.click({ timeout: 10_000 });
          woke = true;
          console.log('Requested wake-up from the Community Cloud sleep page.');
        }
        const heading = frame.getByRole('heading', { name: 'Forward Test', exact: true });
        const results = frame.getByRole('button', { name: 'Download Excel results', exact: true });
        if (await heading.isVisible() && await results.isVisible()
            && streams.some(s => s.received && !s.socket.isClosed())) {
          // Keep the established session open briefly, then confirm it did not
          // disappear during startup. This is availability, not a model audit.
          await page.waitForTimeout(holdMs);
          if (!await results.isVisible() || await frame.locator('[data-testid="stException"]').count()
              || !streams.some(s => s.received && !s.socket.isClosed())) {
            throw new Error('Streamlit session failed after rendering.');
          }
          const message = `Gamma Lens loaded at ${new Date().toISOString()} with an active Streamlit session${woke ? ' after waking' : ''}.`;
          console.log(message);
          if (process.env.GITHUB_STEP_SUMMARY) {
            appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${message}\n`);
          }
          return;
        }
      }
      await page.waitForTimeout(pollMs);
    }
    throw new Error('Timed out waiting for Gamma Lens results and an active Streamlit session.');
  } finally {
    await context.close();
  }
}

module.exports = { visit };

if (require.main === module) {
(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        await visit(browser);
        return;
      } catch (error) {
        console.error(`Visit ${attempt}/3 failed: ${error.message}`);
        if (attempt === 3) throw error;
        await new Promise(resolve => setTimeout(resolve, 10_000));
      }
    }
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
}
