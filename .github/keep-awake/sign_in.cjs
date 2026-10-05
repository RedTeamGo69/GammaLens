// Run interactively on the owner's computer. Passwords and identity-provider
// cookies stay in this temporary browser; only app-host cookies are retained.
const { chromium } = require('playwright');
const { mkdirSync, writeFileSync } = require('node:fs');
const { join } = require('node:path');
const { appSession } = require('./session.cjs');

(async () => {
  const appUrl = new URL(process.env.STREAMLIT_APP_URL || 'https://gex-q-dashboard.streamlit.app/');
  if (appUrl.protocol !== 'https:' || !appUrl.hostname.endsWith('.streamlit.app')) {
    throw new Error('Use an HTTPS Streamlit Community Cloud URL.');
  }
  appUrl.search = '?tab=forward';
  appUrl.hash = '';
  const browser = await chromium.launch({ headless: false });
  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(appUrl.href, { waitUntil: 'domcontentloaded', timeout: 60_000 });
    console.log('Sign in to Gamma Lens in the browser window. Waiting up to 15 minutes.');
    const deadline = Date.now() + 15 * 60_000;
    while (Date.now() < deadline) {
      if (!browser.isConnected() || context.pages().length === 0) {
        throw new Error('The sign-in browser closed before Gamma Lens finished loading. No session was saved.');
      }
      // OAuth can finish in a new tab rather than the original login page.
      for (const frame of context.pages().flatMap(tab => tab.frames())) {
        if (new URL(frame.url()).hostname !== appUrl.hostname) continue;
        if (await frame.getByRole('button', { name: 'Download Excel results', exact: true })
            .isVisible().catch(() => false)) {
          const state = appSession(await context.storageState(), appUrl.hostname);
          if (!process.env.LOCALAPPDATA) throw new Error('Run this helper on Windows.');
          const folder = join(process.env.LOCALAPPDATA, 'GammaLens', 'keep-awake');
          mkdirSync(folder, { recursive: true });
          const path = join(folder, 'viewer-session.json');
          writeFileSync(path, JSON.stringify(state), { mode: 0o600 });
          console.log(`Saved ${state.cookies.length} app-scoped cookies to ${path}`);
          console.log('No Google, GitHub, or Streamlit workspace cookies were saved.');
          return;
        }
        const wake = frame.getByRole('button', { name: /Yes, get this app back up!/i });
        if (await wake.isVisible().catch(() => false)) {
          await wake.click({ timeout: 10_000 }).catch(() => {});
        }
      }
      // Do not anchor the wait to the original tab: the login redirect may
      // close it while a new sign-in tab remains open in this context.
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    throw new Error('Sign-in did not finish. No session was saved.');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error.message.includes('has been closed')
    ? 'The sign-in browser closed before Gamma Lens finished loading. No session was saved.'
    : error.message);
  process.exitCode = 1;
});
