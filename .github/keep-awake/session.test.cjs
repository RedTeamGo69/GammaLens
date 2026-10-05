const test = require('node:test');
const assert = require('node:assert/strict');
const { appSession } = require('./session.cjs');

const hostname = 'gex-q-dashboard.streamlit.app';
const cookie = (domain, expires = -1) => ({
  name: 'session', value: 'test-only', domain, path: '/', expires,
  httpOnly: true, secure: true, sameSite: 'Lax',
});

test('retain app-host cookies while discarding identity and workspace access', () => {
  const own = cookie(hostname);
  const state = appSession({
    cookies: [own, cookie('.streamlit.app'), cookie('share.streamlit.io'),
      cookie('.google.com'), cookie('.github.com'), cookie('another.streamlit.app')],
    origins: [{ origin: 'https://accounts.google.com', localStorage: [{ name: 'token', value: 'test' }] }],
  }, hostname);
  assert.deepEqual(state, { cookies: [own], origins: [] });
});

test('allow leading-dot exact app domains', () => {
  assert.equal(appSession({ cookies: [cookie(`.${hostname}`)] }, hostname).cookies.length, 1);
});

test('drop expired credentials and require a current app-scoped cookie', () => {
  assert.throws(() => appSession({ cookies: [cookie(hostname, Date.now() / 1000 - 60)] }, hostname),
    /No current app-scoped cookies/);
  assert.throws(() => appSession({ cookies: [cookie('.streamlit.app')] }, hostname),
    /No current app-scoped cookies/);
});
