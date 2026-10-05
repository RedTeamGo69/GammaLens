// Only app-host cookies may be sent to Actions. Never retain Google, GitHub,
// share.streamlit.io, or wildcard *.streamlit.app authentication state.
function appSession(state, hostname) {
  const cookies = (state.cookies || []).filter(cookie =>
    cookie.domain.replace(/^\./, '') === hostname
    && (cookie.expires === -1 || cookie.expires > Date.now() / 1000));
  if (!cookies.length) {
    throw new Error('No current app-scoped cookies found. Sign in to Gamma Lens again.');
  }
  return { cookies, origins: [] };
}

module.exports = { appSession };
