"""
PWA / installable-app head injection for Gamma Lens.

Streamlit renders the app inside a host page whose ``<head>`` we don't control
directly. This module injects the PWA manifest, Apple "web-app-capable" meta
tags, an apple-touch-icon, a theme-color, and patches the viewport for
safe-area support by running a tiny same-origin script (via
``st.components.v1.html``) that reaches ``window.parent.document.head``.

Injection is idempotent — guarded by element ids — because Streamlit re-runs
the script on every interaction. The whole thing is wrapped in try/catch so a
sandboxed/cross-origin iframe degrades to a no-op instead of erroring.

Pure presentation — no data/model logic.
"""
from __future__ import annotations

import base64
import json

import streamlit.components.v1 as components

from ui_theme import logo_data_uri

# NOTE: ``st.components.v1.html`` is deprecated in favour of ``st.iframe``, but
# ``st.iframe`` only embeds a *URL* — it can't run our inline script, and a
# ``data:`` URL iframe is cross-origin so ``window.parent`` head access would be
# blocked by the same-origin policy. The component's same-origin ``srcdoc``
# iframe is the only way to reach the host page's <head>, so we keep using it and
# guard the call so a future removal degrades to a harmless no-op.

#: Brand background — matches theme.COLORS / TOKENS["bg_base"] and config.toml.
_THEME_COLOR = "#0a0d13"
_APP_NAME = "Gamma Lens"


def _build_manifest_data_uri(icon_uri: str) -> str:
    """Return a base64 data URI for an inline web-app manifest.

    Inlined (rather than served as a static file) so it works unchanged across
    local dev and Streamlit Cloud without depending on a static path / base URL.
    """
    icons: list[dict[str, str]] = []
    if icon_uri:
        icons = [
            {"src": icon_uri, "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": icon_uri, "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ]
    manifest = {
        "name": _APP_NAME,
        "short_name": _APP_NAME,
        "display": "standalone",
        "orientation": "any",
        "background_color": _THEME_COLOR,
        "theme_color": _THEME_COLOR,
        "start_url": ".",
        "scope": ".",
        "icons": icons,
    }
    raw = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    return "data:application/manifest+json;base64," + base64.b64encode(raw).decode("ascii")


def inject_pwa_head() -> None:
    """Inject PWA manifest + Apple meta tags + icon into the parent document head.

    Idempotent and safe to call once per rerun. Renders a 0-height iframe (hidden
    by ``ui_theme`` CSS) so it adds no visible space. Relies on the component
    iframe being same-origin (true on localhost and Streamlit Cloud).
    """
    icon_uri = logo_data_uri()
    manifest_uri = _build_manifest_data_uri(icon_uri)

    # json.dumps so every value is safely quoted/escaped inside the JS string.
    icon_js = json.dumps(icon_uri)
    manifest_js = json.dumps(manifest_uri)
    theme_js = json.dumps(_THEME_COLOR)
    title_js = json.dumps(_APP_NAME)

    script = f"""
<script>
(function() {{
  try {{
    var doc = window.parent.document;
    if (!doc || !doc.head) return;
    var head = doc.head;

    function setMeta(name, content) {{
      var id = 'gl-meta-' + name;
      var el = doc.getElementById(id);
      if (!el) {{
        el = doc.createElement('meta');
        el.id = id;
        el.setAttribute('name', name);
        head.appendChild(el);
      }}
      el.setAttribute('content', content);
    }}

    function setLink(rel, id, href, extra) {{
      var el = doc.getElementById(id);
      if (!el) {{
        el = doc.createElement('link');
        el.id = id;
        el.setAttribute('rel', rel);
        head.appendChild(el);
      }}
      el.setAttribute('href', href);
      if (extra) {{ Object.keys(extra).forEach(function(k) {{ el.setAttribute(k, extra[k]); }}); }}
    }}

    // Installable / fullscreen when launched from the Home Screen
    setMeta('apple-mobile-web-app-capable', 'yes');
    setMeta('mobile-web-app-capable', 'yes');
    setMeta('apple-mobile-web-app-status-bar-style', 'black-translucent');
    setMeta('apple-mobile-web-app-title', {title_js});
    setMeta('theme-color', {theme_js});

    var iconUri = {icon_js};
    if (iconUri) {{
      setLink('apple-touch-icon', 'gl-apple-touch-icon', iconUri);
      setLink('icon', 'gl-icon', iconUri, {{'type': 'image/png'}});
    }}
    setLink('manifest', 'gl-manifest', {manifest_js});

    // viewport-fit=cover so env(safe-area-inset-*) applies on notched iPhones
    var vp = doc.querySelector('meta[name="viewport"]');
    if (vp) {{
      var c = vp.getAttribute('content') || '';
      if (c.indexOf('viewport-fit') === -1) {{
        vp.setAttribute('content', c + (c ? ', ' : '') + 'viewport-fit=cover');
      }}
    }} else {{
      setMeta('viewport', 'width=device-width, initial-scale=1, viewport-fit=cover');
    }}
  }} catch (e) {{ /* cross-origin or sandboxed iframe — non-fatal no-op */ }}
}})();
</script>
"""
    try:
        components.html(script, height=0, width=0)
    except Exception:
        # PWA head injection is a progressive enhancement — if the component API
        # is ever removed, the app still works without the installable polish.
        pass
