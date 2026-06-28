"""
Mobile "app-shell" enhancements for Gamma Lens (Part 2).

Progressive enhancement injected the same way as ui_pwa: a same-origin
``components.html`` script that operates on ``window.parent.document``. It:
  1. publishes the sticky header's height as the ``--gl-header-h`` CSS variable
     (so the settings drop-down sheet anchors just beneath the header),
  2. wires the header gear button to toggle the settings sheet
     (``body.gl-settings-open``), closing on an outside / backdrop tap,
  3. turns each ``.gl-pager`` into a swipe carousel: syncs the ``.gl-dot``
     indicators to the scroll position and scrolls to a card when its dot is
     tapped.

All of this is inert on desktop — the gear and dots are ``display:none`` and
``.gl-track`` is ``display:contents``, so there is nothing to scroll or click.
It re-applies safely across Streamlit reruns: the document-level click handler,
resize handler and MutationObserver are each installed once (guarded by flags on
the parent ``window``), and per-pager wiring is guarded by a ``data-gl-init``
attribute. The observer is what re-wires freshly rendered pagers when Streamlit
skips re-running this (identical) component. Wrapped in try/catch so a sandboxed
iframe degrades to the Part-1 stacked layout.

Pure presentation — no data/model logic.
"""
from __future__ import annotations

import streamlit.components.v1 as components

_SCRIPT = """
<script>
(function () {
  try {
    var doc = window.parent.document;
    if (!doc || !doc.body) return;
    var win = doc.defaultView || window.parent;

    // 1) publish the sticky-header height so the settings sheet anchors under it
    function syncHeaderHeight() {
      var h = doc.querySelector('.term-header');
      if (h) doc.documentElement.style.setProperty('--gl-header-h', h.offsetHeight + 'px');
    }

    // 2) settings sheet toggle + outside-tap-to-close (installed once)
    if (!win.__glShellWired) {
      win.__glShellWired = true;
      doc.addEventListener('click', function (e) {
        var t = e.target;
        var toggle = t && t.closest ? t.closest('.gl-settings-toggle') : null;
        if (toggle) {
          e.preventDefault();
          // guard against double-fire (touch + synthesized click) toggling twice
          var now = Date.now();
          if (now - (win.__glLastToggle || 0) < 300) return;
          win.__glLastToggle = now;
          doc.body.classList.toggle('gl-settings-open');
          return;
        }
        if (doc.body.classList.contains('gl-settings-open')) {
          var inside = t && t.closest ? t.closest('.st-key-settings_card') : null;
          if (!inside) doc.body.classList.remove('gl-settings-open');
        }
      }, true);
      win.addEventListener('resize', syncHeaderHeight);
    }

    // 3) carousel — sync dots to scroll position; tap a dot to scroll to a card
    function activeIndex(track) {
      var slides = track.children, c = track.scrollLeft + track.clientWidth / 2;
      var best = 0, bestD = Infinity;
      for (var i = 0; i < slides.length; i++) {
        var s = slides[i], center = s.offsetLeft + s.offsetWidth / 2, d = Math.abs(center - c);
        if (d < bestD) { bestD = d; best = i; }
      }
      return best;
    }
    function wirePager(pager) {
      if (pager.getAttribute('data-gl-init') === '1') return;
      pager.setAttribute('data-gl-init', '1');
      var track = pager.querySelector('.gl-track');
      var dots = pager.querySelectorAll('.gl-dot');
      if (!track || !dots.length) return;
      function update() {
        var idx = activeIndex(track);
        for (var i = 0; i < dots.length; i++) dots[i].classList.toggle('active', i === idx);
      }
      track.addEventListener('scroll', function () { win.requestAnimationFrame(update); }, { passive: true });
      for (var i = 0; i < dots.length; i++) {
        (function (j) {
          dots[j].addEventListener('click', function () {
            var s = track.children[j];
            if (s) s.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
          });
        })(i);
      }
      update();
    }
    function wireAll() {
      syncHeaderHeight();
      var pagers = doc.querySelectorAll('.gl-pager');
      for (var i = 0; i < pagers.length; i++) wirePager(pagers[i]);
    }

    wireAll();

    // re-wire after Streamlit re-renders the pager / header (debounced to a frame)
    if (!win.__glShellObserver) {
      var pending = false;
      win.__glShellObserver = new win.MutationObserver(function () {
        if (pending) return;
        pending = true;
        win.requestAnimationFrame(function () { pending = false; wireAll(); });
      });
      win.__glShellObserver.observe(doc.body, { childList: true, subtree: true });
    }
  } catch (e) { /* sandboxed/cross-origin — falls back to the stacked layout */ }
})();
</script>
"""


def inject_mobile_shell() -> None:
    """Inject the mobile app-shell behavior (carousel dots + settings sheet)."""
    try:
        components.html(_SCRIPT, height=0, width=0)
    except Exception:
        # Progressive enhancement — the app still works without it.
        pass
