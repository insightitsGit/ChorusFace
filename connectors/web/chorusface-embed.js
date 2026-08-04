/**
 * Embed the ChorusFace MJPEG stream in a host page (Insightits, etc.).
 *
 *   <div id="chorusfaceEmbed"></div>
 *   <script src="/js/chorusface-embed.js"></script>
 *   <script>
 *     window.CHORUSFACE_EMBED = {
 *       streamUrl: 'http://127.0.0.1:8766/stream.mjpg?token=chorusface-beta',
 *       mount: '#chorusfaceEmbed',
 *     };
 *     ChorusFaceEmbed.mount();
 *   </script>
 */
(function (global) {
  'use strict';

  const DEFAULTS = {
    streamUrl: 'http://127.0.0.1:8766/stream.mjpg?token=chorusface-beta',
    mount: '#chorusfaceEmbed',
    alt: 'ChorusFace',
    width: 320,
    height: 320,
    className: 'chorusface-embed-img',
  };

  function config() {
    return Object.assign({}, DEFAULTS, global.CHORUSFACE_EMBED || {});
  }

  function mount(opts) {
    const cfg = Object.assign({}, config(), opts || {});
    const el =
      typeof cfg.mount === 'string'
        ? document.querySelector(cfg.mount)
        : cfg.mount;
    if (!el) {
      if (global.console && console.warn) {
        console.warn('[chorusface-embed] mount not found:', cfg.mount);
      }
      return null;
    }
    let img = el.querySelector('img.' + cfg.className);
    if (!img) {
      img = document.createElement('img');
      img.className = cfg.className;
      el.appendChild(img);
    }
    img.alt = cfg.alt;
    img.width = cfg.width;
    img.height = cfg.height;
    img.src = cfg.streamUrl;
    img.decoding = 'async';
    img.setAttribute('referrerpolicy', 'no-referrer');
    return img;
  }

  function unmount(opts) {
    const cfg = Object.assign({}, config(), opts || {});
    const el =
      typeof cfg.mount === 'string'
        ? document.querySelector(cfg.mount)
        : cfg.mount;
    if (!el) return;
    el.querySelectorAll('img.' + cfg.className).forEach(function (n) {
      n.remove();
    });
  }

  global.ChorusFaceEmbed = { mount: mount, unmount: unmount, config: config };
})(typeof window !== 'undefined' ? window : globalThis);
