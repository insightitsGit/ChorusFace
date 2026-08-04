/**
 * Browser connector for local/http demos → ChorusFace FaceBridge /speak.
 *
 * Prefer server-side speak for HTTPS hosts (mixed content blocks localhost).
 * Env-style config via window.CHORUSFACE_BRIDGE = { url, token, enabled }.
 */
(function (global) {
  'use strict';

  const DEFAULTS = {
    url: 'http://127.0.0.1:8766',
    token: 'chorusface-beta',
    enabled: true,
    timeoutMs: 2500,
  };

  function config() {
    const cfg = Object.assign({}, DEFAULTS, global.CHORUSFACE_BRIDGE || {});
    cfg.url = String(cfg.url || DEFAULTS.url).replace(/\/$/, '');
    cfg.token = String(cfg.token || DEFAULTS.token);
    cfg.enabled = cfg.enabled !== false;
    cfg.timeoutMs = Number(cfg.timeoutMs || DEFAULTS.timeoutMs);
    return cfg;
  }

  function stripMarkdownLite(text) {
    return String(text || '')
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/`[^`]*`/g, ' ')
      .replace(/!\[[^\]]*]\([^)]*\)/g, ' ')
      .replace(/\[[^\]]*]\([^)]*\)/g, '$1')
      .replace(/[#>*_~]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  /**
   * @param {string} text
   * @param {object} [opts]
   * @returns {Promise<{ok:boolean, queued?:boolean, error?:string}>}
   */
  async function speak(text, opts) {
    const cfg = Object.assign({}, config(), opts || {});
    if (!cfg.enabled) {
      return { ok: false, error: 'disabled' };
    }
    const spoken = stripMarkdownLite(text);
    if (!spoken) {
      return { ok: false, error: 'empty text' };
    }
    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timer = controller
      ? setTimeout(function () {
          controller.abort();
        }, cfg.timeoutMs)
      : null;
    try {
      const response = await fetch(cfg.url + '/speak', {
        method: 'POST',
        headers: {
          Authorization: 'Bearer ' + cfg.token,
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({ text: spoken }),
        signal: controller ? controller.signal : undefined,
      });
      if (timer) clearTimeout(timer);
      const body = await response.json().catch(function () {
        return {};
      });
      if (!response.ok) {
        return { ok: false, error: body.error || 'HTTP ' + response.status };
      }
      return { ok: true, queued: !!body.queued, text: body.text || spoken };
    } catch (err) {
      if (timer) clearTimeout(timer);
      return { ok: false, error: String((err && err.message) || err) };
    }
  }

  function speakFireAndForget(text, opts) {
    speak(text, opts).then(function (result) {
      if (!result.ok && global.console && console.warn) {
        console.warn('[chorusface-bridge] speak failed:', result.error);
      }
    });
  }

  global.ChorusFaceBridge = {
    speak: speak,
    speakFireAndForget: speakFireAndForget,
    stripMarkdownLite: stripMarkdownLite,
    config: config,
  };
})(typeof window !== 'undefined' ? window : globalThis);
