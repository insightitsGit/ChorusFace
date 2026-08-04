#!/usr/bin/env bash
# Container entrypoint for ChorusFace FaceBridge.
# Set CHORUSFACE_USE_XVFB=1 for Docker Desktop / no-GPU hosts (software GL via Xvfb).
#
# PID 1 must not be xvfb-run alone (SIGUSR1 readiness hang). Prefer tini as PID 1
# (Dockerfile ENTRYPOINT), then this script → xvfb-run → python.
set -euo pipefail

if [[ "${CHORUSFACE_USE_XVFB:-0}" == "1" ]]; then
  # Compose stacks sometimes leave MODERNGL_BACKEND=egl from Linux/GPU defaults.
  # Xvfb + pyglet need GLX, not EGL.
  unset MODERNGL_BACKEND || true
  export MODERNGL_WINDOW="${MODERNGL_WINDOW:-pyglet}"
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
  # FieldRuntime requests GL 4.3. A 3.3 override makes pyglet context creation fail.
  # Pin 4.5 so llvmpipe advertises a usable core profile under Xvfb.
  export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-4.5}"

  if [[ -e /usr/lib/x86_64-linux-gnu/libGL.so.1 && ! -e /usr/lib/x86_64-linux-gnu/libGL.so ]]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
  fi
  if [[ -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 && ! -e /usr/lib/x86_64-linux-gnu/libEGL.so ]]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libEGL.so
  fi
  ldconfig >/dev/null 2>&1 || true

  # With tini (or any non-xvfb-run PID 1), xvfb-run's SIGUSR1 handshake works.
  exec xvfb-run -a -e /tmp/chorusface-xvfb.log -s '-screen 0 1024x1024x24' -- "$@"
fi

exec "$@"
