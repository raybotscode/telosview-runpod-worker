// gpu/context.js — one WebGPU device for the whole pipeline.
//
// The trainer and the SIFT matcher share it (two devices double descriptor
// VRAM and can never share buffers), and a host that already owns a device —
// e.g. a PlayCanvas app — can hand it in instead.

/**
 * @typedef {object} GpuContext
 * @property {GPUDevice} device
 * @property {GPUAdapter|null} adapter   null when the device was handed in
 * @property {boolean} owned             whether dispose() destroys the device
 * @property {() => void} dispose
 */

/**
 * @param {{ device?: GPUDevice, powerPreference?: GPUPowerPreference }} [opts]
 * @returns {Promise<GpuContext>}
 */
export async function createGpu(opts = {}) {
  if (opts.device) {
    return watchLost({ device: opts.device, adapter: null, info: opts.info || {}, owned: false, dispose() {} });
  }
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('WebGPU not available in this environment');
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: opts.powerPreference || 'low-power',
  });
  if (!adapter) throw new Error('no WebGPU adapter');
  // full-res training-target buffers can exceed the 128MB default binding
  // limit — ask for up to 1GB where the adapter allows it
  const want = 1 << 30;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, want),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, want),
    },
  });
  const info = adapter.info || {};
  return watchLost({ device, adapter, info, owned: true, dispose() { device.destroy(); } });
}

/** Surface real device loss (iOS reclaims WebGPU devices from backgrounded
 *  tabs; drivers reset). An intentional dispose() also settles device.lost,
 *  with reason 'destroyed' — that one is not a loss. */
function watchLost(ctx) {
  ctx.lost = false;
  if (ctx.device.lost && typeof ctx.device.lost.then === 'function') {
    ctx.device.lost.then((info) => {
      if (info && info.reason === 'destroyed') return;
      ctx.lost = true;
      if (ctx.onLost) ctx.onLost(info);
    }).catch(() => {});
  }
  return ctx;
}
