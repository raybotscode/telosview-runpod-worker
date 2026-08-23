// Native Dawn WebGPU test — validates that native Dawn (via the `webgpu`
// package) reaches the NVIDIA adapter through Vulkan directly, with NO
// Chromium browser checks (SupportsExternalImages / BackendType::Null).
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);
const gpu = create(['backend=vulkan']);

const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) {
  console.log('NATIVE_RESULT: NO_ADAPTER');
  process.exit(1);
}
const info = adapter.info || {};
console.log(`NATIVE_RESULT: vendor=${info.vendor} arch=${info.architecture} desc=${info.description}`);

const device = await adapter.requestDevice();

// Minimal compute-shader round-trip: proves adapter -> device -> compute works.
const shader = device.createShaderModule({
  code: `
    @group(0) @binding(0) var<storage, read_write> data: array<u32>;
    @compute @workgroup_size(1) fn main() { data[0] = 42u; }
  `,
});
const pipeline = device.createComputePipeline({
  layout: 'auto',
  compute: { module: shader, entryPoint: 'main' },
});
const buf = device.createBuffer({
  size: 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});
const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: buf } }],
});
const readback = device.createBuffer({
  size: 4,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});

const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(1);
pass.end();
encoder.copyBufferToBuffer(buf, 0, readback, 0, 4);
device.queue.submit([encoder.finish()]);

await readback.mapAsync(GPUMapMode.READ);
const result = new Uint32Array(readback.getMappedRange());
console.log(`NATIVE_RESULT: compute=${result[0]} (expect 42)`);
