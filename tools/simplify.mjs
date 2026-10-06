// meshoptimizer simplification for the tree pipeline (called from tools/blender/build_tree.py).
// Usage: node tools/simplify.mjs <in.bin> <out.bin> <ratio> <relativeError>
// in.bin: u32 vertexCount, u32 indexCount, f32 positions[vertexCount*3], u32 indices[indexCount]
// out.bin: u32 indices of the simplified mesh (referencing the original vertices).
import { readFileSync, writeFileSync } from "node:fs";
import { MeshoptSimplifier as S } from "meshoptimizer";

const [inPath, outPath, ratioArg, errArg] = process.argv.slice(2);
await S.ready;
const buf = readFileSync(inPath);
const head = new Uint32Array(buf.buffer, buf.byteOffset, 2);
const [nv, ni] = head;
const pos = new Float32Array(buf.buffer.slice(buf.byteOffset + 8, buf.byteOffset + 8 + nv * 12));
const idx = new Uint32Array(buf.buffer.slice(buf.byteOffset + 8 + nv * 12, buf.byteOffset + 8 + nv * 12 + ni * 4));
const target = Math.max(3, Math.floor((ni * Number(ratioArg)) / 3) * 3);
const err = Number(errArg);
let [out, e] = S.simplify(idx, pos, 3, target, err, ["Prune"]);
let mode = "simplify";
if (out.length > target * 1.5) {
  // Topology-preserving simplification stalled (open twig tubes); finish with sloppy.
  [out, e] = S.simplifySloppy(out, pos, 3, null, target, err * 4);
  mode = "sloppy";
}
writeFileSync(outPath, Buffer.from(out.buffer, out.byteOffset, out.byteLength));
console.log(JSON.stringify({ mode, from: ni / 3, to: out.length / 3, target: target / 3, error: e }));
