#!/bin/sh
# Validate every entry shader (and its imports) against a real WebGPU device.
cd "$(dirname "$0")/.." || exit 1
status=0
for f in src/shaders/*.wgsl; do
  out=$(npm exec --no -- vgpu check --require-validation "$f" 2>&1)
  if echo "$out" | grep -q '"ok": true' && ! echo "$out" | grep -q '"severity": "error"'; then
    echo "ok   $f"
  else
    status=1
    echo "FAIL $f"
    echo "$out" | grep -E '"(message|line|file)"' | head -12
  fi
done
exit $status
