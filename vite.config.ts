import { defineConfig } from "vite";
import { wgslVitePlugin } from "@vgpu/wgsl/loader-vite";

export default defineConfig({
  // Relative base: the build runs from any path (GitHub Pages serves it under /<repo>/).
  base: "./",
  plugins: [wgslVitePlugin()],
  build: { target: "es2022" },
});
