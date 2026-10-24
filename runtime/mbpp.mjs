import { readFileSync } from "node:fs";
import { loadPyodide } from "../node_modules/pyodide/pyodide.mjs";

// Node/npm installs assets; only permission-restricted Deno executes candidates.
const request = JSON.parse(readFileSync(0, "utf8"));
const driver = readFileSync(
  new URL("./mbpp_driver.py", import.meta.url),
  "utf8",
);
const encoder = new TextEncoder();
let remaining = request.config.max_output_bytes;
let truncated = false;
function write(bytes) {
  while (bytes.length) bytes = bytes.subarray(Deno.stdout.writeSync(bytes));
}
function emit(text) {
  const bytes = encoder.encode(text);
  const kept = bytes.subarray(0, remaining);
  write(kept);
  remaining -= kept.length;
  truncated ||= bytes.length > kept.length;
}

console.log = console.warn = console.error = () => {};
try {
  const python = await loadPyodide({
    indexURL: new URL("../node_modules/pyodide/", import.meta.url).pathname,
    fullStdLib: false,
    stdout: () => {},
    stderr: () => {},
  });
  await Deno.permissions.revoke({ name: "read" });
  await Deno.permissions.revoke({ name: "env" });
  globalThis.__mbpp_emit = emit;
  python.globals.set("_request_json", JSON.stringify(request));
  write(encoder.encode(request.token + "\n"));
  const passed = python.runPython(driver);
  if (truncated) {
    write(encoder.encode("\n[Output truncated.]\n"));
  }
  if (passed) write(encoder.encode("\n" + request.token + "\n"));
  Deno.exit(passed ? 0 : 1);
} catch (error) {
  emit(String(error));
  Deno.exit(1);
}
