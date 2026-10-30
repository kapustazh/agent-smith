import { readFileSync } from "node:fs";
import { loadPyodide } from "../node_modules/pyodide/pyodide.mjs";

const request = JSON.parse(readFileSync(0, "utf8"));
const driver = readFileSync(
  new URL("./mbpp_driver.py", import.meta.url),
  "utf8",
);
const encoder = new TextEncoder();
const writeOutput = Deno.stdout.writeSync.bind(Deno.stdout);
const exit = Deno.exit.bind(Deno);
let remaining = request.config.max_output_bytes;
let truncated = false;
function write(bytes) {
  while (bytes.length) bytes = bytes.subarray(writeOutput(bytes));
}
function emit(text) {
  const bytes = encoder.encode(text);
  const kept = bytes.subarray(0, remaining);
  write(kept);
  remaining -= kept.length;
  if (bytes.length > kept.length && !truncated) {
    truncated = true;
    write(encoder.encode("\n[Output truncated.]\n"));
  }
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
  python.globals.set("_config_json", JSON.stringify(request.config));
  python.globals.set("_emit", emit);
  const runtime = python.runPython(driver);
  python.globals.delete("_config_json");
  python.globals.delete("_emit");
  write(encoder.encode(request.token + "\n"));
  const imports = (request.test_imports ?? []).join("\n");
  let passed = !imports || runtime.execute(imports, "<test imports>");
  if (passed) passed = runtime.execute(request.code, "<solution>");
  for (let index = 0; passed && index < request.test_list.length; index++) {
    passed = runtime.execute(request.test_list[index], `<test ${index + 1}>`);
  }
  runtime.destroy();
  if (passed) write(encoder.encode("\n" + request.token + "\n"));
  exit(passed ? 0 : 1);
} catch (error) {
  emit(String(error));
  exit(1);
}
