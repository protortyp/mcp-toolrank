import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const [checkpoint, url, output] = process.argv.slice(2);
if (!checkpoint || !url || !output) throw new Error("Pass the INT8 checkpoint, demo URL, and output path.");
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? "playwright");
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1024, height: 768 } });
  const requests = [];
  const errors = [];
  const artifactRequests = [];
  page.on("request", (request) => requests.push([request.method(), request.url(), request.postData()]));
  page.on("pageerror", (error) => errors.push(error.message));
  const revision = "a24a121266154006e2bd2e32a41576437c3455c2";
  await page.route(`https://huggingface.co/christianengel/mcp-toolrank-33m/resolve/${revision}/**`, async (route) => {
    const artifact = new URL(route.request().url()).pathname.split(`/${revision}/`)[1];
    artifactRequests.push(artifact);
    await route.fulfill({ path: resolve(checkpoint, artifact), headers: { "access-control-allow-origin": "*" },
      contentType: artifact.endsWith(".onnx") ? "application/octet-stream" : "application/json" });
  });
  await page.goto(url);
  const rank = page.getByRole("button", { name: "Rank tools" });
  assert.ok(await page.getByRole("radio", { name: "BM25" }).isChecked());
  await rank.click();
  await page.getByText(/Numbers show normalized BM25 scores/).waitFor();
  assert.ok(await page.locator("#ranking li").count());
  await page.locator("#query").fill("zzznonlexicaloverlap");
  await rank.click();
  await page.getByText("No indexed terms overlap with the request.").waitFor();
  assert.equal(await page.locator("#ranking li").count(), 0);
  assert.deepEqual(artifactRequests, []);
  assert.ok(requests.every(([, address]) => !address.endsWith(".wasm")));

  await page.getByText("Tool catalog", { exact: true }).click();
  const tool = { name: "private-fixture-test", description: "Find private fixture test records",
    inputSchema: { type: "object", properties: { name: { type: "string" } } } };
  await page.locator("#catalog-file").setInputFiles({ name: "test.json", mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({ tools: [tool] })) });
  await page.getByText("Loaded 1 tools.").waitFor();
  await page.locator("#query").fill("Find the private-fixture-test record");
  await rank.click();
  await page.getByText("Applicability for this request is unverified.").waitFor();
  assert.deepEqual(artifactRequests, []);
  await page.getByRole("radio", { name: "Hybrid" }).check();
  await page.getByText(/Hybrid orders tools/).waitFor({ timeout: 30000 });
  await page.getByText("Input schema", { exact: true }).click();
  assert.match(await page.locator(".result-schema pre").textContent(), /properties/);
  assert.match(await page.locator("#context-size").textContent(), /1 of 1 tools/);
  assert.ok(artifactRequests.includes("onnx/embedding-int8.onnx"));
  const modelRequests = artifactRequests.length;
  await page.getByRole("radio", { name: "Dense" }).check();
  await page.getByText("Numbers show embedding cosine.", { exact: true }).waitFor();
  assert.equal(artifactRequests.length, modelRequests);

  await page.getByText("Evaluate the API holdout", { exact: true }).click();
  await page.getByRole("button", { name: "Run benchmark" }).click();
  await page.getByText("Benchmarked 72 of 96 requests locally.").waitFor({ timeout: 60000 });
  assert.equal(await page.locator("#benchmark-results tbody tr").count(), 3);
  assert.ok(requests.every(([method, address, body]) => method === "GET" && !body
    && !address.includes("private-fixture-test")));
  assert.deepEqual(errors, []);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 390);
  const report = { browser: `Chromium ${browser.version()}`, default_mode: "BM25",
    bm25_artifact_requests: 0, bm25_wasm_requests: 0, no_overlap_matches: 0,
    hybrid_artifact_requests: modelRequests, dense_reused_encoder: true, benchmark_modes: 3,
    benchmark_answerable: 72, benchmark_total: 96, imported_schema_visible: true, network_get_only: true,
    custom_input_in_requests: false, desktop_and_mobile_checked: true };
  report.revision = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  report.checkpoint_revision = revision;
  report.onnx_sha256 = createHash("sha256").update(await readFile(resolve(checkpoint, "onnx/embedding-int8.onnx"))).digest("hex");
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
} finally {
  await browser.close();
}
