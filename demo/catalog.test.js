import assert from "node:assert/strict";
import test from "node:test";

import { importCatalog, parseCatalog } from "./catalog.js";

const spec = {
  openapi: "3.1.0",
  paths: { "/exhibits/{id}": { $ref: "#/components/pathItems/exhibits~1read" } },
  components: { pathItems: { "exhibits/read": {
    get: { operationId: "museum/find", summary: "Find an exhibit" },
    head: { description: "Check availability" },
  } } },
};

test("imports OpenAPI JSON with local path references and a missing operation ID", () => {
  assert.deepEqual(parseCatalog(JSON.stringify(spec)), [
    { name: "head_exhibits_by_id", description: "Check availability\nHEAD /exhibits/{id}" },
    { name: "museum_find", description: "Find an exhibit\nGET /exhibits/{id}" },
  ]);
});

test("imports a YAML specification and the legacy MCP response format", async () => {
  const yaml = "openapi: 3.0.3\npaths:\n  /hours:\n    get:\n      operationId: opening_hours\n";
  assert.deepEqual(await importCatalog(new File([yaml], "museum.yaml")), [
    { name: "opening_hours", description: "GET /hours" },
  ]);
  assert.deepEqual(parseCatalog('{"result":{"tools":[{"name":"lookup"}]}}'),
    [{ name: "lookup", description: "" }]);
});

test("rejects external or cyclic references without resolving them", () => {
  for (const ref of ["https://example.invalid/other.yaml", "#/paths/~1exhibits~1{id}"]) {
    const modified = { ...spec, paths: { "/exhibits/{id}": { $ref: ref } } };
    assert.throws(() => parseCatalog(JSON.stringify(modified)), /local references|cycle/);
  }
});

test("rejects over-limit files before reading and keeps the 128-tool limit", async () => {
  await assert.rejects(importCatalog({ size: 1_000_001, text: () => assert.fail("file was read") }), /under 1 MB/);
  const people = Array.from({ length: 129 }, (_, index) => ({ name: `tool_${index}` }));
  assert.throws(() => parseCatalog(JSON.stringify(people)), /1 to 128 tools/);
});

test("keeps imported MCP input schemas for local inspection", () => {
  const inputSchema = { type: "object", properties: { ref: { type: "string" } }, required: ["ref"] };
  const tools = parseCatalog(JSON.stringify({ tools: [{ name: "lookup", inputSchema }] }));
  assert.deepEqual(tools[0].inputSchema, inputSchema);
});
