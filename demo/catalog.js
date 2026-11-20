const HTTP_METHODS = ["get", "put", "post", "delete", "options", "head", "patch", "trace"];
export const MAX_IMPORT_BYTES = 1_000_000;

function object(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function pathItem(spec, item, visited = new Set()) {
  if (!object(item)) throw new Error("Each OpenAPI path item must be an object.");
  if (!Object.hasOwn(item, "$ref")) return item;
  const { $ref: ref, ...fields } = item;
  if (typeof ref !== "string" || !ref.startsWith("#/")) {
    throw new Error("OpenAPI path items must use local references.");
  }
  if (visited.has(ref)) throw new Error("OpenAPI path item references contain a cycle.");
  visited.add(ref);
  const keys = decodeURIComponent(ref.slice(2)).split("/")
    .map((key) => key.replaceAll("~1", "/").replaceAll("~0", "~"));
  let target = spec;
  for (const key of keys) {
    if (!object(target) || !Object.hasOwn(target, key)) throw new Error("Unresolved OpenAPI path item reference.");
    target = target[key];
  }
  const resolved = pathItem(spec, target, visited);
  if (Object.keys(fields).some((key) => Object.hasOwn(resolved, key))) {
    throw new Error("OpenAPI path item reference has conflicting fields.");
  }
  return { ...resolved, ...fields };
}

function openApiTools(spec) {
  if (typeof spec.openapi !== "string" || !/^3\.[01]\.\d+$/.test(spec.openapi)) {
    throw new Error("Use an OpenAPI 3.0 or 3.1 specification.");
  }
  if (!object(spec.paths)) throw new Error("The OpenAPI spec needs a paths object.");
  const tools = [];
  for (const [route, raw] of Object.entries(spec.paths)) {
    if (!route.startsWith("/")) continue;
    const item = pathItem(spec, raw);
    for (const method of HTTP_METHODS) {
      if (!Object.hasOwn(item, method)) continue;
      const operation = item[method];
      if (!object(operation)) throw new Error("Each OpenAPI operation must be an object.");
      const { operationId, summary = "", description = "" } = operation;
      if (typeof summary !== "string" || typeof description !== "string") {
        throw new Error("OpenAPI summaries and descriptions must be strings.");
      }
      if (Object.hasOwn(operation, "operationId") && (typeof operationId !== "string" || !operationId.trim())) {
        throw new Error("OpenAPI operation IDs must be nonempty strings.");
      }
      const routeName = route.replace(/^\/+|\/+$/g, "").replace(/\{([^}]+)\}/g, "by_$1");
      const name = (operationId ?? `${method}_${routeName}`).replace(/[^A-Za-z0-9_.-]+/g, "_");
      const details = [...new Set([summary.trim(), description.trim()].filter(Boolean))];
      tools.push({ name, description: [...details, `${method.toUpperCase()} ${route}`].join("\n") });
    }
  }
  return tools;
}

export function normalizeCatalog(data) {
  let tools;
  if (object(data) && Object.hasOwn(data, "openapi")) tools = openApiTools(data);
  else tools = Array.isArray(data) ? data : (data?.result?.tools ?? data?.tools);
  if (!Array.isArray(tools) || tools.length < 1 || tools.length > 128) {
    throw new Error("Provide a catalog with 1 to 128 tools.");
  }
  const seen = new Set();
  return tools.map((tool) => {
    const name = typeof tool?.name === "string" ? tool.name.trim() : "";
    if (!name || name.length > 256 || seen.has(name)) {
      throw new Error("Tool names must be unique and contain 1 to 256 characters.");
    }
    if (tool.description != null && typeof tool.description !== "string") {
      throw new Error("Tool descriptions must be strings.");
    }
    seen.add(name);
    const normalized = { name, description: (tool.description ?? "").slice(0, 12000) };
    if (object(tool.inputSchema)) normalized.inputSchema = tool.inputSchema;
    return normalized;
  }).sort((left, right) => left.name < right.name ? -1 : Number(left.name > right.name));
}

export function parseCatalog(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("The catalog must be valid JSON.");
  }
  return normalizeCatalog(data);
}

export async function importCatalog(file) {
  if (file.size > MAX_IMPORT_BYTES) throw new Error("Choose a catalog or spec under 1 MB.");
  const text = await file.text();
  if (/\.ya?ml$/i.test(file.name)) {
    const { parse } = await import("yaml");
    return normalizeCatalog(parse(text, { merge: false, maxAliasCount: 30 }));
  }
  return parseCatalog(text);
}
