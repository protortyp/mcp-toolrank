import holdout from "../data/onnx_int8_holdout.json";

const samples = [
  { name: "search_calendar", description: "Find scheduled meetings and events within a date range." },
  { name: "create_calendar_event", description: "Schedule a new meeting with a title, time and participants." },
  { name: "search_files", description: "Find documents and folders by title or keyword." },
  { name: "get_weather", description: "Find the weather forecast for a city and date." },
  { name: "search_contacts", description: "Look up names, email addresses and phone numbers." },
  { name: "get_train_departures", description: "List the next trains departing from a station." },
  { name: "search_email", description: "Find email messages by sender, date or keywords." },
];
const apiTools = Object.values(holdout.tools).map(({ name, description }) => ({ name, description }));
const apiCases = Object.fromEntries(holdout.queries.map(({ id, query, positives }) => [
  id, { query, relevant: positives.map((key) => holdout.tools[key].name) },
]));
const defaultExample = "crossref/005";
const catalog = document.querySelector("#catalog");
const catalogEditor = document.querySelector("#catalog-editor");
const query = document.querySelector("#query");
const status = document.querySelector("#status");
const button = document.querySelector("#submit");
const reset = document.querySelector("#sample");
const listing = document.querySelector("#ranking");
const placeholder = document.querySelector("#placeholder");
const scoreHint = document.querySelector("#score-hint");
const example = document.querySelector("#example");
const judgment = document.querySelector("#judgment");
const rankingMode = document.querySelector("#ranking-mode");
const benchmarkButton = document.querySelector("#benchmark");
const benchmarkTable = document.querySelector("#benchmark-results");
let worker;
let ranked;
let exampleCatalog;

function showStatus(message, failed = false) {
  status.textContent = message;
  status.classList.toggle("error", failed);
}

function busy(active) {
  for (const control of [button, reset, example, benchmarkButton]) control.disabled = active;
}

function render(rows) {
  const items = rows.map(({ tool, score }) => {
    const li = document.createElement("li");
    const line = document.createElement("div");
    line.className = "result-name";
    const name = document.createElement("code");
    name.textContent = tool.name;
    const value = document.createElement("span");
    value.className = "value";
    value.textContent = score.toFixed(3);
    const description = document.createElement("p");
    description.textContent = tool.description;
    line.append(name, value);
    li.append(line, description);
    return li;
  });
  listing.replaceChildren(...items);
  placeholder.hidden = true;
  scoreHint.hidden = false;
}

function initWorker() {
  const instance = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  instance.onmessage = ({ data }) => {
    if (data.kind === "status") {
      showStatus(data.message);
      return;
    }
    busy(false);
    if (data.kind === "error") {
      showStatus(data.message, true);
      return;
    }
    if (data.kind === "benchmark") {
      renderBenchmark(data.result);
      showStatus(`Benchmarked ${data.answerable} of ${data.total} requests locally.`);
      return;
    }
    ranked = data;
    renderRanked();
    showStatus(`Ranked ${data.count} tools.`);
  };
  instance.onerror = () => {
    instance.terminate();
    worker = undefined;
    busy(false);
    showStatus("The browser runtime could not start. Reload the page and try again.", true);
  };
  return instance;
}

function parseCatalog(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("The catalog must be valid JSON.");
  }
  const tools = Array.isArray(data) ? data : (data?.result?.tools ?? data?.tools);
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
    return { name, description: (tool.description ?? "").slice(0, 12000) };
  }).sort((left, right) => left.name < right.name ? -1 : Number(left.name > right.name));
}

function clearRanking() {
  ranked = undefined;
  listing.replaceChildren();
  placeholder.hidden = false;
  scoreHint.hidden = true;
  judgment.hidden = true;
  rankingMode.hidden = true;
}

function renderRanked() {
  if (!ranked) return;
  const mode = rankingMode.querySelector("input:checked").value;
  render(ranked.results[mode]);
  rankingMode.hidden = false;
  scoreHint.textContent = mode === "hybrid"
    ? "Numbers show embedding cosine. Hybrid orders tools using cosine, TF-IDF and BM25."
    : "Dense ranks by embedding cosine. A high score can appear even when no tool applies.";
  judgment.hidden = ranked.relevant === null;
  if (!judgment.hidden) {
    judgment.textContent = ranked.relevant.length
      ? `Labelled match: ${ranked.relevant.join(", ")}`
      : "The benchmark labels this request as having no matching tool.";
  }
}

function renderBenchmark(result) {
  const percent = (value) => `${(100 * value).toFixed(2)}%`;
  const rows = Object.entries(result).map(([mode, { hitAt1, ndcgAt10 }]) => {
    const row = document.createElement("tr");
    for (const value of [mode === "dense" ? "Dense" : "Hybrid", percent(hitAt1), percent(ndcgAt10)]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  });
  benchmarkTable.querySelector("tbody").replaceChildren(...rows);
  benchmarkTable.hidden = false;
}

function loadExample() {
  clearRanking();
  showStatus("");
  const id = example.value;
  if (id === "custom") {
    catalogEditor.open = true;
    return;
  }
  exampleCatalog = JSON.stringify({ tools: id === "calendar" ? samples : apiTools }, null, 2);
  catalog.value = exampleCatalog;
  query.value = id === "calendar" ? "Find tomorrow’s meetings" : apiCases[id].query;
}

function resetExample() {
  example.value = defaultExample;
  catalogEditor.open = false;
  loadExample();
}
reset.addEventListener("click", resetExample);
example.addEventListener("change", loadExample);
for (const field of [query, catalog]) {
  field.addEventListener("input", () => { example.value = "custom"; });
}
rankingMode.addEventListener("change", renderRanked);
resetExample();

benchmarkButton.addEventListener("click", () => {
  worker ??= initWorker();
  busy(true);
  benchmarkTable.hidden = true;
  worker.postMessage({ tools: parseCatalog(JSON.stringify(apiTools)), cases: Object.values(apiCases) });
  showStatus("Preparing the benchmark…");
});

document.querySelector("#rank-form").addEventListener("submit", (event) => {
  event.preventDefault();
  clearRanking();
  try {
    const text = query.value.trim();
    if (!text) throw new Error("Enter a request to rank the tools.");
    const tools = parseCatalog(catalog.value);
    const apiCase = apiCases[example.value];
    const relevant = apiCase?.query === text && catalog.value === exampleCatalog ? apiCase.relevant : null;
    worker ??= initWorker();
    busy(true);
    worker.postMessage({ tools, text, relevant });
    showStatus("Preparing ToolRank…");
  } catch (error) {
    busy(false);
    showStatus(error.message, true);
  }
});
