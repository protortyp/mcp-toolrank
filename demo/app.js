const samples = [
  { name: "search_calendar", description: "Find scheduled meetings and events within a date range." },
  { name: "create_calendar_event", description: "Schedule a new meeting with a title, time and participants." },
  { name: "search_files", description: "Find documents and folders by title or keyword." },
  { name: "get_weather", description: "Find the weather forecast for a city and date." },
  { name: "search_contacts", description: "Look up names, email addresses and phone numbers." },
  { name: "get_train_departures", description: "List the next trains departing from a station." },
  { name: "search_email", description: "Find email messages by sender, date or keywords." },
];
const catalog = document.querySelector("#catalog");
const query = document.querySelector("#query");
const status = document.querySelector("#status");
const button = document.querySelector("#submit");
const reset = document.querySelector("#sample");
const listing = document.querySelector("#ranking");
const placeholder = document.querySelector("#placeholder");
const scoreHint = document.querySelector("#score-hint");
let worker;

function showStatus(message, failed = false) {
  status.textContent = message;
  status.classList.toggle("error", failed);
}

function idle() {
  button.disabled = false;
  reset.disabled = false;
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
    idle();
    if (data.kind === "error") {
      showStatus(data.message, true);
      return;
    }
    render(data.results);
    showStatus(`Ranked ${data.count} tools.`);
  };
  instance.onerror = () => {
    instance.terminate();
    worker = undefined;
    idle();
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

function loadSample() {
  catalog.value = JSON.stringify({ tools: samples }, null, 2);
  showStatus("");
}
reset.addEventListener("click", loadSample);
loadSample();

document.querySelector("#rank-form").addEventListener("submit", (event) => {
  event.preventDefault();
  listing.replaceChildren();
  placeholder.hidden = false;
  scoreHint.hidden = true;
  try {
    const text = query.value.trim();
    if (!text) throw new Error("Enter a request to rank the tools.");
    const tools = parseCatalog(catalog.value);
    worker ??= initWorker();
    button.disabled = true;
    reset.disabled = true;
    worker.postMessage({ tools, text });
    showStatus("Preparing ToolRank…");
  } catch (error) {
    idle();
    showStatus(error.message, true);
  }
});
