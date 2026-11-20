import { Tokenizer } from "@huggingface/tokenizers";
import { env, InferenceSession, Tensor } from "onnxruntime-web/wasm";

const revision = "a24a121266154006e2bd2e32a41576437c3455c2";
const root = `https://huggingface.co/christianengel/mcp-toolrank-33m/resolve/${revision}/`;
const batchSize = 32;
let model;
let catalogKey;
let catalogVectors;

function status(message) {
  self.postMessage({ kind: "status", message });
}

async function artifact(name, binary = false) {
  const response = await fetch(new URL(name, root), { credentials: "omit" });
  if (!response.ok) throw new Error(`Could not load ${name} (HTTP ${response.status}).`);
  return binary ? response.arrayBuffer() : response.json();
}

async function load() {
  env.wasm.numThreads = 1;
  const [data, config, prompts, encoderConfig, bytes] = await Promise.all([
    artifact("tokenizer.json"),
    artifact("tokenizer_config.json"),
    artifact("config_sentence_transformers.json"),
    artifact("sentence_bert_config.json"),
    artifact("onnx/embedding-int8.onnx", true),
  ]);
  const tokenizer = new Tokenizer(data, config);
  const session = await InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  return {
    tokenizer, session, prefix: prompts.prompts.query, maxLength: encoderConfig.max_seq_length,
    padId: tokenizer.token_to_id(config.pad_token),
  };
}

async function encode(encoder, texts) {
  const rows = texts.map((text) => {
    const { ids } = encoder.tokenizer.encode(text, { add_special_tokens: true });
    if (ids.length <= encoder.maxLength) return ids;
    return [...ids.slice(0, encoder.maxLength - 1), ids.at(-1)];
  });
  const length = Math.max(...rows.map((row) => row.length));
  const width = rows.length * length;
  const ids = new BigInt64Array(width).fill(BigInt(encoder.padId));
  const attention = new BigInt64Array(width);
  const tokenTypes = new BigInt64Array(width);
  rows.forEach((row, rowIndex) => {
    const start = rowIndex * length;
    ids.set(row.map(BigInt), start);
    attention.fill(1n, start, start + row.length);
  });
  const shape = [rows.length, length];
  const out = await encoder.session.run({
    input_ids: new Tensor("int64", ids, shape),
    attention_mask: new Tensor("int64", attention, shape),
    token_type_ids: new Tensor("int64", tokenTypes, shape),
  });
  const embedding = out.embedding;
  const dimension = embedding.dims[1];
  const vectors = rows.map((_, index) => embedding.data.slice(index * dimension, (index + 1) * dimension));
  embedding.dispose();
  return vectors;
}

function dot(a, b) {
  let score = 0;
  for (let i = 0; i < a.length; i += 1) score += a[i] * b[i];
  return score;
}

self.onmessage = async ({ data: { tools, text } }) => {
  try {
    if (!model) {
      status("Downloading the model and starting the browser runtime…");
      model = await load();
    }
    const key = JSON.stringify(tools);
    if (catalogKey !== key) {
      status(`Embedding ${tools.length} tools…`);
      const docs = tools.map(({ name, description }) => `${name}: ${description}`);
      const vectors = [];
      for (let i = 0; i < docs.length; i += batchSize) {
        vectors.push(...await encode(model, docs.slice(i, i + batchSize)));
      }
      catalogVectors = vectors;
      catalogKey = key;
    }
    const [request] = await encode(model, [model.prefix + text]);
    const results = tools.map((tool, i) => ({ tool, score: dot(request, catalogVectors[i]) }))
      .sort((left, right) => right.score - left.score)
      .slice(0, 5);
    self.postMessage({ kind: "results", results, count: tools.length });
  } catch (error) {
    self.postMessage({ kind: "error", message: error instanceof Error ? error.message : "ToolRank failed." });
  }
};
