import assert from "node:assert/strict";
import test from "node:test";
import { lexicalIndex } from "./lexical.js";

test("BM25 exposes the same normalized component used by the hybrid", () => {
  const documents = ["get_weather: Forecast for a city", "get_train: Find a train station"];
  const index = lexicalIndex([{ name: "get_weather" }, { name: "get_train" }], documents);
  const { hybrid, bm25 } = index("train", [.1, .1]);
  assert.equal(bm25[1], 1);
  assert.equal(bm25[0], 0);
  assert.ok(hybrid[1] > hybrid[0]);
});
