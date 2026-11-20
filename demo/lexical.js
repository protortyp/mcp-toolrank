function words(text) {
  return (text.replace(/(?<=[a-z0-9])(?=[A-Z])/g, " ").toLowerCase().match(/[a-z]+|[0-9]+/g) ?? [])
    .filter((word) => word.length >= 2);
}

function counts(tokens, bigrams = false) {
  const features = new Map();
  function add(term) { features.set(term, (features.get(term) ?? 0) + 1); }
  for (let i = 0; i < tokens.length; i += 1) {
    add(tokens[i]);
    if (bigrams && i > 0) add(`${tokens[i - 1]} ${tokens[i]}`);
  }
  return features;
}

function postings(documents) {
  const index = new Map();
  documents.forEach((document, row) => {
    for (const [term, count] of document) {
      if (!index.has(term)) index.set(term, []);
      index.get(term).push({ row, count });
    }
  });
  return index;
}

export function lexicalIndex(tools, documents) {
  const tokens = documents.map(words);
  const tfidf = postings(tokens.map((row) => counts(row, true)));
  const idf = new Map();
  const squared = new Float64Array(tools.length);
  for (const [term, entries] of tfidf) {
    const weight = 1 + Math.log((tools.length + 1) / (entries.length + 1));
    idf.set(term, weight);
    for (const entry of entries) {
      entry.weight = entry.count * weight;
      squared[entry.row] += entry.weight ** 2;
    }
  }
  for (const entries of tfidf.values()) {
    for (const entry of entries) entry.weight /= Math.sqrt(squared[entry.row]);
  }

  const weighted = tokens.map((row, i) => {
    const name = /^[A-Za-z_][\w.-]*$/.test(tools[i].name) ? words(tools[i].name) : [];
    return [...name, ...name, ...row];
  });
  const average = (weighted.reduce((total, row) => total + row.length, 0) / tools.length) || 1;
  const bm25 = postings(weighted.map((row) => counts(row)));
  for (const entries of bm25.values()) {
    const weight = Math.log1p((tools.length - entries.length + .5) / (entries.length + .5));
    for (const entry of entries) {
      const penalty = 1.2 * (.25 + .75 * weighted[entry.row].length / average);
      entry.weight = weight * entry.count * 2.2 / (entry.count + penalty);
    }
  }

  return (request, cosine) => {
    const query = words(request);
    const lexical = new Float64Array(tools.length);
    const names = new Float64Array(tools.length);
    let norm = 0;
    for (const [term, count] of counts(query, true)) {
      if (!idf.has(term)) continue;
      const weight = count * idf.get(term);
      norm += weight ** 2;
      for (const entry of tfidf.get(term)) lexical[entry.row] += entry.weight * weight;
    }
    for (const term of new Set(query)) {
      for (const entry of bm25.get(term) ?? []) names[entry.row] += entry.weight;
    }
    const tfidfScale = .4 / (Math.sqrt(norm) || 1);
    const bm25Scale = .15 / (Math.max(...names) || 1);
    const normalizedBm25 = names.map((score) => score * bm25Scale / .15);
    return {
      bm25: normalizedBm25,
      hybrid: cosine.map((score, i) => score + tfidfScale * lexical[i] + bm25Scale * names[i]),
    };
  };
}
