# Hybrid tool selection

Set `lexical_weight=0.4` to add 0.4 times TF-IDF cosine to embedding cosine. The selector indexes tool names and descriptions once, splitting camel case and punctuation before extracting word unigrams and bigrams.

```python
from sentence_transformers import SentenceTransformer
from toolrank.selector import ToolSelector

model = SentenceTransformer("christianengel/mcp-toolrank-33m")
tools = [
    {"name": "find_calendar_events", "description": "Find scheduled meetings"},
    {"name": "check_weather", "description": "Read the weather forecast"},
]
selector = ToolSelector(model, tools, lexical_weight=0.4)
matches = selector.rank("Find tomorrow's meetings", k=1)
print(matches[0].tool["name"])
```

The default weight is zero. Positive weights rank by hybrid score. `Match.cosine` and `min_score` use embedding cosine. Validate rejection thresholds on your catalog.

## Results

We chose 0.4 on a separate development set of 872 queries across 34 tools. Scores are nDCG@10 on answerable requests.

Data | Requests | BGE dense | BGE hybrid | ToolRank dense | ToolRank hybrid
--- | ---: | ---: | ---: | ---: | ---:
ETOM compact | 1,549 | 88.95% | 91.27% | 90.13% | 92.02%
ETOM MCP JSON | 1,549 | 87.94% | 91.08% | 87.98% | 91.02%
Independent API catalog | 72 | 87.64% | 89.82% | 92.15% | 91.39%
ToolRet | 7,961 | 28.81% | 34.60% | 28.53% | 34.00%

ToolRank gained on both ETOM formats and lost 0.76 points on the independent API catalog. BGE hybrid led on MCP JSON. Gains depend on catalog descriptions.

## Reproduce

Use the locked uv environment, local checkpoint snapshot, and pinned BGE and benchmark inputs. Fetch ToolRet shards first. Pass `--device cpu` for CPU. Omit `--dataset` to skip ToolRet.

```sh
uv run --locked --group research --extra cu128 python -m toolrank.replay_hybrid \
  --toolrank /path/to/model-snapshot --output /tmp/toolrank-hybrid.json \
  --dataset etom-plain etom-mcp api toolret
```

[Published checkpoint](https://huggingface.co/christianengel/mcp-toolrank-33m/tree/51a961aa33a2e6dfecc6845df77203778787ec4a) · [Reproduction setup](MODERN_MCP.md#reproduction) · [Machine-readable results](../results/hybrid-tool-selection.json)
