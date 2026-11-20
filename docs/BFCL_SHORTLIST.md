# CPU shortlists on native BFCL menus

BM25, BGE Small INT8 hybrid, and ToolRank INT8 hybrid kept every required function at five on all 1,077 answerable requests. Across 1,961 pinned live BFCL requests with native menus, ToolRank’s top five saved 7.5% of serialized tool JSON bytes.

| Native menu | Requests | Answerable | ToolRank top-five byte savings |
| --- | ---: | ---: | ---: |
| 0–5 tools | 1,729 | 917 | 0% |
| 6–10 tools | 227 | 158 | 29.2% |
| 11–20 tools | 2 | 0 | 59.0% |
| More than 20 tools | 3 | 2 | 82.1% |

Using top five only when it halved tool JSON saved 1.7% overall. Keeping just gold functions would save 72.7% on answerable requests. This static ranking test contains few large menus.

With frozen weights and NWS cosine cutoffs, ToolRank kept all required tools on 469/1,077 requests and exposed tools on 117/884 unsupported requests. BGE kept 161/1,077 and exposed tools on 44/884. Validate cutoffs on labeled target requests.

Save `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` in `REVISION` alongside the three data files and applicable `possible_answer` labels. The runner verifies hashes.

```sh
PYTHONPATH=. uv run --no-sync python tests/bfcl_shortlist/run.py \
  --bfcl /path/to/bfcl --bge /path/to/bge-small-int8 \
  --toolrank /path/to/toolrank-int8 --output /tmp/bfcl-shortlist.json
```

[BFCL data at the pinned revision](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data) · [Manifest](../tests/bfcl_shortlist/manifest.json) · [Results](../results/bfcl_shortlist.json) · [NWS pilot](NWS_PILOT.md)
