# Experiment plan

The experiment tests whether a 33.36M-parameter encoder can improve tool retrieval with intent-level contrastive supervision. It starts from the MIT-tagged BGE small checkpoint. The candidate receives only original single-intent examples written for this project. Tools in a training family use related actions that can serve as hard negatives.

## Splits and protocol

- Training uses 88 original positive queries for 44 tools in 11 families. Each tool has two paraphrases. Neither evaluation corpus contributes queries, tools, label aliases, templates, nor mining candidates to training.
- Development uses three disjoint families. It determines the training recipe and per-model abstention thresholds. Rank recipes by development Recall@1 averaged over three seeds, then by the lower learning rate, then by the named recipe order in the run script. Use the first seed from the winning recipe for deployment measurements.
- The authored test uses four other families, a pooled 12-tool catalog, and both relevant and unanswerable requests. All three schema variants preserve the same names and descriptions. The JSON variants test serialization changes with minimal schemas. They do not test real parameter semantics or production workload diversity.
- ToolRet is a secondary public test. Use the complete downloadable mangopy catalog across all three categories, query text without benchmark-supplied task instructions, and exact normalized cosine search. Model-level retrieval prefixes still apply. Compute query-weighted Recall@1/5/10, nDCG@10 with linear gains, and completeness. Keep per-task results for inspection. This is distinct from MTEB subset evaluation and from paper headline aggregates. The Tool-Embed run caps input at 512 tokens to match the BGE comparators. No ToolRet metric selects a checkpoint or threshold.
- CPU and GPU profiles use a fixed, independent workload of eight requests, batch-one encoder latency including tokenization and a warmed batch-16 encoder throughput measurement. Report peak process RSS and CUDA allocated memory using separate processes. Full ToolRet evaluation also measures document embedding throughput.

## Training and decision

Compare in-batch positives against same-family negatives and base-model mined negatives using InfoNCE. Pilot recipe changes cover batch size, learning rate, temperature, and negative sampling. Run all recipes with seeds 11, 23, and 37. Training weights and logs remain local.

The quality path to GO requires a positive paired 95% interval on primary unseen-family Recall@1 against ToolScout, with no decline greater than 2 percentage points on ToolRet nDCG@10 versus ToolScout. The efficiency path requires comparable quality within 2 points and at least 2× faster warmed CPU p50 or 2× lower resident memory against the strongest qualified comparison, without a loss greater than 2 points against ToolScout. Families are the statistical unit for the authored test. Four families give limited statistical power, and a broader independently authored holdout is required before making a public generalization claim. Stop after this local pilot if the evidence does not support paid runs.

For abstention, select the development max-score threshold by maximizing the average of answerable coverage and unanswerable rejection, resolving ties toward greater rejection. Apply that threshold to the same model and representation without refitting on test.

All runs are local. The initial external cap is $20 and the lifetime cap is $100. An external purchase, model upload, or code release is outside this pilot.
