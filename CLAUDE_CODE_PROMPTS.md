# Claude Code prompts — one per stage

How to use: put CLAUDE.md in your project folder, open Claude Code there, and paste ONE prompt
at a time. Do not paste the next prompt until the "Done when" checks pass.
Read the code it writes, especially the timing code.

---

## Stage 0 — Setup
```
Read CLAUDE.md. Create the repo layout from it, a requirements.txt, a config.yaml (seeds, graph
sizes, number of repeats, tie threshold, tier definitions), and a pytest setup.
Do not write any algorithms yet.
```
Done when: folders exist, `pytest` runs (even with zero tests), config loads.

---

## Stage 1 — Graph structure and algorithms
```
Read CLAUDE.md. Implement src/graph.py (shared adjacency-list graph with optional node
coordinates and a weighted flag) and src/algorithms.py with: Dijkstra (binary heap),
Bidirectional Dijkstra, A* (Euclidean heuristic, only when coordinates exist), BFS and
Bidirectional BFS (only for unweighted graphs). Each algorithm returns (cost, path, op_counts)
where op_counts holds nodes_popped and edges_relaxed. Add an `applicable_algorithms(graph)` helper.
Write tests: on 200 random graphs, every applicable algorithm must return the same shortest-path
cost as networkx. Also test unreachable targets and source == target.
```
Done when: all tests pass. If costs ever differ, fix before moving on.

---

## Stage 2 — Generators
```
Implement src/generators.py for the graph families listed in CLAUDE.md, driven by config.yaml.
Each generator returns our Graph structure, with coordinates for grid and random geometric graphs.
Add a function that samples several (source, target) query pairs per graph, only from the same
connected component. Every graph gets a unique graph_id. Save instances to data/raw_instances.
Tests: sizes and densities match the config; fixed seed gives identical graphs.
```
Done when: you can generate a few thousand graphs, with a summary table of counts per family.

---

## Stage 3 — Tiered fingerprints
```
Implement src/fingerprint.py with feature functions grouped in Tier-0 to Tier-3 as in CLAUDE.md.
Tier-2 uses sampling (sample size from config). Every function must record its own runtime in
nanoseconds. Provide `extract(graph, tiers=[0,1,...])` returning a feature dict plus a
cost dict per tier. Tests: Tier-0 and Tier-1 values match networkx on small graphs; Tier-2
estimates are within a reasonable error of the exact Tier-3 values on mid-size graphs.
```
Done when: tests pass and you can print the average cost per tier per graph size.

---

## Stage 4 — Benchmark and labels
```
Implement src/benchmark.py. For every (graph, query) run every applicable algorithm with warm-up
runs and the median of N repeats (from config), timing only the search using perf_counter_ns.
Save to data/dataset.csv: graph_id, family, query ids, features per tier, feature costs,
runtimes per algorithm, op counts per algorithm, best algorithm by time, best by op count, and a
tie flag. Make it resumable and parallel-safe (separate processes, but do not run timing jobs
at the same time on the same cores).
Then write a small report script that prints: how often each algorithm wins, per family and per
size bucket, and how noisy the repeated timings are.
```
Done when: you have the report. CHECK THIS BEFORE MODELLING. If one algorithm wins >80%,
change the graph mix in config.yaml and regenerate.

---

## Stage 5 — Baselines and models
```
Implement src/models.py and src/evaluate.py. Models: Random Forest (primary), SVM (RBF, scaled),
small MLP, plus baselines: always-Dijkstra, best-single-fixed-algorithm, random choice, oracle.
Use a GROUP split by graph_id (and cross-validation by group). Metrics: accuracy, regret (mean and
95th percentile), and net time saved (feature time + inference time + chosen algorithm time),
compared to the best single fixed algorithm AND to running all candidates. Respect algorithm
applicability masks. Save results as CSV and make plots into results/.
```
Done when: one command regenerates the results table and plots. Check that the models beat
the best fixed algorithm in net time, not just in accuracy.

---

## Stage 6 — Cost-aware experiment (main contribution 1)
```
In src/experiments/tiers.py, train and evaluate models using Tier-0 only, Tier-0+1, Tier-0+1+2,
and all tiers. Plot selection quality (regret, accuracy) against average feature cost, and plot
net time saved per tier set. Mark the Pareto frontier. Report for which graph sizes expensive
features stop paying off.
```
Done when: you have the Pareto plot and a short table.

---

## Stage 7 — Probing experiment (main contribution 2)
```
Implement src/probe.py: run the default algorithm for the first k node expansions (k from config,
also try a few values) and record frontier growth rate, queue size over time, branching-factor
variance on the frontier, and probe time. Then in src/experiments/probing.py compare
static features vs static+probe features, with probe time counted in the total cost.
Show the accuracy gained per millisecond spent.
```
Done when: you can say honestly whether probing helps, even if it does not.

---

## Stage 8 — Uncertainty and explainability (contributions 3 and 4)
```
In src/models.py add probability calibration and a conformal prediction routine that returns a
SET of algorithms with a target coverage (e.g. 90%). Add a "selector with abstention": if the set
has one algorithm, use it; otherwise run a safe default or the top 2. Evaluate regret and net time
with and without abstention at several confidence levels.
In src/explain.py add SHAP values, permutation importance (grouped by correlated features), and
distil the Random Forest into a shallow decision tree and a plain-text rule list. Compare the
rules with textbook advice and report fidelity of the small tree to the forest.
```
Done when: results/ has the abstention table, SHAP plots, and the rule list.

---

## Stage 9 — Strength checks
```
In src/experiments/ add: (1) size_shift.py — train on small graphs, test on much larger ones;
(2) real_graphs.py — train on synthetic, test on real graphs in data/real_graphs (SNAP/Network
Repository files, plus a city road network from OSMnx); (3) labels_ablation.py — compare models
trained on wall-clock labels vs op-count labels, and test on timings from a second machine if a
second dataset file is provided. Report the drop in regret for each shift.
```
Done when: you have a "how well does it generalize" table.

---

## Stage 10 — Wow layer
```
Build app/demo.py with Streamlit: the user picks a graph type or uploads an edge list, picks a
source and target; the app shows the fingerprint, the selector's choice with a SHAP explanation
and its confidence, then actually runs all applicable algorithms and shows the real timings and
whether the pick was right. Add a second page with the 2D algorithm map (UMAP of fingerprints,
coloured by winning algorithm) and a third with the city road network case study.
```
Done when: `streamlit run app/demo.py` works end to end on your laptop.

---

## Final stage — Report support
```
Write a script that regenerates every table and figure in results/ from data/dataset.csv.
Then draft a README.md with: problem, method, experiment list, how to reproduce, and a
"Limitations" section. Do not include any number that is not produced by a script.
```

---

## Tips
- Commit to git after every stage so you can roll back.
- If Claude Code changes earlier files, ask it to explain why.
- Run the timing benchmark with your laptop plugged in, other apps closed, and the same settings every time.
- Keep a short log of choices and surprises; it becomes your report's discussion section.
