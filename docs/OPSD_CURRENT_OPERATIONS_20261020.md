# OPSD current operations and readiness (2026-10-20)

This is the current status note for OPSD SA-negative hardware preparation and the matched topology plan. It does not change either frozen protocol. Historical reports remain intact; this note supersedes their operational status statements where noted.

## Frozen v8 preparation

- Package: `export/opsd_sa_negative_kaiwu_preparation_v8/`; v7 is an independent historical package and was not relabeled.
- Protocol: `fa-bm-vae-opsd-spqc-550-sa-negative-v3-vs-crps`; frozen SHA-256 `2319f9210bd7138e88cee941b0f5bea0a586b106877211c4d46d3f60ad085710`.
- Checkpoint SHA-256: `0281db5b136c21e5c3327b9a30c4f31dadeb4c04b90998a28670fc57ff347a5c`.
- Dataset SHA-256: `507ed5ae709c05f01e6e67209e5ae08b2567f96755c84ed2444dc5cd7d296555`.
- Preparation manifest SHA-256: `d5b2749a45ac94b0ef0c73b25a9bb0ac8202db3621f82dcbde8a81c1ee77ffff`; quality reference manifest SHA-256: `47f228f752ca74a26e90b08d20bcc21e70e675b88f57b0fd502ce5a45f18e1c3`.
- The quality policy is approved as a decision rule; this does not approve sampling. Engineering identity, matrix, response, and decoding errors are hard stops. The four latent indicators—energy Wasserstein, effective-beta error, edge-moment MAE, and magnetization MAE—are diagnostics. Their historical thresholds are reference values, not validated acceptance limits for current sampling distributions, and must not be searched or tuned in this workflow.
- Gain selection requires the complete valid real VS panel. Recompute predictions from the raw responses and select minimum mean Raw VS CRPS; an exact tie goes to the smaller gain. A complete valid real VS response set and frozen real gain do not yet exist. TEST remains locked and needs independent explicit confirmation. No existing lock or approval state was changed.
- v8 latent diagnostics use **int8 hardware-matrix SA** as their reference. Floating-point logical SA is a separate inference backend and must not be described as the v8 reference distribution.
- Historic v8 dry-run/mock records remain non-physical fixtures. They cannot establish a real platform result, freeze the real gain, or authorize TEST.

## Matched topology plan

`export/opsd_sa_matched_random_graph_v1/formal_task_manifest.json` contains six commands with disjoint output paths: `formal/temporal/seed_N/` and `formal/random/seed_N/`, `N=0,1,2`. The two configs use distinct run labels. Every command checks that its output directory is absent before execution. If a path exists later, identity verification and separate confirmation are required before recovery.

The random graph is one preregistered topology generated with graph seed 20261020. The three training seeds share this same topology; they are not three independent random graphs and cannot support a claim of superiority over the random-graph population. Training seeds are 0/1/2, graph seed is 20261020, and paired generation seeds are 0/1/2. The completed two-epoch, five-scenario VS run was a short functional check only. All six formal training jobs are **NOT RUN**. Real hardware experiments are also **NOT RUN**.

Frozen topology evaluation: Raw Ramp CRPS and Variogram are primary; Raw CRPS, Energy, coverage, and bias are secondary; ECC is excluded from primary topology attribution. Report date-level results and retain both arms by seed.

## Kaiwu local package facts

Read-only inspection recorded distribution version `1.4.1` and imported module version `1.0.7`; these are different version fields from the installed package and are not evidence of an upgrade need by themselves. Current import path and fresh environment inspection are recorded in the zero-credit audit. No SDK upgrade or reinstall was done. Account routing, license, project access, and remaining Sampling Credits are unverified.

## Current execution state

| Work | State |
|---|---|
| Offline protocol, adapter, quality, selection, and preparation tests | See independent audit report for this revision |
| v8 identity and local zero-response preflight | See independent audit report |
| Single-task canary dry-run, VS / gain 125 / 100 reads | See independent audit report; dry-run only, no SDK submission |
| Real canary and real VS | **NOT RUN**, requires separate user authorization and external account/billing checks |
| Formal matched-topology training (six jobs) | **NOT RUN**, requires separate user authorization |
| TEST sampling or new TEST metrics | **NOT RUN**; TEST remains locked |

Audit reports are stored under a fresh `export/opsd_zero_credit_acceptance_*` directory; they do not modify v8 or topology artifacts.

## 2026-10-20 matched-topology completion addendum

The six formal classical topology jobs listed above have now completed in
`export/opsd_sa_matched_random_graph_v1/formal/`. The older `NOT RUN` wording
in this retained status note describes the state before this addendum and is
not a claim about the current artifacts.

- Temporal graph: seeds 0, 1, and 2.
- Matched random graph: seeds 0, 1, and 2, all sharing the one fixed graph
  generated with `graph_seed=20261020`.
- Training and inference used SA; generation seeds were explicitly paired to
  training seeds (`0->0`, `1->1`, `2->2`) and the output metadata records the
  VS/TEST split seed map. Only VS scenarios were generated in these six jobs.
- The evaluation is in
  `export/opsd_sa_matched_random_graph_v1/formal_vs_evaluation/` and reports
  Raw daily metrics. Raw Ramp CRPS and Variogram were preregistered as the
  topology-primary measures; Raw CRPS, Energy, PICP90, interval width, and
  bias are secondary. No ECC or TEST result was used for the topology
  attribution.
- These are three training seeds on one fixed random topology, not three
  independent random graphs. The comparison cannot establish superiority over
  a population of random graphs. Hardware flags remain false and no platform
  task was submitted.

The reproducible task-level provenance is in
`export/opsd_sa_matched_random_graph_v1/formal_task_manifest_v2.json`; the
handoff report and numerical summary are in
`export/opsd_sa_matched_random_graph_v1/formal_vs_evaluation/TOPOLOGY_HANDOFF_REPORT.md`.
