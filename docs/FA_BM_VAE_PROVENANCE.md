# FA-BM-VAE GEFCom Wind five-seed provenance

This report audits existing artifacts; it does not retrain models, regenerate scenarios, or change metrics. `QBMVAE_2` is a historical artifact name for the FA-BM-VAE mainline.

## Artifact identity

The following are SHA-256 hashes of complete files in `export/qbm_vae_wind/`. For each seed, the columns identify the model pickle, anchor checkpoint, VS scenarios, and TEST scenarios, respectively. The generated result directory is ignored by Git; these hashes identify the files used for the reported comparison.

| Seed | Model | Anchor | VS scenarios | TEST scenarios |
|---:|---|---|---|---|
| 0 | `74090e325290145ee89c156b5ac7b3e555fb848e35bca79bcfe4439a382b7a47` | `b1c86415f71c9dda1b65ec00c15443e84700b03ea7f92c37d4e13c58c8f20960` | `1884c3687186a3c3fcee0c153ac6b60059ffdca7d3499c57d514ba8af1cdfe80` | `b71a83ccb11429840d796a938e6648c36f638baeb1d044aa09b819ac376ee256` |
| 1 | `d53104c94caa8808f661d45f97fba6be739180b0a8ac60e10822ff8605822e61` | `57619854e8ec93d887ff63045df68923782c1987b38d659a6bc9a2bd066808fb` | `88231bf23457f4ad9740b08ed7f4f250dc93a5966f961bd0f66651f2987ed198` | `3a4069ed61d0c8a85f4e687e0b359ffd3e3fcabe85c9e109649cd09ce0380226` |
| 2 | `98751bdf1815a9da1eb547ac01272073d64b63d4ed58768fc5bbae3aa031f94e` | `61589ab4cbfc836e7b7371e04f53135aea25c1eb08e360c0c27fb0752e10f60f` | `f38648060b642abf91bfb0e853c3fb15f42793070aebc24fb992e2807845a463` | `bb2f0c88259e7697161e7b0a141f4c8bc29adf4cb88fcdd9f2b0f500e5221b5e` |
| 3 | `783a3077dc7cfb7903096575898027ff1c1be2bbd428c042cbbe7f5143ef775e` | `69391ca7eb9efc395992b462b0f2b7231d18883e67209f0bb87bb867e4de6f7c` | `36414e0e75316ea6ba23d1e171b2259ad9054e29fbb1dd4fb842efb7becdb5db` | `7b29b33b70c685ea90430a9f6b1ef673fa22c06eb37ab47e0cab011d7c06d8d2` |
| 4 | `3942c2637cbf7f14d2300dc003cf73678f853cec26ec75bd947814b298b63089` | `6ef334eada6649e88138e40db059d991df0036333fed08806614e85ddb3acaba` | `3bb4d2964b1256cf3de8d1a7a0b93bb7b69e9c3e8fd9af27eb103960d3da97cb` | `eeae4d9ec635f9525029f728e8bb08ca3006d0046a5b5c239aa10064be6e6979` |

The five model metadata JSON hashes, in seed order, are `ed3a4d23cc90ee6ea0e990d3c678e364078e2199521b1059b5c499145d85bdba`, `021724a49f0a0e40bd92acdfc848f946c1f7c66023d1586888190cadbb31f2b4`, `20a6d03d76b80a0e40923406880ca43deb2142dd6db01089186d3868c130e5c9`, `d23effd99b1cb1f8f5df5a97d4f761254914f7e10ea706dd93f49d70ffffde41`, and `1ff937da07e03ff51c9d4b2012632c93fbf950100866769e632e46a51e4e25c5`.

## Recovered training records

The local Codex session log at `%USERPROFILE%\.codex\sessions\2026\06\03\rollout-2026-06-03T21-53-19-019e8dc2-93dd-7472-9d6d-d5dd98775ff4.jsonl` records the three seed 0–2 command invocations on 2026-07-19. This log is outside the repository and is **not** included in the Git-tracked evidence package. Its timestamps are UTC; the local times below are UTC+08:00. All commands used `D:\研究生作业\组会4\generative-models-main` as their working directory:

```text
D:\anaconda\envs\wsy\python.exe -u -m GEFcom2014.models.QBM_VAE.qbm_vae --tag wind --sampler sa --learned-forecast-anchor --run-label lanchor --anchor-epochs 100 --anchor-patience 15 --epochs 80 --early-stopping-patience 20 --n-scenarios 100 --seed 0 --skip-plots
D:\anaconda\envs\wsy\python.exe -u -m GEFcom2014.models.QBM_VAE.qbm_vae --tag wind --sampler sa --learned-forecast-anchor --run-label lanchor --anchor-epochs 100 --anchor-patience 15 --epochs 80 --early-stopping-patience 20 --n-scenarios 100 --seed 1 --skip-plots
D:\anaconda\envs\wsy\python.exe -u -m GEFcom2014.models.QBM_VAE.qbm_vae --tag wind --sampler sa --learned-forecast-anchor --run-label lanchor --anchor-epochs 100 --anchor-patience 15 --epochs 80 --early-stopping-patience 20 --n-scenarios 100 --seed 2 --skip-plots
```

| Seed | Command log line | Local start time | Saved loss rows | Finite TEST-loss entries | Loss-file SHA-256 |
|---:|---:|---|---:|---:|---|
| 0 | 8375 | 11:08:19 | 80 | 0 | `43387e43fcdde2dca86403258379397a2944f5bebfbceb18c551468314883cf1` |
| 1 | 8512 | 11:17:40 | 76 | 0 | `92231ca3edaf14a45cb37cc81c7d0cbb9f387405e0c28e286d997d13af8a6e3e` |
| 2 | 8651 | 11:28:52 | 75 | 0 | `e9c341189d236d8a78e426fbf6b3a13bcc459f196e20dd737a6c1956b6a24604` |
| 3 | — | — | 80 | 0 | `a68b1ffd47ae861031088ff1ba9fdeed34d1fba45353c691275da2e398c6e9d7` |
| 4 | — | — | 80 | 0 | `227e1e6e528e56760db7238f9e2f190edb54660e268735d96f1d45a1d5fd186a` |

Each loss file is a three-column LS/VS/TEST history. The TEST column is entirely NaN in all five saved histories. Seeds 0 and 1 have successful command exit records (log lines 8491 and 8648). The seed 2 terminal wait timed out (line 8744), but its model and 75-row loss file were saved, and the same session later observed both VS and TEST scenario files (line 8777). Its final process exit code and full stdout were not retained; a timeout of the terminal wait is not evidence of a model-training exception.

## Training-path interpretation and remaining gaps

- Each saved model contains the same symmetric 48-by-48 graph mask: SHA-256 `02bc50d9075b00573e916392943ac3088d158abef0df361ba154f3e2e86c4f28`, 180 nonzero upper-triangle edges, upper-triangle weight sum `71.99713134765625`. Historical seed 0–2 JSONs say `graph_edges=71`, whereas seed 3–4 JSONs say `180`; the old field is inconsistent with the checkpoint and has not been rewritten. Its original calculation is not recorded.
- All five saved anchor checkpoints record `fit_split=LS` and `selection_split=VS`. The QBM/BM checkpoint selection is separately supported by the historical source record: `validation_score = vs_components["recon"].item()` updates `best_model` when the VS reconstruction loss improves, and the training script saves that `best_model` (session log line 628; current code retains the same rule). The selected best epoch was not saved and cannot be reconstructed from the total-loss columns alone.
- Seeds 3–4 explicitly record `negative_phase_backend=internal` and `training_backend=classical`; seeds 0–2 omit these metadata fields. For seeds 0–2, the run commands, pre-run source records (session lines 628, 5529, and 7693), and the later change introducing `--negative-phase-backend` (line 10420) together support a **high-confidence historical reconstruction**: the training negative phase used internal persistent Gibbs with `negative_steps=5` (also recorded in each model JSON), while `--sampler sa` controlled post-training scenario generation. This was not logged as an explicit per-run backend field, and the exact source commit used on 2026-07-19 cannot be recovered.
- The source-level checkpoint rule uses VS reconstruction, not TEST, and the all-NaN TEST-loss histories corroborate that TEST loss was not computed in these training histories. This supports `TEST not used for training-time checkpoint selection` for the observed code path; it does **not** independently audit whether a researcher inspected TEST results when choosing variants or planning later experiments. Do not generalize this evidence to a claim that TEST was never consulted at any stage.
- No exact 2026-07-19 Git commit, dependency-environment snapshot, or selected QBM/BM best epoch was recorded. Accordingly, the historical runs are documented and their saved outputs are identifiable, but this report does not claim bit-for-bit reproduction of the original training environment.

The Raw five-seed summary and machine-readable artifact audit are generated by `GEFcom2014.forecast_quality.aggregate_wind_five_seed` into `export/wind_five_seed_comparison/`. They are not physical-platform evidence. This report is an evidence inventory, **not** a claim that the old artifacts were reproduced from the Git commit containing this document.
