# A deep generative model for probabilistic energy forecasting in power systems: normalizing flows
Official implementation of generative models to compute scenario of renewable generation and consumption on the GEFcom2014 open dataset presented in the paper: A deep generative model for probabilistic energy forecasting in power systems: normalizing flows.
- [Applied Energy link until November 16, 2021](https://authors.elsevier.com/a/1dpj015eif0fZ5)
- [arXiv link](https://arxiv.org/abs/2106.09370)

## Cite

If you make use of this code, please cite our paper:

```
@article{DUMAS2022117871,
title = {A deep generative model for probabilistic energy forecasting in power systems: normalizing flows},
journal = {Applied Energy},
volume = {305},
pages = {117871},
year = {2022},
issn = {0306-2619},
doi = {https://doi.org/10.1016/j.apenergy.2021.117871},
url = {https://www.sciencedirect.com/science/article/pii/S0306261921011909},
author = {Jonathan Dumas and Antoine Wehenkel and Damien Lanaspeze and Bertrand Cornélusse and Antonio Sutera},
keywords = {Deep learning, Normalizing flows, Energy forecasting, Time series, Generative adversarial networks, Variational autoencoders},
abstract = {Greater direct electrification of end-use sectors with a higher share of renewables is one of the pillars to power a carbon-neutral society by 2050. However, in contrast to conventional power plants, renewable energy is subject to uncertainty raising challenges for their interaction with power systems. Scenario-based probabilistic forecasting models have become a vital tool to equip decision-makers. This paper presents to the power systems forecasting practitioners a recent deep learning technique, the normalizing flows, to produce accurate scenario-based probabilistic forecasts that are crucial to face the new challenges in power systems applications. The strength of this technique is to directly learn the stochastic multivariate distribution of the underlying process by maximizing the likelihood. Through comprehensive empirical evaluations using the open data of the Global Energy Forecasting Competition 2014, we demonstrate that this methodology is competitive with other state-of-the-art deep learning generative models: generative adversarial networks and variational autoencoders. The models producing weather-based wind, solar power, and load scenarios are properly compared in terms of forecast value by considering the case study of an energy retailer and quality using several complementary metrics. The numerical experiments are simple and easily reproducible. Thus, we hope it will encourage other forecasting practitioners to test and use normalizing flows in power system applications such as bidding on electricity markets, scheduling power systems with high renewable energy sources penetration, energy management of virtual power plan or microgrids, and unit commitment.}
}
```

Note: the reference will be changed if the paper is accepted for publication in Applied Energy

# Framework of the study
![strategy](https://github.com/jonathandumas/generative-models/blob/9549e0c301b448a749660ce716742ff928dc2778/figures/applied-energy-framework.png)

# Numerical experiments of the study
![numerical-experiments](https://github.com/jonathandumas/generative-models/blob/918ba080d82b04f541e2196a803165708f64fb73/figures/numerical-experiments-methodology.png)

# Dependencies
Two libraries are required to implement the normalizing flows models used in the study:
* https://github.com/AWehenkel/Normalizing-Flows -> to access the repositories lib and model
* https://github.com/AWehenkel/UMNN -> to implement the Unconstrained Monotonic Neural Networks normalizing flows

If you make use of the Unconstrained Monotonic Neural Networks code, please cite the paper:

```
@inproceedings{wehenkel2019unconstrained,
  title={Unconstrained monotonic neural networks},
  author={Wehenkel, Antoine and Louppe, Gilles},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1543--1553},
  year={2019}
}
```

Concerning the forecast value assessment: the Python Gurobi library is used to implement the algorithms in Python 3.7, and [Gurobi](https://www.gurobi.com/) 9.0.2 to solve all the optimization problems.


## Data
The GEFcom2014/data folder contains the GEFcom2014 dataset: load, wind, and PV tracks.


## Generative models
The GEFcom2014/models folder contains the generative models in the following folders:
* GAN: generative adversarial networks
* VAE: variational autoencoders
* NFs: normalizing flows (affine and unconstrained monotonic neural networks)
* GC: gaussian copula
* RAND: random

Then, inside each folder, there is a Python file to run the model. For instance, for the NFs:

```bash
nf_autoregressive.py 
```

For each model, the structure of the Python file is similar. 

First, select the track by indicating the track considered:

```bash
tag = 'load'  # pv, wind, load
```

Second, specify the number of scenarios and quantiles to be generated per day: 

```bash
n_s = 100 # number of scenarios
N_q = 99 # number of quantiles
```

The hyper-parameters of the models for each track are already specified but can be modified. By default, they are set to the values of the paper.

Finally, by running the Python file, the model is trained and generates scenarios and quantiles over the learning, validation, and testing set.

Note: the function

```bash
quantiles_and_evaluation(dir_path=dir_path, s_VS=s_VS, s_TEST=s_TEST, N_q=N_q, df_y_VS=df_y_VS, df_y_TEST=df_y_TEST, name=name, ymax_plf=ymax_plf, ylim_crps=ylim_crps, tag=tag, nb_zones=nb_zones)
```
allows computing some metrics over the validation and testing sets.

## Forecast quality
The GEFcom2014/forecast_quality folder contains the Python files to evaluate the forecast quality of the generative models. The scenarios to be evaluated are located in the folder GEFcom2014/forecast_quality/scenarios. This folder contains the scenarios per generative model: GAN, GC, NFs, etc. 

For instance, GEFcom2014/forecast_quality/scenarios/gan comprises scenarios over the load, wind, and PV tracks over the testing set.

The file:
```bash
CRPS_QS_all_tracks.py
```
allows computing the CRPS, QS, and reliability diagram metrics for all tracks and models.

The file:
```bash
CRPS_QS_DM_test.py
```
computes DM-statistical test of the CRPS and QS metrics for all models and tracks.

The file:
```bash
ES_VS_metrics.py 
```
computes the ES, VS multivariate metrics for all tracks and models. It also computes the DM-statistical test.

The file:
```bash
scenario_correlations.py
```
computes the correlation matrices between scenarios for a given day.

The file:
```bash
clf_metric.py
```
contains the classifier-based metric. Note: scenarios over the learning set are required to fit the classifier. Due to the size of the files, they are not included in the GEFcom2014/forecast_quality/scenarios folder. They need to be generated and included in this folder.

## Forecast value
The GEFcom2014/forecast_value/ folder contains the Python files to evaluate the forecast value of the generative models. It uses the retailer energy case study to bid optimally on a day-ahead basis by solving a stochastic optimization problem.

The file:
```bash
bidding_retailer.py 
```
computes the day-ahead bids for all models and the corresponding profits.

The parameters:
```bash
nb_s = 50 # number of scenarios considered per stochastic optimization problem
soc_max = 1 # battery storage capacity
dad_price = 100  # euros /MWh
q_pos = 2
q_neg = 2
pos_imb = q_pos * dad_price  # euros /MWh
neg_imb = q_neg * dad_price  # euros /MWh
```
are set to the default values used in the paper.

Warning: the Gurobi Python API must be activated to use this file.

## FA-BM-VAE experiment extension

The current main experiment is **FA-BM-VAE**: Forecast Anchor supplies the
deterministic main trend, while a classical conditional BM/VAE models residual
uncertainty and scenarios. The internal Gibbs negative phase is classical;
the optional bosonic/SPQC adapter replaces only the frozen post-training
latent sampling step. The historical `models/QBM_VAE` directory, class names,
and old QBM-VAE artifacts remain unchanged for compatibility. The
path-integral transverse-field implementation is retained as a historical
ablation branch and is not part of the default comparison.

The paper configurations are frozen as JSON files. Explicit command-line
arguments, such as `--seed`, override the corresponding JSON default:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.qbm_vae --config configs\paper\fa_bm_vae_wind.json --seed 0
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.qbm_vae --config configs\paper\fa_bm_vae_opsd_wind.json --seed 0
```

Replace the internal persistent-Gibbs negative phase with the explicit
classical SA control while leaving the generation backend unchanged:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.qbm_vae --config configs\paper\fa_bm_vae_wind.json --negative-phase-backend sa --negative-sampler-sweeps 20 --run-label lanchor_sa_neg --seed 0
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.qbm_vae --config configs\paper\fa_bm_vae_opsd_wind.json --negative-phase-backend sa --negative-sampler-sweeps 20 --run-label anchor_sa_neg --seed 0
```

Validate small Ising problems against their exact Boltzmann laws before using
an external sampler:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.validate_ising_samplers --n-bits 12 --num-reads 5000 --seeds 0 1 2
```

Train the three strong conditional baselines on Wind and export 100 validation
and test scenarios:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.probabilistic_baselines --model spline-nf --tag wind --epochs 200 --n-scenarios 100
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.probabilistic_baselines --model ddpm --tag wind --epochs 200 --diffusion-steps 100 --inference-steps 100 --n-scenarios 100
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.anchor_score_sde --tag wind --epochs 200 --sampling-steps 32 --n-scenarios 100 --seed 0
```

Retrain the original CVAE, WGAN-GP, and UMNN architectures without inspecting
TEST during optimization. Each command uses VS-only checkpoint selection and
exports paired VS/TEST scenarios under a new directory, leaving the historical
TEST-only artifacts unchanged:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.legacy_baselines --model cvae --tag wind --seed 0 --n-scenarios 100
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.legacy_baselines --model wgan-gp --tag wind --seed 0 --n-scenarios 100
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.legacy_baselines --model umnn --tag wind --seed 0 --n-scenarios 100 --umnn-day-batch-size 8
```

UMNN writes a best-state checkpoint whenever VS NLL improves and stores each
scenario batch as a resumable NumPy chunk. If scenario generation is
interrupted after training, resume it without retraining:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.legacy_baselines --model umnn --tag wind --seed 0 --sample-only --n-scenarios 100 --umnn-day-batch-size 8
```

Apply the same validation-only calibration and LS-template temporal rank
coupling to all models with complete VS/TEST scenario pairs:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.unified_postprocessing --tag wind --reuse-completed
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.aggregate_unified_results --tag wind --bootstrap-repetitions 20000 --permutation-repetitions 20000
```

Build the QBM temporal dependence scenarios and export interpretable AR/Ising
summaries with:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag wind --seeds 0 1 2
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.qbm_temporal_diagnostics --tag wind --seeds 0 1 2
```

The protocol exports `Raw`, `Cal`, `TRC`, and `Cal+TRC` results under
`export/unified_postprocessing/<track>/`. Calibration is fitted only on VS;
TEST is evaluation-only. The manifest lists skipped legacy models whose
historical artifacts contain TEST scenarios but no matching VS scenarios.
The aggregation command first selects each model family's post-processing
variant from mean multi-seed VS CRPS, then reports TEST mean and standard
deviation. Paired inference uses 50 date blocks aggregated across the ten Wind
zones, with block-bootstrap confidence intervals and paired sign permutations.

### D3U and Treeffuser strong baselines

The frozen comparison suite now includes two additional published model
families. `D3U (NWP-adapted)` follows the ICLR 2025 decomposition with an
LS-trained deterministic predictor, a frozen condition representation,
residual diffusion, and a patch-based AdaLN-Zero denoising transformer. The
adaptation is explicit: future NWP replaces the historical-window conditioner
used by the official D3U implementation. `Treeffuser` uses the official
`treeffuser==0.2.0` implementation and jointly samples all 24 target periods;
its LightGBM early-stopping holdout is drawn from LS only.

The Treeffuser paper configuration is CPU-budgeted for this repository:
`n_repeats=5`, `n_estimators=500`, and `n_jobs=4`. It still uses the official
Treeffuser 0.2.0 implementation. Run the frozen five-seed experiment and then
apply the same validation-only calibration/TRC pipeline with:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.run_strong_baselines --seeds 0 1 2 3 4 --evaluate --repetitions 2000
```

Paper configurations are stored in `configs/paper/d3u_*.json` and
`configs/paper/treeffuser_*.json`. Raw joint scenarios remain the primary
endpoint; calibrated and TRC variants are sensitivity analyses. The completed
five-seed audit is summarized below; inference used 2,000 block-bootstrap and
paired-permutation repetitions.

| Dataset | Model | Selected variant | CRPS mean +/- std | Energy | Variogram | Ramp quantile MAE |
|---|---|---:|---:|---:|---:|---:|
| GEFCom Wind | D3U (NWP-adapted) | Raw | 0.083755 +/- 0.000656 | 0.522994 | 0.028750 | 0.009417 |
| GEFCom Wind | Treeffuser | Raw | 0.089749 +/- 0.000349 | 0.556357 | 0.030517 | 0.033188 |
| OPSD Wind | D3U (NWP-adapted) | Cal | 0.028797 +/- 0.000213 | 0.170063 | 0.006681 | 0.010027 |
| OPSD Wind | Treeffuser | Raw | 0.033830 +/- 0.001127 | 0.210895 | 0.010537 | 0.049089 |

The machine-readable outputs are under
`export/unified_postprocessing/wind/` and
`export/unified_postprocessing/opsd-wind/`. These results show that the
current FA-BM-VAE/Score-SDE models remain stronger on aggregate CRPS than the
new baselines; no universal D3U or Treeffuser advantage is claimed.

### Current Wind comparison

The following TEST results use three seeds. A single post-processing variant is
selected for each family from mean VS CRPS before TEST is read.

| Model | Variant | CRPS mean +/- std | Reliability MAE | Energy | Variogram | Ramp quantile MAE |
|---|---:|---:|---:|---:|---:|---:|
| Forecast-anchored Gaussian mixture (K=4) + Temporal ECC | Raw | 0.082830 +/- 0.000340 | 2.2869 | 0.515363 | 0.028377 | 0.012581 |
| Forecast-anchored Gaussian mixture (K=4) | Raw | 0.082830 +/- 0.000340 | 2.2869 | 0.517709 | 0.029075 | 0.053090 |
| Forecast-anchored conditional Score-SDE + Temporal ECC | Cal | 0.082841 +/- 0.000472 | 2.6311 | 0.514827 | 0.028265 | 0.016107 |
| Forecast-anchored conditional Score-SDE | Cal | 0.082841 +/- 0.000472 | 2.6311 | 0.515838 | 0.028463 | 0.020539 |
| Forecast-anchored conditional Gaussian + Temporal ECC | Raw | 0.083229 +/- 0.000373 | 1.6118 | 0.516269 | 0.028421 | 0.013426 |
| Forecast-anchored conditional Gaussian | Raw | 0.083229 +/- 0.000373 | 1.6118 | 0.522471 | 0.030117 | 0.074914 |
| FA-BM-VAE (internal Gibbs negative phase), learned anchor + Temporal ECC | Cal | 0.083382 +/- 0.000493 | 2.3172 | 0.516077 | 0.028325 | 0.016726 |
| FA-BM-VAE (internal Gibbs negative phase), learned anchor | Cal | 0.083382 +/- 0.000493 | 2.3172 | 0.519990 | 0.029848 | 0.073390 |
| FA-BM-VAE, learned anchor, SA negative phase + Temporal ECC | Cal | 0.083418 +/- 0.000157 | 2.5301 | 0.516416 | 0.028325 | 0.016950 |
| FA-BM-VAE, learned anchor, SA negative phase | Cal | 0.083418 +/- 0.000157 | 2.5301 | 0.520184 | 0.029832 | 0.073065 |
| Forecast-anchored conditional Spline Flow + Temporal ECC | Raw | 0.084502 +/- 0.000487 | 2.1184 | 0.523473 | 0.028894 | 0.016244 |
| Forecast-anchored conditional Spline Flow | Raw | 0.084502 +/- 0.000487 | 2.1184 | 0.523545 | 0.028938 | 0.012563 |
| Spline Conditional NF | Raw | 0.085414 +/- 0.000511 | 1.5001 | 0.532772 | 0.029116 | 0.011720 |
| Conditional DDPM | Cal | 0.086495 +/- 0.000172 | 1.8284 | 0.535175 | 0.029136 | 0.018591 |
| Legacy UMNN, VS retrained | Cal | 0.087593 +/- 0.000403 | 2.2781 | 0.551425 | 0.032498 | 0.090150 |
| QBM-VAE + Temporal ECC | Cal | 0.088051 +/- 0.000234 | 3.7324 | 0.541133 | 0.028939 | 0.020806 |
| Legacy CVAE, VS retrained | Raw | 0.088136 +/- 0.000429 | 2.5059 | 0.548328 | 0.031183 | 0.029085 |
| Legacy WGAN-GP, VS retrained | Cal | 0.090864 +/- 0.001837 | 1.4485 | 0.559544 | 0.031362 | 0.022848 |

Date-block paired inference shows that QBM-ECC has worse CRPS than Spline NF
(mean difference 0.002637, p=0.0004) and worse Energy score (difference
0.008361, p=0.0383). Its Variogram and ramp-CRPS differences from Spline NF are
not significant. Against legacy UMNN, QBM-ECC has statistically
indistinguishable CRPS (p=0.735) but significantly better Variogram and
ramp-CRPS (both p<0.0001). These results support a temporal-dependence
contribution, not a claim of overall superiority or quantum advantage.

For GEFCom Wind, which has no target-scale point forecast in its context, the
learned-anchor variant first fits a deterministic `250 -> 24` MLP on LS with
VS-only early stopping. Its predictions are frozen and appended as the QBM
decoder anchor. Across three seeds, learned-anchor QBM+ECC significantly
outperforms Spline NF in CRPS (difference -0.002032, p=0.0166), Energy
(p=0.00035), Variogram (p=0.00345), and ramp CRPS (p=0.0129). It also
significantly outperforms Conditional DDPM in all four scores. The anchor-only
point forecast has TEST CRPS around `0.118`, so the probabilistic improvement
cannot be attributed to the deterministic MLP alone.

The matched conditional-Gaussian residual ablation reuses the same frozen
learned anchor and the same Temporal-AR ECC templates. Its CRPS is slightly
lower than learned-anchor QBM+ECC, but the difference is not significant
(`-0.000153`, `p=0.654` for Gaussian minus QBM). Energy and Variogram
differences are also not significant. Learned-anchor QBM+ECC does have
significantly lower daily ramp CRPS (`-0.000458`, `p<0.001` for QBM minus
Gaussian). However, a stronger four-component conditional Gaussian mixture
obtains CRPS `0.082830` and removes that ramp advantage. Its paired differences
from QBM+ECC are not significant for CRPS (`p=0.0906`), Energy (`p=0.679`),
Variogram (`p=0.712`), or ramp CRPS (`p=0.651`). Thus, the GEFCom evidence
attributes most aggregate improvement to forecast anchoring, multimodal
residual learning, and dependence reconstruction rather than uniquely to the
Boltzmann prior.

The forecast-anchored rational-quadratic Spline Flow is a stronger invertible
residual-density baseline using the same frozen anchor. Learned-anchor QBM+ECC
significantly outperforms Flow+ECC in CRPS (`-0.001120`, `p=0.0134`), Energy
(`p=0.00225`), Variogram (`p=0.0050`), and ramp CRPS (`p<0.001`). This supports
a robust advantage over the matched conditional Flow, while the Gaussian and
mixture ablations still prevent interpreting it as a universal Boltzmann-prior
or quantum advantage.

The forecast-anchored conditional Score-SDE models the same 24-hour residual
target with continuous-noise EDM training, EMA validation checkpoints, and a
32-step deterministic Heun sampler. Score-SDE+ECC significantly improves over
the original Conditional DDPM in CRPS, Energy, Variogram, and ramp CRPS
(`p<0.001` for CRPS and ramp CRPS). It also significantly outperforms the
matched anchored Spline Flow+ECC in all four scores. Relative to learned-anchor
QBM+ECC, its CRPS difference is `-0.000541` but is not significant
(`p=0.0971`); Energy and Variogram are also tied, while Score-SDE has lower
ramp CRPS (`p=0.00015`). Thus, the strengthened diffusion baseline is
competitive with the classical FA-BM-VAE mainline rather than evidence that
either family universally dominates.

Replacing the internal persistent-Gibbs negative phase with a 20-sweep
classical SA backend leaves the GEFCom result unchanged within sampling
uncertainty. SA-negative+ECC obtains CRPS `0.083418`, compared with `0.083382`
for internal-Gibbs+ECC. Their paired differences are nonsignificant for CRPS
(`p=0.847`), Energy (`p=0.758`), Variogram (`p=0.996`), and ramp CRPS
(`p=0.680`). This establishes SA as a valid classical control for the future
CIM experiment; it is not a quantum result.

### Forecast-anchor x Ising-coupling ablation

The matched `2 x 2` ablation separates the learned forecast anchor from the
Temporal-MI Ising couplings `J`. All four cells use the same latent dimension,
network capacity, training schedule, SA generation sampler, and 100-scenario
budget. Run or resume all three seeds and then evaluate them with:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.run_anchor_coupling_ablation --seeds 0 1 2 --evaluate --repetitions 20000
```

All variants below were selected as `Cal` from mean three-seed VS CRPS. TEST
was not used for selection.

| Forecast anchor | Temporal-MI J | TEST CRPS | Energy | Variogram | Ramp quantile MAE |
|---|---|---:|---:|---:|---:|
| No | No | 0.088013 | 0.545185 | 0.030977 | 0.084454 |
| No | Yes | 0.088030 | 0.544224 | 0.030402 | 0.075554 |
| Learned | No | 0.083408 | 0.520469 | 0.030044 | 0.077359 |
| Learned | Yes | 0.083382 | 0.519990 | 0.029848 | 0.073390 |

Paired inference uses 50 date blocks averaged over seeds, 20,000 block
bootstrap repetitions, 20,000 paired sign permutations, and one Holm
correction across the 20 reported tests. Adding `J` does not significantly
change CRPS or Energy. It significantly reduces Variogram and ramp CRPS both
without an anchor (`-0.000575` and `-0.002005`) and with an anchor
(`-0.000195` and `-0.000972`; all Holm-adjusted `p<=0.003`). Adding the anchor
significantly reduces CRPS by about `0.0046` and Energy by about `0.024`.
Positive, significant interaction terms for Variogram (`0.000379`) and ramp
CRPS (`0.001033`) show diminishing rather than additive dependence gains.
Thus, under VS-selected marginal calibration, the anchor primarily improves
marginal sharpness and calibration, while adding Ising couplings changes the
temporal scores. This single-dataset calibrated result does not by itself show
that the Temporal-MI topology is better than another graph with the same
sparsity and coupling-weight distribution.

### Cross-dataset Raw graph-structure control

The stricter graph ablation compares `J=0`, a same-edge/same-weight random
graph, and the Temporal-MI graph on GEFCom Wind and OPSD Wind. `Raw` scenarios
are the primary endpoint; `Cal` is retained only as a sensitivity analysis, and
TRC/ECC is excluded. Run or resume the nine missing three-seed cells and the
20,000-repetition inference with:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.run_graph_structure_ablation --seeds 0 1 2 --evaluate --repetitions 20000
```

Negative differences favor Temporal-MI. Holm correction is applied across the
eight primary dependence tests.

| Dataset | Raw contrast | Variogram diff. (Holm p) | Ramp CRPS diff. (Holm p) | Seeds favoring Temporal-MI |
|---|---|---:|---:|---:|
| GEFCom Wind | Temporal-MI - `J=0` | -0.000126 (0.205) | -0.000021 (1.000) | 3/3, 2/3 |
| GEFCom Wind | Temporal-MI - random | -0.000025 (1.000) | +0.000177 (0.0014) | 2/3, 1/3 |
| OPSD Wind | Temporal-MI - `J=0` | -0.000004 (1.000) | -0.000137 (0.0004) | 1/3, 3/3 |
| OPSD Wind | Temporal-MI - random | +0.000006 (1.000) | -0.000014 (1.000) | 2/3, 2/3 |

The predeclared robustness criterion is not met. Temporal-MI significantly
improves Raw ramp CRPS over `J=0` on OPSD, but it is significantly worse than
the matched random graph on GEFCom ramp CRPS and has no significant Variogram
advantage over random on either dataset. The current evidence therefore
supports the forecast anchor and the BM family as modeling components, but not
a general performance claim for the Temporal-MI topology. Graph-mask audits,
date-block intervals, adjusted p-values, and seed-level directions are under
`export/graph_structure_ablation/`.

### VS-only graph and trajectory-objective development

Three follow-up candidates were evaluated without reading or generating TEST
scenarios. Development used seeds 0--2, Raw scenarios, a fixed 100-member
ensemble, and a predeclared gate requiring lower Variogram/ramp dependence
error on both Wind datasets, at least two favorable seeds per dataset, and no
metric regression above 1%.

1. A bootstrap-stable residual Temporal-MI graph failed against the existing
   Temporal-MI graph on both datasets (`0/3` favorable seeds each).
2. A bootstrap-stable posterior partial-correlation graph failed on GEFCom
   Wind (`0/3`) and was not consistent on OPSD (`1/3`).
3. A differentiable trajectory-score objective showed promising mean OPSD
   development scores, but the seed-direction gate failed (`1/3`).

The third candidate was therefore tested once with predeclared, independent
seeds 3--5 against a paired control that differed only in the trajectory loss.
It passed the GEFCom Wind gate (`2/3`; dependence composite change `-0.106%`),
including lower ramp CRPS (`-0.000181`, 95% CI `[-0.000298, -0.000072]`, paired
permutation `p=0.0030`, Holm-adjusted `p=0.0210`). It failed on OPSD Wind (`1/3`; dependence composite
change `+0.712%`), where ramp CRPS increased by `0.000133` (95% CI
`[0.000060, 0.000205]`, raw `p=0.00065`, Holm-adjusted `p=0.00520`). The candidate is therefore not promoted
to the main model and no post-failure hyperparameter search is performed.
Machine-readable gates and paired inference are stored under
`export/stable_residual_graph_selection/`, `export/posterior_graph_selection/`,
`export/trajectory_score_selection/`, and
`export/trajectory_score_confirmation/`. All four records set
`test_scenarios_generated=false`.

### Hardware-facing Ising protocol

The learned-anchor FA-BM-VAE can now export condition-matched Ising instances
for disjoint VS and TEST splits. Effective temperature is fitted jointly over
VS responses only; the resulting coefficient scale is frozen before TEST
payloads are generated. The same-instance audit compares energy distributions,
magnetization, edge moments, effective temperature, state diversity, and
latency, with paired bootstrap intervals and sign-flip tests.

A 20-instance numerical pilot injected `beta=0.65`. The VS estimator
recovered `0.6500` (95% CI `[0.6450, 0.6550]`) and selected scale `1.5385`.
This historical head-selected pilot covered one Wind zone only. On its 20 TEST
instances, mean effective beta changed from `0.6479`
to `1.0002`, while the target-temperature reference was `0.9968`. Energy
Wasserstein distance to the reference decreased from `2.9385` to `0.1220`;
magnetization and edge-moment MAE decreased from `0.1110/0.0916` to
`0.0224/0.0245` (all paired `p <= 1e-4`). This validates the calibration
protocol under simulation, not a physical-platform or quantum advantage
claim. Full commands and response contracts are documented in
`GEFcom2014/models/QBM_VAE/README.md`.

The corrected hardware protocol uses a 50-instance VS panel balanced across
all ten Wind zones and the complete 500-instance TEST pool. Its hash-checked
end-to-end SA replay recovered `beta_eff=1.0000` with 95% CI
`[0.9950, 1.0000]`. The real VS submission archive is frozen, while TEST
packaging is deliberately blocked until responses from the physical platform
have produced a new VS-only calibration.

### FA-BM-VAE bosonic/SPQC sampling chain

The current hardware-facing method is named **FA-BM-VAE**. Forecast Anchor
provides the deterministic main trend, while the conditional classical Ising
BM/VAE models residual uncertainty and scenario dependence. Training remains
classical. A bosonic SPQC/CIM platform is used only after the model is frozen,
replacing the conditional Ising latent sampling step:

```text
classical FA-BM-VAE training
    -> condition-matched h(x), J export
    -> VS-only platform temperature calibration
    -> frozen TEST platform responses
    -> imported spins
    -> frozen FA-BM-VAE decoder
    -> reconstructed scenarios and forecast metrics
```

This repository does not claim complete QBM training or quantum advantage from
this workflow. `models/QBM_VAE` and artifact names containing `QBMVAE_2` are
historical compatibility names only. To reconstruct scenarios from canonical
imported responses, run:

```powershell
python -m GEFcom2014.models.QBM_VAE.reconstruct_hardware_scenarios `
  --tag wind `
  --model-name wind_QBMVAE_2_lanchor_sa_0 `
  --manifest export\bosonic_calibration\wind_lanchor_seed0\test_calibrated_manifest.json `
  --responses-dir export\bosonic_responses\wind_lanchor_seed0_real_test `
  --output-dir export\fa_bm_vae_hardware\wind_test
```

The generated pickle keeps the existing scenario-array layout and is paired
with JSON metadata naming **FA-BM-VAE**, recording the platform backend,
frozen-model path, source manifest, response audit, and `hardware_claim=false`.

For the Kaiwu/SPQC 8-bit path, provide one global gain explicitly when
exporting. The exporter creates `M`, `Q=round(gain*M)`, a separate 49x49 matrix
file, and a quantization audit; it never clips or rescales instances
individually:

```powershell
python -m GEFcom2014.models.QBM_VAE.export_ising_instances `
  --tag wind --model-name wind_QBMVAE_2_lanchor_sa_0 `
  --split VS --selection stratified --num-instances 10 `
  --hardware-gain 100 `
  --output-dir export\kaiwu_instances\fa_bm_vae_wind_gain100

python -m GEFcom2014.models.QBM_VAE.prepare_bosonic_submission `
  --manifest export\kaiwu_instances\fa_bm_vae_wind_gain100\wind_wind_QBMVAE_2_lanchor_sa_0_vs_manifest.json `
  --stage vs-calibration --requested-reads 100 `
  --output-dir export\kaiwu_submissions\fa_bm_vae_wind_gain100_vs
```

The value `100` is a reproducible local preflight candidate, not a frozen
hardware temperature. Real VS calibration must test several global gains and
freeze one only after the original `h,J` models and all selected VS instances
pass the int8 range and sampling diagnostics.

The Kaiwu response must contain 49 raw spins, the instance ID, a task ID, the
matrix SHA-256, and index-ascending bit order. The importer stores both
`hardware_samples` (49 columns) and normalized logical `samples` (48 columns).
The latter is the only representation consumed by benchmark and scenario
reconstruction.

After the platform environment is configured externally, submit one matrix
explicitly with:

```powershell
python -m GEFcom2014.models.QBM_VAE.submit_kaiwu_sampling `
  --matrix-file <package>\hardware_matrices\<instance_id>.npz `
  --instance-id <instance_id> --num-reads 100 `
  --output-dir export\kaiwu_raw_responses
```

This is the only command in the repository that can submit a Kaiwu task. It
uses a unique task name and `CIMOptimizer` sampling mode; it requires
`KAIWU_PROJECT_NO` or an explicit runtime `--project-no`.

For every hardware run, generate both controls on the identical frozen
instances:

```powershell
python -m GEFcom2014.models.QBM_VAE.sample_exported_ising `
  --manifest <manifest.json> --backend sa --matrix-space logical `
  --num-reads 1000 --output-dir export\kaiwu_controls

python -m GEFcom2014.models.QBM_VAE.sample_exported_ising `
  --manifest <manifest.json> --backend sa --matrix-space hardware-quantized `
  --num-reads 1000 --output-dir export\kaiwu_controls
```

These are respectively floating-point logical SA, the same quantized 8-bit
matrix sampled by local SA, and the physical Kaiwu/SPQC response. Differences
between the latter two are the platform sampling effect, while differences
between the first two are quantization effects.

Kaiwu integration is optional and lazy. The runtime environment must provide
the SDK license and project number externally, for example through
`KAIWU_PROJECT_NO`; no account, license, project number, or key is stored in
the repository. The adapter uses `CIMOptimizer` in `SAMPLING` mode with a
unique task name. No task is submitted by repository tests.

### Independent OPSD Wind validation

The external validation uses the fixed Open Power System Data Time Series
release `2019-06-05` (DOI `10.25832/time_series/2019-06-05`). The source file
is verified before preprocessing with SHA-256
`659fe789af2672aabe989aebc8c5c21052a1a96e4da70b0fc941910a1cd4de9d`.
The experiment uses 50Hertz onshore day-ahead forecasts as context and
onshore generation as the 24-hour target. Scaling is fitted on LS only.

Build the daily bundle from the official hourly CSV:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.external_datasets `
  --source-csv GEFcom2014\data\external\opsd_time_series_60min_2019-06-05.csv `
  --output GEFcom2014\data\external\opsd_wind_daily.npz `
  --validation-days 90 --test-days 90 --scale-quantile 1.0
```

The chronological split contains 1,036 LS days (`2016-01-01` to
`2018-11-01`), 90 VS days (`2018-11-02` to `2019-01-30`), and 90 TEST days
(`2019-01-31` to `2019-04-30`). Train each command with seeds 0, 1, and 2:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.probabilistic_baselines --model spline-nf --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --epochs 200 --n-scenarios 100 --seed 0
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.probabilistic_baselines --model ddpm --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --epochs 200 --n-scenarios 100 --seed 0
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.qbm_vae --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --sampler sa --epochs 80 --n-scenarios 100 --seed 0 --skip-plots
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.qbm_vae --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --sampler sa --decoder-covariance ar1 --run-label tar1m --epochs 80 --n-scenarios 100 --seed 0 --skip-plots
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.qbm_vae --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --sampler sa --forecast-anchor --run-label anchor --epochs 80 --n-scenarios 100 --seed 0 --skip-plots
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --anchor-source context --epochs 80 --n-scenarios 100 --seed 0
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --anchor-source context --components 4 --epochs 80 --n-scenarios 100 --seed 0
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.anchor_spline_flow --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --anchor-source context --epochs 200 --n-scenarios 100 --seed 0
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.anchor_score_sde --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --anchor-source context --epochs 200 --sampling-steps 32 --n-scenarios 100 --seed 0
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.probabilistic_baselines --model residual-bootstrap --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --n-scenarios 100 --seed 0
```

Build QBM-ECC and evaluate all complete three-seed pairs:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --seeds 0 1 2
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --source-label anchor --template-label tar1m --output-label QBMAnchorECC --seeds 0 1 2
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --source-dir export\anchor_gaussian_opsd-wind --source-version AnchorGaussian --template-dir export\qbm_vae_opsd-wind --template-label tar1m --output-dir export\anchor_gaussian_ecc_opsd-wind --output-label AnchorGaussianECC --seeds 0 1 2
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --source-dir export\anchor_gmm4_opsd-wind --source-version AnchorGMM4 --template-dir export\qbm_vae_opsd-wind --template-label tar1m --output-dir export\anchor_gmm4_ecc_opsd-wind --output-label AnchorGMM4ECC --seeds 0 1 2
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --source-dir export\anchor_spline_flow_opsd-wind --source-version AnchorSplineFlow --template-dir export\qbm_vae_opsd-wind --template-label tar1m --output-dir export\anchor_spline_flow_ecc_opsd-wind --output-label AnchorSplineFlowECC --seeds 0 1 2
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --source-dir export\anchor_score_sde_opsd-wind --source-version AnchorScoreSDE --template-dir export\qbm_vae_opsd-wind --template-label tar1m --output-dir export\anchor_score_sde_ecc_opsd-wind --output-label AnchorScoreSDEECC --seeds 0 1 2
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.unified_postprocessing --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.forecast_quality.aggregate_unified_results --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --bootstrap-repetitions 20000 --permutation-repetitions 20000
```

All selected variants are chosen by mean three-seed VS CRPS before TEST is
evaluated. Current TEST results are:

| Model | Variant | CRPS mean +/- std | Energy | Variogram | Ramp quantile MAE |
|---|---:|---:|---:|---:|---:|
| FA-BM-VAE, SA negative phase + Temporal ECC | Cal | 0.027284 +/- 0.000088 | 0.162616 | 0.006138 | 0.006204 |
| FA-BM-VAE, SA negative phase | Cal | 0.027284 +/- 0.000088 | 0.164684 | 0.006767 | 0.031497 |
| Forecast-anchored conditional Gaussian + Temporal ECC | Cal | 0.027432 +/- 0.000060 | 0.164115 | 0.006202 | 0.006500 |
| Forecast-anchored conditional Gaussian | Cal | 0.027432 +/- 0.000060 | 0.166771 | 0.007068 | 0.034229 |
| FA-BM-VAE (internal Gibbs negative phase) + Temporal ECC | Cal | 0.027529 +/- 0.000302 | 0.164603 | 0.006237 | 0.006371 |
| FA-BM-VAE (internal Gibbs negative phase) | Cal | 0.027529 +/- 0.000302 | 0.166817 | 0.006894 | 0.032296 |
| Forecast-anchored conditional Score-SDE + Temporal ECC | Cal | 0.027655 +/- 0.000143 | 0.164993 | 0.006273 | 0.005879 |
| Forecast-anchored conditional Score-SDE | Cal | 0.027655 +/- 0.000143 | 0.164222 | 0.006209 | 0.006779 |
| Forecast-anchored conditional Spline Flow + Temporal ECC | Cal | 0.028309 +/- 0.000573 | 0.169070 | 0.006515 | 0.009145 |
| Forecast-anchored conditional Spline Flow | Cal | 0.028309 +/- 0.000573 | 0.168422 | 0.006458 | 0.009007 |
| Daily Residual Bootstrap | Cal | 0.028546 +/- 0.000020 | 0.168656 | 0.006382 | 0.006028 |
| Conditional DDPM | Cal | 0.028670 +/- 0.000146 | 0.168434 | 0.006471 | 0.011206 |
| Forecast-anchored Gaussian mixture (K=4) + Temporal ECC | Cal | 0.029348 +/- 0.000093 | 0.175280 | 0.006786 | 0.008476 |
| Forecast-anchored Gaussian mixture (K=4) | Cal | 0.029348 +/- 0.000093 | 0.176666 | 0.007718 | 0.032599 |
| QBM-VAE + Temporal ECC | Cal | 0.030361 +/- 0.000426 | 0.183137 | 0.007676 | 0.008199 |
| QBM-VAE | Cal | 0.030361 +/- 0.000426 | 0.183837 | 0.008276 | 0.035747 |
| Spline Conditional NF | Cal | 0.031934 +/- 0.000277 | 0.188136 | 0.007470 | 0.011256 |

FA-BM-VAE (internal Gibbs negative phase) uses the supplied day-ahead wind forecast as a fixed decoder
location anchor and learns the conditional stochastic residual. ECC preserves
its hourly marginals, so QS, CRPS, and reliability remain identical by
construction. Relative to Conditional DDPM, FA-BM-VAE+ECC has significantly
lower CRPS (difference -0.001142, p=0.0055) and ramp CRPS; its lower Energy and
Variogram point estimates are not significant. It also has significantly lower
CRPS than the LS-only daily residual bootstrap (difference -0.001018,
p=0.0058). These results isolate gains beyond merely reusing the point
forecast. They do not establish quantum advantage: the prior and sampler in
this experiment are classical Ising/SA. Full outputs are written to
`export/unified_postprocessing/opsd-wind/`.

Against the matched Gaussian+ECC ablation, FA-BM-VAE+ECC has no significant
difference in CRPS (`p=0.688`), Energy (`p=0.693`), or Variogram (`p=0.464`).
Its lower ramp CRPS is suggestive but not significant (`p=0.0776`). This
independent result does not support a general Boltzmann-prior advantage; it
supports retaining the prior contribution as a targeted ramp-dynamics
hypothesis requiring further validation.

The fixed `K=4` Gaussian mixture is significantly worse than FA-BM-VAE+ECC
on OPSD CRPS, Energy, Variogram, and ramp CRPS (all `p<0.001`). This shows that
FA-BM-VAE is more robust than this over-parameterized multimodal alternative,
but it does not establish a unique Boltzmann advantage because the simpler
conditional Gaussian remains statistically tied with FA-BM-VAE.

FA-BM-VAE+ECC also significantly outperforms the matched anchored Spline
Flow+ECC on OPSD CRPS (`-0.000781`, `p=0.0027`), Energy (`p=0.0032`),
Variogram (`p=0.0143`), and ramp CRPS (`p<0.001`). Together with the GEFCom
result, this establishes a reproducible advantage over a high-capacity
conditional normalizing-flow residual model, but not over every classical
anchored residual distribution.

On OPSD, Score-SDE+ECC significantly improves CRPS over the original
Conditional DDPM (`-0.001015`, `p=0.0405`) and improves ramp CRPS
(`p<0.001`). Its CRPS, Energy, and Variogram differences from FA-BM-VAE+ECC
are not significant (`p=0.6295`, `0.7903`, and `0.6447`, respectively), while
its ramp CRPS is significantly lower (`p<0.001`). This independent validation
confirms that the stronger diffusion comparison closes the aggregate
performance gap without invalidating the structured BM result.

The SA-negative experiment is independently reproducible on OPSD. Relative to
the internal-Gibbs FA-BM-VAE+ECC, it improves CRPS by `0.000245`
(`p=0.0434`), Energy by `0.001987` (`p=0.0021`), and Variogram by
`0.000099` (`p=0.0004`); the ramp difference is not significant. Its CRPS
differences from the matched Gaussian+ECC and Score-SDE+ECC remain
nonsignificant. Therefore, negative-phase sampling can affect generalization,
but this is not evidence that SA or a future CIM has a universal advantage.

Run the focused regression suite with:

```powershell
D:\anaconda\envs\wsy\python.exe -m pytest tests/test_unified_postprocessing.py tests/test_probabilistic_baselines.py tests/test_qbm_path_integral.py tests/test_wind_reliability_inference.py -q
```
