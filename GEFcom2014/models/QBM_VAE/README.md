# QBM-VAE and transverse-field extension

This directory contains two distinct prior models. They must not be described
as the same method in experiments or papers.

## Model definitions

`QBMVAE_2` uses the classical Ising energy

```text
E_x(s) = -h(x)^T s - 0.5 s^T J s.
```

`PIQBMVAE_1` adds a non-commuting transverse field:

```text
H_x = -sum_i h_i(x) sigma_i^z
      -sum_{i<j} J_ij sigma_i^z sigma_j^z
      -sum_i Gamma_i sigma_i^x.
```

The quantum Gibbs state is approximated with a finite-replica Suzuki-Trotter
path integral. Training uses a contrastive Trotter-bound surrogate with a
persistent negative phase. This is a controlled approximation, not an exact
quantum ELBO and not native quantum-hardware Gibbs sampling.

## Local validation

Run the small-system exact diagonalization check:

```powershell
python -m GEFcom2014.models.QBM_VAE.validate_trotter_mapping --replicas 2 3 4 6
python -m unittest tests.test_qbm_path_integral -v
```

The validation compares the diagonal of the exact transverse-field quantum
Gibbs state with the exactly enumerated finite-replica marginal. Report this
error separately from forecasting error.

## Training

Start with a fixed transverse field. It gives a clean ablation against the
classical prior and avoids identifiability between effective temperature and
the learned field.

```powershell
python -m GEFcom2014.models.QBM_VAE.qbm_vae `
  --tag load `
  --sampler path-integral `
  --transverse-field 0.5 `
  --trotter-replicas 4 `
  --latent-s 48 `
  --seed 0 `
  --skip-plots
```

After fixed-field ablations are stable, add
`--learnable-transverse-field`. The learned value is constrained to a positive
bounded range.

Use at least three seeds and test the following independent factors:

```text
Gamma:     0.10, 0.25, 0.50, 1.00
Replicas:  2, 4, 8
Graph:     temporal-mi, random, full, none
Prior:     classical Ising, finite-replica TFIM
```

The latent size must be divisible by the number of modeled periods. This makes
period-level graph and field interpretations well defined.

## Explainability

```powershell
python -m GEFcom2014.forecast_quality.plot_qbm_explainability `
  --tag load `
  --model-name load_PIQBMVAE_1_path-integral_0 `
  --beta 1.0
```

The report includes conditional fields, coupling edges, weather-to-field
sensitivity, transverse-field strength, imaginary-time disagreement,
transverse magnetization, action decomposition, period aggregation, and
coupling variability across seed models. The baseline-shifted action should be
used when comparing longitudinal and imaginary-time contributions because the
raw Trotter action contains a large aligned-replica constant.

## Bosonic platform gate

Do not connect the platform until all of these checks pass:

1. Forecast rankings are stable across at least three seeds.
2. Results are reported with CRPS, quantile score, reliability, multivariate
   energy/variogram scores, and runtime.
3. Increasing the replica count does not materially change validation metrics.
4. Fixed-Gamma and learned-Gamma ablations beat or explain the classical prior.
5. The expanded problem size `latent_s * replicas` fits the platform limit.
6. Effective temperature and coefficient scaling are calibrated on VS only.

Then export the executable classical path-integral graph:

```powershell
python -m GEFcom2014.models.QBM_VAE.export_ising_instances `
  --tag load `
  --model-name load_PIQBMVAE_1_path-integral_0 `
  --split VS `
  --beta 1.0 `
  --trotter-replicas 4
```

The exported payload stores both the physical `h, J, Gamma` model and the
expanded classical Ising graph. A bosonic/CIM backend sampling this expanded
graph is a quantum-inspired implementation of the finite-Trotter prior. It
must not be claimed as direct preparation of a non-commuting quantum Gibbs
state unless the hardware and measurement protocol actually implement that
state.

## Frozen SA/CIM comparison and temperature calibration

For the classical FA-BM-VAE mainline, export disjoint VS and TEST instances
with exactly the same learned-anchor transformation used in training:

```powershell
python -m GEFcom2014.models.QBM_VAE.export_ising_instances `
  --tag wind --model-name wind_QBMVAE_2_lanchor_sa_0 `
  --split VS --num-instances 50 --selection stratified `
  --selection-seed 2026 `
  --output-dir export\bosonic_instances\wind_lanchor_seed0_hardware

python -m GEFcom2014.models.QBM_VAE.export_ising_instances `
  --tag wind --model-name wind_QBMVAE_2_lanchor_sa_0 `
  --split TEST --selection all `
  --output-dir export\bosonic_instances\wind_lanchor_seed0_hardware
```

The VS panel contains five conditions from each of the ten Wind zones. The
TEST pool contains all 500 zone-date conditions. Build and hash-lock the VS
package before submission:

```powershell
python -m GEFcom2014.models.QBM_VAE.prepare_bosonic_submission `
  --manifest export\bosonic_instances\wind_lanchor_seed0_hardware\wind_wind_QBMVAE_2_lanchor_sa_0_vs_manifest.json `
  --stage vs-calibration --requested-reads 1000 `
  --output-dir export\bosonic_submissions\wind_lanchor_seed0_real_vs
```

Set `--max-bits`, `--max-edges`, and `--coefficient-limit` from the actual
platform contract. The classical logical VS model remains 48 spins with 180
logical coupling edges, maximum degree 10, and maximum absolute coefficient
`1.041965`. A Kaiwu hardware package is different: it contains 49 spins
(including the auxiliary spin) and 228 source edges (180 `J` edges plus 48
field edges). The actual nonzero edge count after global int8 quantization is
recorded in the audit and must not be silently changed by clipping. If a
platform limit is exceeded, rebuild from the original `h,J` with one
documented global scale and repeat VS calibration.

Platform responses may be per-instance JSON, CSV, or NPZ. JSON may contain
`samples`, or `bitstrings` with optional `counts`. Importing verifies all
payload hashes, spin order, dimensions, values, response counts, and raw-file
hashes:

```powershell
python -m GEFcom2014.models.QBM_VAE.import_hardware_responses `
  --submission-manifest export\bosonic_submissions\wind_lanchor_seed0_real_vs\package\submission_manifest.json `
  --input-dir <raw-platform-vs-response-directory> `
  --output-dir export\bosonic_responses\wind_lanchor_seed0_real_vs
```

Use these imported VS responses only to estimate one shared effective inverse
temperature and freeze the TEST coefficient scale:

```powershell
python -m GEFcom2014.models.QBM_VAE.calibrate_hardware_temperature `
  --vs-manifest export\bosonic_instances\wind_lanchor_seed0_hardware\wind_wind_QBMVAE_2_lanchor_sa_0_vs_manifest.json `
  --vs-responses-dir export\bosonic_responses\wind_lanchor_seed0_real_vs `
  --test-manifest export\bosonic_instances\wind_lanchor_seed0_hardware\wind_wind_QBMVAE_2_lanchor_sa_0_test_manifest.json `
  --target-beta 1.0 --bootstrap-repetitions 2000 `
  --output-dir export\bosonic_calibration\wind_lanchor_seed0
```

The TEST package command rejects payloads without a VS calibration record:

```powershell
python -m GEFcom2014.models.QBM_VAE.prepare_bosonic_submission `
  --manifest export\bosonic_calibration\wind_lanchor_seed0\test_calibrated_manifest.json `
  --stage test-evaluation --requested-reads 100 `
  --output-dir export\bosonic_submissions\wind_lanchor_seed0_real_test
```

Submit this frozen archive without further fitting, import the TEST responses
with the same importer, and validate them independently:

```powershell
python -m GEFcom2014.models.QBM_VAE.benchmark_hardware_samples `
  --manifest export\bosonic_instances\wind_lanchor_seed0_hardware\wind_wind_QBMVAE_2_lanchor_sa_0_test_manifest.json `
  --responses-dir <platform-test-response-directory> `
  --output-dir export\bosonic_benchmark\wind_lanchor_seed0

python -m GEFcom2014.models.QBM_VAE.compare_hardware_responses `
  --manifest export\bosonic_instances\wind_lanchor_seed0_hardware\wind_wind_QBMVAE_2_lanchor_sa_0_test_manifest.json `
  --reference-dir <sa-reference-response-directory> `
  --candidate-dir <platform-test-response-directory> `
  --reference-label SA --candidate-label CIM `
  --output-dir export\bosonic_comparison\wind_lanchor_seed0
```

The calibration estimator was first checked with a 20-instance pilot recovery
test. This old pilot used the historical head selection and therefore covered
only one Wind zone; it validates the numerical recovery mechanism, not spatial
representativeness.
SA responses were generated at a hidden `beta=0.65`; VS-only estimation
recovered `beta_eff=0.6500` with bootstrap 95% CI `[0.6450, 0.6550]` and froze
a coefficient scale of `1.5385`. On 20 held-out TEST instances, mean
`beta_eff` moved from `0.6479` to `1.0002` (target reference `0.9968`).
Relative to the target sampler, energy Wasserstein distance fell from `2.9385`
to `0.1220`, magnetization MAE from `0.1110` to `0.0224`, and edge-moment MAE
from `0.0916` to `0.0245`. All paired improvements had bootstrap intervals
above zero and sign-flip `p <= 1e-4`.

The corrected ten-zone VS package was then replayed end to end through the
hash-checked response importer. With 50 instances and 1000 SA reads per
instance at target `beta=1`, the shared estimator recovered `beta_eff=1.0000`
with 95% CI `[0.9950, 1.0000]`. The real VS archive is
`export/bosonic_submissions/wind_lanchor_seed0_real_vs/vs-calibration.zip`,
with SHA-256
`99cc63eb66a31d223e0d76c1d2d090e6db52cfe528e7d932bee8ada67d12aa85`.

This recovery experiment validates the leakage-free calibration and comparison
protocol only. It is not a real CIM result and does not establish quantum
advantage. A hardware claim requires responses returned by the physical
platform for the same frozen payloads, alongside SA controls, read count,
latency, failures, coefficient clipping, embedding, and platform metadata.

### FA-BM-VAE platform-to-scenario reconstruction

The paper method is **FA-BM-VAE**: Forecast Anchor supplies the main trend;
the conditional classical Ising BM/VAE supplies residual and scenario
variation. Training is classical. A bosonic SPQC/CIM platform substitutes only
the latent conditional-Ising sampling step after training has been frozen.
This is a sampler-integration protocol, not complete QBM training and not a
quantum-advantage claim. The directory name `QBM_VAE`, Python class names, and
`QBMVAE_2` artifacts are retained only for backward compatibility.

After importing platform responses, reconstruct scenarios with the frozen
decoder:

```powershell
python -m GEFcom2014.models.QBM_VAE.reconstruct_hardware_scenarios `
  --tag wind `
  --model-name wind_QBMVAE_2_lanchor_sa_0 `
  --manifest export\bosonic_calibration\wind_lanchor_seed0\test_calibrated_manifest.json `
  --responses-dir export\bosonic_responses\wind_lanchor_seed0_real_test `
  --output-dir export\fa_bm_vae_hardware\wind_test `
  --output-label BosonicSPQC
```

For OPSD Wind, add `--dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz`
and use `opsd-wind_QBMVAE_2_anchor_sa_0`. The output is
`scenarios_FA-BM-VAE_BosonicSPQC_<reads>_<split>.pickle`, accompanied by JSON
metadata and the response-import audit path. The generated scenarios can then
be passed to the existing unified postprocessing and forecast-quality scripts.

### Kaiwu 8-bit matrix contract and staged experiment

When `--hardware-gain G` is supplied to `export_ising_instances`, the original
48-dimensional logical model is converted to

```text
M[:48,:48] = -J/2
M[i,48] = M[48,i] = -h[i]/2
Q = round-half-to-even(G M)
```

with `u=[s,+1]` satisfying `u.T @ M @ u = E(s)`. `Q` is checked as a
symmetric zero-diagonal int8-range matrix. The package records the source
228-edge count, the actual post-quantization nonzero edge count, maximum
integer, zeroed-coefficient fraction, relative error, gain, and matrix
SHA-256. It never clips, uses `PrecisionReducer`, splits variables, or applies
per-instance scaling.

The real-platform response contract is 49 raw columns in index-ascending order.
The importer validates the task ID and matrix hash, then computes
`logical_spins = raw_spins[:, :48] * raw_spins[:, 48:49]`. Scenario decoding,
benchmarking, and CRPS evaluation continue to consume the resulting 48-column
logical samples.

The opt-in single-instance submission command is:

```powershell
python -m GEFcom2014.models.QBM_VAE.submit_kaiwu_sampling `
  --matrix-file <package>\hardware_matrices\<instance_id>.npz `
  --instance-id <instance_id> --num-reads 100 `
  --output-dir export\kaiwu_raw_responses `
  --checkpoint-dir export\kaiwu_checkpoints
```

`--checkpoint-dir` is required. A task-specific subdirectory is created below
it and assigned to Kaiwu's `CheckpointManager.save_dir`, so the SDK default
cache path is not used. The real integration tests are opt-in only:
`FA_BM_VAE_RUN_REAL_KAIWU=1` plus the platform variables documented in
`tests/test_kaiwu_real_integration.py`.

It is the only repository entry point that can contact Kaiwu. Tests never call
it, and the repository contains no license, account, key, or project number.

Gain selection is VS-only. Prepare one candidate manifest/response directory
per global gain and evaluate them with:

```powershell
python -m GEFcom2014.models.QBM_VAE.calibrate_hardware_temperature `
  --gain-candidates <gain_candidates.json> `
  --test-manifest <selected_gain_test_manifest.json> `
  --output-dir export\kaiwu_calibration\fa_bm_vae_wind
```

The candidate file is a JSON list of `{gain, vs_manifest, responses_dir}`
objects; an optional `reference_dir` points to floating-point SA responses for
the same instances. With that reference, selection uses energy-distribution
Wasserstein distance, effective-beta error, edge-moment MAE, magnetization MAE,
and state diversity. Without it, selection minimizes the absolute effective-
beta error on the original floating-point `h,J`, then prefers state diversity
and lower absolute magnetization. A selected gain is frozen; any new gain
requires rebuilding and re-quantizing from the original `h,J`.

Run the physical experiment in three stages: (1) 5--10 stratified VS
instances with 100 reads to validate the interface, (2) 50 stratified VS
instances with 1000 reads for gain selection and temperature diagnostics, and
(3) only after freezing the gain, 500 TEST instances with 100 reads. Every
stage compares floating-point SA, quantized-matrix SA, and Kaiwu/SPQC samples.

## Temporal marginal-dependence decomposition

The classical QBM-VAE mainline supports an optional variance-preserving
Temporal-AR(1) Gaussian decoder:

```powershell
python -m GEFcom2014.models.QBM_VAE.qbm_vae --tag wind --sampler sa --decoder-covariance ar1 --run-label tar1m --seed 0 --skip-plots
```

For period `t`, the standardized decoder residual follows
`u_t = rho_t u_(t-1) + sqrt(1-rho_t^2) epsilon_t`. This preserves the learned
hourly marginal standard deviation while making temporal persistence explicit.
Directly replacing the diagonal decoder improves ramp dependence but can
degrade CRPS. The promoted variant therefore uses conditional ensemble copula
coupling: calibrated scenarios from the stronger diagonal QBM-VAE provide the
hourly marginals, while matched Temporal-AR scenarios provide only trajectory
ranks. The coupling preserves every hourly sample multiset exactly.

```powershell
python -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag wind --seeds 0 1 2
python -m GEFcom2014.forecast_quality.qbm_temporal_diagnostics --tag wind --seeds 0 1 2
```

This is a marginal/dependence decomposition, not evidence of quantum
advantage. The Ising `h,J` prior remains the hardware-facing component; ECC is
a deterministic rank coupling applied after decoding.

For an external daily dataset bundle, pass the same bundle to training, ECC,
and evaluation so target dimensions and chronological splits are audited from
one source:

```powershell
python -m GEFcom2014.models.QBM_VAE.qbm_vae --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --sampler sa --seed 0 --skip-plots
python -m GEFcom2014.models.QBM_VAE.qbm_vae --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --sampler sa --decoder-covariance ar1 --run-label tar1m --seed 0 --skip-plots
python -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --seeds 0 1 2
```

## Forecast-anchored residual decoder

When the conditioning data begins with a target-sized day-ahead point forecast,
`--forecast-anchor` turns the decoder into an explicit residual model:

```text
y = forecast_anchor + stochastic_QBM_decoder_residual.
```

The anchor is transformed with the LS target scaler, appended to the model
context, and added to the decoder mean. The residual mean head is initialized
at zero. No VS or TEST target is used to construct the anchor or fit scaling.
Do not enable this flag when the first context block is not a target-scale
point forecast.

```powershell
python -m GEFcom2014.models.QBM_VAE.qbm_vae --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --sampler sa --forecast-anchor --run-label anchor --seed 0 --skip-plots
python -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --source-label anchor --template-label tar1m --output-label QBMAnchorECC --seeds 0 1 2
```

On the three-seed OPSD Wind experiment, the forecast-anchored QBM has CRPS
`0.027529`, compared with `0.028670` for Conditional DDPM and `0.028546` for
an LS daily-residual bootstrap. ECC keeps that CRPS unchanged while reducing
ramp quantile MAE from `0.032296` to `0.006371`. The CRPS improvements over
DDPM and residual bootstrap are significant under paired date-block inference.
This supports the anchored residual and dependence decomposition; it is not a
quantum-advantage result because this experiment uses classical Ising SA.

### Learned anchor for weather-only contexts

GEFCom Wind provides 250 weather/zone features but no target-scale power point
forecast. `--learned-forecast-anchor` therefore fits a deterministic MLP on LS,
selects its checkpoint by VS MSE, freezes its 24-dimensional prediction, and
uses that prediction as the residual decoder anchor. The anchor checkpoint and
its LS/VS history are saved beside the QBM model.

```powershell
python -m GEFcom2014.models.QBM_VAE.qbm_vae --tag wind --sampler sa --learned-forecast-anchor --run-label lanchor --anchor-epochs 100 --anchor-patience 15 --epochs 80 --seed 0 --skip-plots
python -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag wind --source-label lanchor --template-label tar1m --output-label QBMLearnedAnchorECC --seeds 0 1 2
```

The current three-seed learned-anchor QBM+ECC result on GEFCom Wind is CRPS
`0.083382 +/- 0.000493`, Energy `0.516077`, Variogram `0.028325`, and ramp
quantile MAE `0.016726`. It significantly improves all four scores over the
existing Spline Conditional NF and Conditional DDPM baselines. The frozen
anchor alone has point-forecast CRPS around `0.118`.

The matched forecast-anchored conditional Gaussian ablation is available as:

```powershell
python -m GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline --tag wind --anchor-source learned --epochs 80 --seed 0
python -m GEFcom2014.models.QBM_VAE.anchor_gaussian_baseline --tag wind --anchor-source learned --components 4 --epochs 80 --seed 0
python -m GEFcom2014.models.QBM_VAE.anchor_spline_flow --tag wind --anchor-source learned --epochs 200 --seed 0
python -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag wind --source-dir export\anchor_gaussian_wind --source-version AnchorGaussian --template-dir export\qbm_vae_wind --output-dir export\anchor_gaussian_ecc_wind --output-label AnchorGaussianECC --seeds 0 1 2
python -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag wind --source-dir export\anchor_gmm4_wind --source-version AnchorGMM4 --template-dir export\qbm_vae_wind --output-dir export\anchor_gmm4_ecc_wind --output-label AnchorGMM4ECC --seeds 0 1 2
python -m GEFcom2014.forecast_quality.qbm_temporal_ecc --tag wind --source-dir export\anchor_spline_flow_wind --source-version AnchorSplineFlow --template-dir export\qbm_vae_wind --output-dir export\anchor_spline_flow_ecc_wind --output-label AnchorSplineFlowECC --seeds 0 1 2
```

Across three seeds, Gaussian+ECC obtains CRPS `0.083229 +/- 0.000373`, Energy
`0.516269`, and Variogram `0.028421`. None differs significantly from the
learned-anchor QBM+ECC result. QBM+ECC significantly improves daily ramp CRPS
(`p<0.001`), but the independent OPSD comparison shows only a nonsignificant
ramp trend (`p=0.0776`). A stronger four-component Gaussian mixture obtains
GEFCom CRPS `0.082830`, Energy `0.515363`, and Variogram `0.028377`; none of its
four paired score differences from QBM+ECC is significant, and it removes the
single-Gaussian ramp advantage. On OPSD, the same fixed `K=4` mixture is
significantly worse than QBM+ECC on all four scores. The defensible conclusion
is that anchoring and ECC explain most aggregate gains, while QBM is more
robust than this complex mixture on OPSD but has no isolated, general
Boltzmann-prior advantage yet.

The forecast-anchored Spline Flow uses five conditional rational-quadratic
autoregressive transforms and models the same scaled residual target. QBM+ECC
significantly outperforms Flow+ECC in CRPS, Energy, Variogram, and ramp CRPS on
both GEFCom Wind and OPSD Wind. This is the strongest current evidence for the
robustness of the structured BM latent model, although the statistically tied
single-Gaussian ablation means it is still not evidence of universal
Boltzmann-prior superiority or quantum advantage.

### Forecast-anchored conditional Score-SDE baseline

The strengthened diffusion baseline models the same anchored residual target
with continuous-noise EDM denoising, an EMA checkpoint selected on VS, and a
32-step deterministic Heun sampler:

```powershell
python -m GEFcom2014.models.QBM_VAE.anchor_score_sde --tag wind --epochs 200 --sampling-steps 32 --n-scenarios 100 --seed 0
python -m GEFcom2014.models.QBM_VAE.anchor_score_sde --tag opsd-wind --dataset-bundle GEFcom2014\data\external\opsd_wind_daily.npz --anchor-source context --epochs 200 --sampling-steps 32 --n-scenarios 100 --seed 0
```

Across three seeds, Score-SDE+ECC obtains CRPS `0.082841 +/- 0.000472` on
GEFCom Wind and `0.027655 +/- 0.000143` on OPSD Wind. It significantly
improves CRPS over the original Conditional DDPM on both datasets. Its CRPS,
Energy, and Variogram differences from FA-BM-VAE+ECC are not significant on
either dataset, while its ramp CRPS is significantly lower. The paper should
therefore describe Score-SDE and FA-BM-VAE as statistically competitive strong
models, not claim that the BM prior universally outperforms diffusion.

### Forecast-anchor x Ising-coupling ablation

The frozen Wind ablation crosses a learned forecast anchor with either an
independent prior (`J=0`) or the learned Temporal-MI Ising graph. Run the
matched three-seed experiment with:

```powershell
python -m GEFcom2014.models.QBM_VAE.run_anchor_coupling_ablation --seeds 0 1 2 --evaluate --repetitions 20000
```

Mean TEST CRPS for `(no anchor, J=0)`, `(no anchor, learned J)`,
`(learned anchor, J=0)`, and `(learned anchor, learned J)` is respectively
`0.088013`, `0.088030`, `0.083408`, and `0.083382`. The anchor lowers CRPS by
about `0.0046` and Energy by about `0.024`. Learned `J` has no significant
CRPS or Energy effect, but significantly improves Variogram and ramp CRPS in
both anchor conditions after a single Holm correction over 20 tests on the
VS-selected `Cal` scenarios. The
significant positive interaction on both dependence scores indicates
diminishing returns: anchoring and coupling are complementary in role but
partly overlapping in temporal benefit. This calibrated Wind-only comparison
attributes marginal accuracy mainly to the anchor, but does not isolate the
Temporal-MI topology from a sparsity-matched random graph.
The detailed tables and audit are under
`export/anchor_coupling_ablation/wind/`.

### Cross-dataset graph-structure ablation

The stricter test uses Raw scenarios as the primary endpoint and compares the
Temporal-MI graph against both `J=0` and random graphs with exactly the same
edge count and nonzero weight multiset. It covers three seeds on GEFCom Wind
and OPSD Wind, excludes TRC/ECC, and applies grouped Holm correction across the
eight primary dependence tests:

```powershell
python -m GEFcom2014.models.QBM_VAE.run_graph_structure_ablation --seeds 0 1 2 --evaluate --repetitions 20000
```

Temporal-MI improves Raw ramp CRPS over `J=0` on OPSD (`-0.000137`, adjusted
`p=0.0004`, 3/3 seeds), but is worse than the matched random graph on GEFCom
ramp CRPS (`+0.000177`, adjusted `p=0.0014`, only 1/3 seeds favorable). It has
no significant Raw Variogram advantage over random on either dataset. Thus,
the current results do not establish a stable cross-dataset benefit from the
Temporal-MI topology. Detailed aggregate, date-block, graph-mask, and seed
audits are under `export/graph_structure_ablation/`.

### VS-only candidate development and independent confirmation

The stable residual Temporal-MI and posterior partial-correlation graph
candidates did not pass the two-dataset VS gate. A differentiable trajectory
objective combining Energy, Variogram, and ramp scores was then evaluated with
development seeds 0--2. Because its OPSD mean improved but only one seed had a
favorable dependence direction, it was not accepted from the development run.

An independently predeclared confirmation with seeds 3--5 compared the
trajectory objective against a paired, otherwise identical control. GEFCom
Wind passed (`2/3` favorable seeds; dependence composite `-0.106%`), but OPSD
Wind failed (`1/3`; `+0.712%`). OPSD ramp CRPS increased by `0.000133` (95% CI
`[0.000060, 0.000205]`, raw paired permutation `p=0.00065`, Holm-adjusted
`p=0.00520`). The trajectory-loss
candidate is therefore retained as an experimental option, not as the reported
main model. No candidate TEST scenarios were generated. Reproduce the final
confirmation with:

```powershell
python -m GEFcom2014.models.QBM_VAE.run_trajectory_score_confirmation --seeds 3 4 5 --repetitions 20000
```

The gate, seed directions, confidence intervals, and p-values are in
`export/trajectory_score_confirmation/`.

### Additional strong baselines

The comparison pipeline discovers `D3U (NWP-adapted)` and official
`Treeffuser 0.2.0` scenario pairs automatically. D3U retains the published
deterministic/uncertain decomposition, frozen conditioner, residual diffusion,
and PatchDN structure while adapting its conditioner to the available future
NWP inputs. Treeffuser fits the joint 24-dimensional conditional distribution
with its official LightGBM score estimator. The repository configuration uses
`n_repeats=5`, `n_estimators=500`, and `n_jobs=4` as an explicit CPU budget.
Run all five seeds with:

```powershell
python -m GEFcom2014.models.run_strong_baselines --seeds 0 1 2 3 4 --evaluate --repetitions 2000
```

These are classical strong baselines and do not change the BM/QBM naming or
hardware claims. The completed five-seed outputs are in
`export/unified_postprocessing/wind/` and
`export/unified_postprocessing/opsd-wind/`; aggregate CRPS is `0.083755` and
`0.089749` for D3U and Treeffuser on GEFCom Wind, and `0.028797` and `0.033830`
on OPSD Wind, respectively. These baselines do not establish a quantum
advantage.

### Frozen paper configurations

The classical FA-BM-VAE settings used for the main comparison are versioned
under `configs/paper/`:

```powershell
python -m GEFcom2014.models.QBM_VAE.qbm_vae --config configs\paper\fa_bm_vae_wind.json --seed 0
python -m GEFcom2014.models.QBM_VAE.qbm_vae --config configs\paper\fa_bm_vae_opsd_wind.json --seed 0
```

Command-line values override JSON defaults. The historical class and artifact
names retain `QBMVAE_2` for compatibility, but these frozen experiments use a
classical Ising Hamiltonian, internal Gibbs negative-phase training, and SA
scenario generation. The precise method name for the paper is therefore
**FA-BM-VAE** until a real bosonic/CIM sampler is used.

### External negative-phase samplers

Generation and training samplers are configured independently:

- `--sampler` controls latent sampling during scenario generation.
- `--negative-phase-backend` controls model samples used by the training
  positive-minus-negative phase.
- `internal` preserves the historical persistent-Gibbs training path.
- `sa`, `gibbs`, and `exact` activate explicit external classical samplers.
- `cim` remains an SA placeholder and must not be reported as hardware.
- `bosonic` is reserved for a programmatic platform client; the training CLI
  rejects it because platform sampling is post-training only. Use the export,
  submission, import, calibration, and reconstruction chain above.

Run the small exact-law diagnostic first:

```powershell
python -m GEFcom2014.models.QBM_VAE.validate_ising_samplers --n-bits 12 --num-reads 5000 --seeds 0 1 2
```

Then run the three-seed SA negative-phase control:

```powershell
python -m GEFcom2014.models.QBM_VAE.qbm_vae --config configs\paper\fa_bm_vae_wind.json --negative-phase-backend sa --negative-sampler-sweeps 20 --run-label lanchor_sa_neg --seed 0
python -m GEFcom2014.models.QBM_VAE.qbm_vae --config configs\paper\fa_bm_vae_opsd_wind.json --negative-phase-backend sa --negative-sampler-sweeps 20 --run-label anchor_sa_neg --seed 0
```

On GEFCom Wind, SA-negative+ECC and internal-Gibbs+ECC are statistically tied:
CRPS `0.083418` versus `0.083382` (`p=0.847`). On OPSD, SA-negative+ECC
obtains CRPS `0.027284`, compared with `0.027529` for internal Gibbs
(`p=0.0434`), with significant Energy and Variogram improvements. These
results validate the external negative-phase path and justify using SA as the
paired classical control for CIM. They do not establish quantum advantage.
