# FA-BM-VAE OPSD framework — 2026-10-16

## Scope
Architecture illustration for the historical fixed-window OPSD SA-negative-phase main model. Not a performance figure or proof of hardware execution. Based on current repository source and the frozen seed-0 configuration.

## Caption
FA-BM-VAE framework for conditional 24-hour wind-power scenario generation in the OPSD fixed-window study. A provided power forecast supplies the forecast anchor. A Bernoulli encoder and a conditional BM prior are trained classically, with simulated annealing used for the main model's negative phase. During inference, a classical sampler or the optional SPQC-550 branch supplies binary latent states to the frozen conditional decoder. The decoder combines the anchor, a learned residual mean correction, and observation noise. Raw scenarios and matched calibration/ECC variants are evaluated separately. The hardware branch remains pending physical validation.

## Verified implementation mappings
- Provided forecast anchor, not a learned MLP, in this OPSD configuration.
- Context: 33 standardized original features plus 24 forecast values transformed with the target scaler = 57 inputs.
- Encoder receives standardized target y and context c. No explicit y-minus-anchor preprocessing before the encoder.
- Binary latent dimension: 48. Straight-through logistic relaxation in training; not Gaussian latent variables.
- Conditional field: h(c) = h0 + Wc + b. Symmetric masked coupling matrix J is shared across conditions.
- Decoder: diagonal conditional Gaussian; mean = anchor + learned correction; observation noise is enabled.
- Model parameters are frozen before hardware inference. Hardware does not participate in training/backpropagation.
- Hardware representation: 49 x 49 int8 matrix with one auxiliary spin; normalize and remove the auxiliary spin before converting 48 logical spins to binary z.
- Marginal calibration is VS-fitted; ECC is separate temporal post-processing.
- Planned hardware 100 reads per condition does not imply 100 statistically independent or distinct latent states.

## Contribution boundaries
1. Forecast-anchored uncertainty modelling: a candidate methodological contribution supported by matched ablations; does not claim invention of residual learning.
2. Conditional structured binary latent prior: candidate contribution of the particular model combination and its empirical evaluation, not invention of BM or VAE.
3. Hardware-compatible inference: a planned validation contribution. Must not be reported as achieved until real responses have been validated and evaluated.
4. Quantization checks, provenance, quality gates, and split discipline strengthen reliability; they are not automatically new model innovations.
5. ECC and marginal calibration should not be presented as newly invented methods.
No priority/first-in-literature claim has been verified in this figure task.

## Sources
- GEFcom2014/models/QBM_VAE/utils_qbm_vae.py: ConditionalQBMVAE, qbm_params, encode_logits, sample_binary_latent, decode_temporal_parameters, loss.
- export/qbm_vae_opsd-wind/opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0.json: frozen main-model configuration.
- GEFcom2014/models/QBM_VAE/reconstruct_hardware_scenarios.py: spin-to-binary mapping, frozen decoding, observation noise, inverse scaling, clipping.
- configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json: hardware preparation, sampling budget, and quality-policy status.
Source commit at drafting: f87aa263b92ce12cfde9a5dc810ba486abe4fb00.

## Figure/table trace
- artifact_id: fa-bm-vae-opsd-framework-20261016
- source_data: the implementation/configuration files listed above; architecture diagram, no numerical performance data.
- transformation: manual code-to-block mapping described in Verified implementation mappings, rendered using built-in imagegen; exact generation/edit prompts in PROMPTS.md.
- caption_claim: classical training is separate from optional frozen-model hardware inference; anchor, binary latent prior, decoder and matched post-processing are distinct.
- supported_manuscript_claims: [] (no manuscript has been updated in this task).
- limitations: OPSD-specific anchor/context; raster figure needs typesetting review before submission; hardware branch is pending; no superiority or quantum-advantage evidence is implied.

## Visual check
Built-in imagegen was used (not CLI). The first render had an ambiguous Raw bypass in panel D; a targeted revision replaced it with three explicit evaluation routes. The final render was visually checked against the implementation mappings above. This is a raster architecture illustration, not numerical evidence or independent scientific peer review. No training, hardware call, or GitHub push was performed during figure creation.

## LaTeX inclusion
Place the delivered PNG in the manuscript's figures folder, then use:
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fa_bm_vae_opsd_framework_v2.png}
  \caption{FA-BM-VAE architecture for the OPSD fixed-window study. Classical training is separated from frozen-model inference. The optional SPQC-550 sampling branch awaits physical validation; Raw, Cal, and Cal+ECC are evaluated as matched variants.}
  \label{fig:fa-bm-vae-opsd}
\end{figure*}
```
Requires graphicx. Use figure rather than figure* for a single-column manuscript.

