# Revised contributions and original-style framework

## Manuscript-ready contributions (current evidence)

**Our main contributions are as follows:**

- We propose a forecast-anchored BM-VAE for conditional wind-power scenario generation, combining a deterministic forecast anchor with a structured conditional Boltzmann prior and a probabilistic decoder to model forecast-error corrections and conditional uncertainty.
- We combine a temporally informed sparse Ising prior with separate marginal calibration and temporal ensemble copula coupling (ECC). Matched ablations distinguish the effects of forecast anchoring and latent coupling from those of post-processing, while ECC preserves the marginal samples supplied to it.
- We evaluate the framework using OPSD as the primary case study and GEFCom2014 as a cross-dataset benchmark, with comparisons against diffusion, normalizing-flow, and conventional probabilistic baselines. Repeated-seed evaluations and matched ablations characterize its dataset- and metric-dependent strengths and limitations.

## Why the original wording changed

- "learn non-Gaussian forecast residuals" is stronger than the architectural facts alone establish. The decoder uses a conditional diagonal Gaussian given binary z; mixing over z allows flexible non-Gaussian marginal distributions, but this does not by itself demonstrate that the observed residual distribution has been faithfully learned. The revised wording does not assume that result.
- The current temporal-MI mask is built from LS power trajectories. It should not be described as a newly learned residual-specific copula or an end-to-end trainable ECC module.
- ECC preserves its input empirical marginal samples by reordering. This does not guarantee perfect calibration or universal improvement in temporal metrics.
- J=0 ablations assess the contribution of coupling in the studied setting. They do not alone establish that the selected sparse topology is optimal or superior to all alternative graphs.
- Current OPSD three-seed Raw CRPS: Gaussian-versus-FA-BM-VAE Holm p = 0.46065393460653936. Cal+ECC CRPS: Score-SDE-versus-FA-BM-VAE Holm p = 0.22617738226177383. Therefore blanket significant superiority is not supported.
- Statistical evidence must be attached to the exact data split, metric, comparator and processing variant. No claim of significance for every diffusion/flow baseline is made here.
- The hardware branch remains pending physical validation. A completed hardware contribution must be added or revised only after the real sampling experiment is accepted and evaluated.
- These are proposed contribution statements, not a verified first-in-literature claim.

## Figure v3

Delivered image: fa_bm_vae_opsd_original_style_v3.png. Built-in imagegen was used. One targeted correction removed an erroneous observed-target-to-prior connector, corrected auxiliary-spin processing order, and separated forecast/calendar inputs. The revised image was visually inspected. Input labels (y,c), c, and a(c) are repeated connector references rather than a single global wiring graph. Some dense typography should be checked at final journal print size; no scientific performance claim is conveyed by the diagram. No experiment, hardware submission, or GitHub push was performed for this drawing task.

The original supplied image is a STYLE reference only: three dashed pastel panels, serif typography, trapezoid neural-network blocks and a bottom post-processing band. The left/middle/right content is reorganized for clear flow and corrected to match the OPSD implementation. The previous v2 image is retained.

Sources:
- GEFcom2014/models/QBM_VAE/utils_qbm_vae.py
- GEFcom2014/models/QBM_VAE/ising.py: temporal_mi_sparse_graph
- GEFcom2014/models/QBM_VAE/qbm_vae.py: build_graph_mask
- configs/paper/fa_bm_vae_opsd_sa_negative_kaiwu_protocol.json
- export/opsd_sa_negative_matched_v1/main_table_seed012_strict_no_anchor_v2/paired_statistics_holm.csv

Caption: FA-BM-VAE for the OPSD main case, showing forecast anchoring and conditioning, classical training of the binary latent model, alternative frozen-model sampling paths, and probabilistic scenario reconstruction. Marginal calibration and temporal ECC are applied and evaluated separately. The SPQC-550 branch represents a planned hardware inference pathway, not completed hardware evidence.

Figure/table trace:
- artifact_id: fa-bm-vae-opsd-original-style-v3
- source_data: implementation, protocol and frozen model configuration listed in the v2 README; diagram contains no numerical performance claims.
- transformation: manual code-to-block mapping and built-in imagegen rendering using the supplied image as style reference; exact prompt saved in PROMPT_V3.md.
- caption_claim: forecast anchoring, structured binary latent modelling, optional hardware inference and post-processing are separate stages.
- supported_manuscript_claims: [] (no full manuscript has been modified).
- limitations: OPSD-specific anchor/context; hardware validation pending; raster image; not evidence of novelty or predictive superiority.

