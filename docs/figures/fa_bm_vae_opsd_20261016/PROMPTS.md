# Built-in imagegen prompts

## Initial generation

Create a precise, elegant, journal-style scientific framework diagram, landscape about 16:10, very high resolution with crisp readable English text. White background, flat vector-like styling, restrained navy/blue/teal/amber palette, thin dark arrows, rounded rectangular blocks, generous whitespace. No decorative photos, no logos, no invented data plots. Title: "FA-BM-VAE for Conditional Wind-Power Scenario Generation". Subtitle: "OPSD fixed-window study | Classical training and optional hardware inference". This is a new corrected architecture figure based on code, not a results claim.

Organize into four clearly separated horizontal bands, with letter labels A through D. Use polished scientific typography and compact, legible text.

A. "OPSD inputs and forecast anchor"
Three connected boxes across the band:
"Provided power forecast (24 h) + calendar features" -> "LS-fitted preprocessing | Context c: 57 dimensions" -> "Conditioning supplied to encoder, BM field and decoder".
Below/inside the first two boxes explain context: "33 standardized features + 24 target-standardized anchor values".
A separate small pale amber anchor block in this band: "Forecast anchor a(c): provided forecast, not a learned MLP". This anchor will be referenced in the reconstruction formula later.
Tiny clear note: "Observed power y is used for training and held-out evaluation, never as an inference input."

B. "Classical training"
Horizontal flow: box "Observed power y + context c" -> box "Bernoulli encoder q_phi(z | y,c) | Straight-through binary sampling" -> box "z in {0,1}^48" -> box "Conditional decoder p_psi(y | z,c)" -> box "Reconstruction loss".
Below this row, a single clean training-objective bar: "BM learning: positive / negative energy + posterior entropy | Negative phase: classical SA".
Small note at bottom: "Checkpoint: minimum VS reconstruction loss. Freeze all learned parameters before hardware sampling."
Do NOT draw an explicit residual-subtraction box before the encoder. Do NOT draw Gaussian latent mu/sigma. Do NOT connect hardware to the training objective.

C. "Frozen-model inference"
This should be the largest band. Left box "Context c" -> next box "Conditional BM / Ising prior" with formula on two short lines:
"h(c) = h0 + Wc + b; J shared across conditions"
"E(s | c) = -h(c)^T s - (1/2) s^T J s"
Below this box small line "48 logical spins; sparse learned couplings".
From this prior branch into two ALTERNATIVE sampling routes, clearly marked OR, and merge only their logical spin output:
Upper route blue box "Classical reference sampling | SA / Gibbs".
Lower route AMBER DASHED outlined group, prominently labeled "Optional SPQC-550 inference — hardware validation pending":
three small sequential blocks INSIDE this group:
"8-bit scaling + auxiliary spin | 49 x 49 matrix"
-> "SPQC-550 sampling | 100 reads / condition"
-> "Auxiliary-spin normalization | recover 48 spins".
Both routes merge into box "s -> z = (s + 1)/2" -> box "Frozen conditional decoder".
Then a final reconstruction box "Anchor + residual correction + observation noise" with formula:
"y_tilde = a(c) + m_psi(z,c) + sigma_psi(z,c) * epsilon"
and small text "epsilon ~ N(0,I); standardized target space".
An arrow from this box to "Inverse scaling + clipping" -> output "Raw ensemble | 100 trajectories x 24 hours".
Arrange C in two subrows if needed for readability; make all arrows unambiguous, avoid crossing, no false hardware-to-training connection. Anchor a(c) in formula comes from provided forecast, m_psi and sigma_psi from decoder. Context c feeds decoder too, indicated by a thin labeled conditioning line that does NOT flow through postprocessing.
Under optional hardware group add only a short control note: "VS-only gain selection and quality checks; TEST locked until acceptance". This note is an experimental safeguard, not a neural module. No performance advantage claimed.

D. "Matched post-processing and evaluation"
Raw ensemble has a direct evaluation route labeled "Raw".
A second optional route: "VS-fitted marginal calibration (Cal)" -> "Temporal ECC | reorder ranks; preserve marginals" -> "Cal + ECC".
The Raw, Cal, and Cal+ECC variants all connect to a final evaluation block labeled "CRPS | Energy | Variogram | Ramp CRPS | 90% coverage".
Small bottom note: "Compare matched variants. ECC is post-processing, not a learned decoder component."
Bottom legend: "Solid: implemented classical pipeline   Dashed amber: optional hardware branch, not yet validated".
Do not write quantum advantage, quantum training, best performance, h and J both condition-dependent, pressure/temperature/wind-speed inputs, or learned MLP anchor. Prior must be a branch used for inference, not an encoder step that samples a prior after the posterior during training. Make scientific correctness, clear direction of flow and readable labels the priority.

## Targeted revision

undefined

