# Built-in imagegen prompt: original-style framework v3

Use the attached image ONLY as a STYLE REFERENCE. Redraw the FA-BM-VAE scientific architecture with CORRECTED content specified below. Match its elegant serif typography, bold italic module headings, three tall dashed rounded panels, soft lavender/blue/cream fills, light blue encoder/decoder trapezoids, pastel node boxes, thin black arrow lines, and bottom post-processing band. Do not copy scientifically incorrect nodes from the reference. Make a wide high-resolution landscape image about 16:9, perfectly readable with plenty of space and no clipped text. No results charts or performance claims.

Title: "Forecast-Anchored Boltzmann-Machine VAE for Wind-Power Scenario Generation"
Subtitle in italics: "OPSD main case: classical training, structured latent sampling and separate temporal post-processing"

THREE MAIN COLUMNS, left approximately 25%, center 45%, right 30%:

LEFT LAVENDER: title "Forecast Anchor and Conditioning"
Two input rectangles "Provided power forecast (24 h)" and "Calendar features". Their arrows join a trapezoid labeled "LS-fitted preprocessing", then a box "Context c (57 features)" with small annotation "33 standardized inputs + 24 target-standardized anchor values".
The forecast also branches to a pale amber box "Forecast anchor a(c)" with subtitle "Provided forecast; no learned anchor network".
A separate lower box "Observed power y (24 h)" labeled "Training / evaluation only" connects ONLY to the training encoder, not to inference context.
Keep a small bottom note "All learned parameters are frozen before hardware inference."

CENTER PALE BLUE: title "Structured BM-VAE Latent Module"
Organize two separated subpanels inside this center column:
TOP subpanel labeled "Training path" contains a small input label "(y, c)" -> light-blue trapezoid "Bernoulli encoder q_phi(z | y,c)" -> small box "Binary z in {0,1}^48". Subtitle below encoder "Straight-through binary sampling". This binary output connects RIGHT to the shared decoder in the right main column with an arrow labeled "training". Do NOT pass the posterior sample through a prior sampler.
LOWER subpanel labeled "Inference path" contains a dashed pink rounded box "Conditional BM / Ising prior" containing a small sparse node-and-edge network illustration plus formula
"E(s | c) = -h(c)^T s - (1/2) s^T J s"
and two brief lines "h(c): condition-dependent local field" and "J: shared sparse couplings".
Tiny annotation inside this prior "Temporal-MI graph from LS data".
Context c from left main column has a clearly labeled conditioning arrow to this prior.
Below or to the right of this prior show two ALTERNATIVE sampler branches marked OR:
- solid box "Classical SA / Gibbs"
- pale green DASHED box "Optional SPQC-550 sampling" with three concise lines "8-bit 49 x 49 Ising matrix", "48 logical spins + 1 auxiliary spin", "Hardware validation pending".
For the hardware branch include a small normalization stage label "Auxiliary-spin normalization" before its output merges with the classical branch.
Both branches join a common compact node "48 spins s -> z = (s + 1)/2". This node connects RIGHT to the SAME decoder as the training branch, with arrow labeled "inference". No chain from training encoder into inference sampler.
At bottom of central panel add a compact two-line box "Classical training: reconstruction + BM learning objective" and "SA negative phase | checkpoint: minimum VS reconstruction loss".
Use sparse nodes to illustrate the structured prior but do not claim every latent bit physically equals one time hour. No Gaussian mu/sigma latent block.

RIGHT PALE CREAM: title "Conditional Decoding and Reconstruction"
A large light-blue decoder trapezoid, near the middle, labeled "Conditional decoder p_psi(y | z,c)" receives the two alternative z arrows from center. A dashed conditioning arrow labeled c feeds the decoder; use a short context input port rather than long crossing lines if needed.
Decoder output box "Residual mean m_psi(z,c)" and "Scale sigma_psi(z,c)".
These outputs go to a reconstruction box with formula
"y_tilde = a(c) + m_psi(z,c) + sigma_psi(z,c) * epsilon"
under it "epsilon ~ N(0,I)" and "Standardized target space".
Provide a short separate anchor input port "a(c)" entering reconstruction, referring to the left anchor box, without drawing a confusing long crossing line.
Below reconstruction -> box "Inverse scaling and clipping" -> pale-blue output box "Raw 24-hour scenario ensemble" subtitle "100 trajectories x 24 hours".
No actual target y enters inference or reconstruction.

BOTTOM BAND, separate from all three main columns:
Title "Matched Post-processing and Evaluation".
Use explicit left-to-right connected boxes:
"Raw ensemble (from decoder)" -> "VS-fitted marginal calibration (Cal)" -> "Temporal ECC" -> final evaluation box.
Under Temporal ECC: "Reorder temporal ranks; preserve input marginals".
The ECC-to-evaluation arrow label is "Cal + ECC".
In addition two clean bypass arrows connect Raw DIRECTLY to evaluation (labeled "Raw"), and Cal DIRECTLY to evaluation (labeled "Cal"). Route above and below boxes with whitespace; make arrows originate correctly.
Evaluation box: "CRPS | Energy | Variogram" and "Ramp CRPS | 90% coverage".
Small bottom note: "Hardware branch: inference only, pending validation. ECC is separate post-processing."

Important invariants:
- Keep the original three-panel dashed pastel style and trapezoid neural blocks, NOT a four-horizontal-row layout.
- This is the OPSD configuration: no wind-speed / wind-direction / temperature / pressure inputs, no embedding network, no learned MLP anchor.
- Encoder input is y,c, not explicit residual r. Binary latent, not Gaussian latent.
- Only h depends on c, not J.
- Optional physical sampler is NOT part of training, and NOT proven quantum advantage.
- Reconstruction includes observation noise, not just a mean residual.
- Cal and ECC are explicitly separate from BM and separately evaluated.
- No text claiming statistical significance, universal improvement, or a novel copula algorithm.
Scientific correctness and clean routing are more important than adding decorative detail.

