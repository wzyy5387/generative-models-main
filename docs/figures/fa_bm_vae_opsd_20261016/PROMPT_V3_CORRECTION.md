# Targeted imagegen correction

Make only three scientific routing corrections to this diagram. Preserve its three-column pastel/dashed style, all titles, dimensions, equations, decoder flow, and bottom Raw/Cal/Cal+ECC routes. Keep high resolution and crisp English text.

1. CRITICAL: Remove the entire solid black connector that starts at the "Observed power y (24 h)" box, goes horizontally right across the gap then up toward the conditional BM prior. Remove the "(y,c)" label on that erroneous connector too. Observed y must NEVER be connected to the inference prior. Keep the observed-power box, but change its bottom annotation to "Training encoder / evaluation only". Do not draw any new arrow from this box. The existing "(y,c)" input port immediately before the training encoder already specifies that encoder input correctly and must stay. Keep the dashed context c arrow to the BM prior; it must originate only from the Context c box. There must be no intersection, node, or connector between observed y and this context arrow.

2. CRITICAL: Replace the green optional hardware box and the separate blue Auxiliary-spin normalization box BELOW it with ONE slightly taller pale green dashed rounded box in the SAME area. Text inside:
"Optional SPQC-550 sampling"
"8-bit 49 x 49 Ising matrix"
"48 logical spins + 1 auxiliary spin"
"49-spin returns -> auxiliary normalization"
"-> 48 logical spins"
"Hardware validation pending"
The prior must have an arrow into this green box and the green box must have an arrow OUT to the existing common "48 spins s -> z = (s + 1)/2" box. Remove the old separate normalization box and all its arrows completely. Internal wording now clearly states normalization AFTER hardware sampling; no normalization arrow should point into the hardware sampler. Preserve the classical SA/Gibbs alternative and OR label.

3. Remove the vertical arrow from the Provided power forecast box into Calendar features. Forecast and Calendar are separate inputs, each pointing only into LS-fitted preprocessing (the forecast also still points to the anchor). Keep both independent arrows into preprocessing and the forecast-to-anchor arrow.

Do not change anything else. In particular do not connect encoder output through the prior sampler, and do not remove observation noise, conditional field h(c), shared J, Raw/Cal/ECC distinction, or hardware validation pending label.

