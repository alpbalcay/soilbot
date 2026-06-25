---
title: "pore pressure"
category: hydraulic
influence_score: 191.43
n_papers: 30
already_derived: True
leaky_for_spt: False
derivable_from_inputs: True
tags: [property, geotech]
---
# pore pressure

**Category:** hydraulic · **Influence (citation-weighted):** 191.43 across 30 papers

**Status:** ✅ already derived · 🟢 non-leaky · 🔧 derivable from our inputs

*Also called:* Pore water pressure; Excess pore water pressure; Pore-water pressure; Pore-water salt concentration

## Derivation
u = gamma_w * max(z - gw_depth, 0); computed in stress.rs (pore_pressure_tsf) and already subtracted to form sigma_eff_v0_tsf, but NOT emitted as its own strata_derived column.  [hydrostatic; Terzaghi]

## Notes
Hydrostatic u is implicitly in the model (via sigma_eff) but is trivially cheap to expose explicitly. Marginal since it is collinear with depth-gw; could be added but low priority. Not on the core shortlist.

## Evidence papers (8)
- [[W2055072688|General Theory of Three-Dimensional Consolidation]] (9,578 cites)
- [[W2010804528|The Pore-Pressure Coefficients <i>A</i> and <i>B</i>]] (1,662 cites)
- [[W1500081692|Cone Penetration Testing in Geotechnical Practice]] (1,238 cites)
- [[W2032674078|Soil classification using the cone penetration test]] (1,111 cites)
- [[W2292650318|Soil Behaviour in Earthquake Geotechnics]] (973 cites)
- [[W1992656360|Long-Term Stability of Clay Slopes]] (965 cites)
- [[W1981474160|Evaluation of Liquefaction Potential Using Field Performance Data]] (915 cites)
- [[W1499374311|Measurement of Soil Properties in the Triaxial Test]] (885 cites)
