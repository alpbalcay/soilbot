---
title: "effective stress"
category: stress
influence_score: 264.38
n_papers: 41
already_derived: True
leaky_for_spt: False
derivable_from_inputs: True
tags: [property, geotech]
---
# effective stress

**Category:** stress · **Influence (citation-weighted):** 264.38 across 41 papers

**Status:** ✅ already derived · 🟢 non-leaky · 🔧 derivable from our inputs

*Also called:* Effective stress; Effective overburden pressure; Mean effective stress; Effective stress / mean effective stress

## Derivation
sigma_eff_v0_tsf = integral(gamma(USCS,z) dz) - gamma_w*max(z-gw,0), already integrated on a 1-ft grid in stress.rs and emitted as sigma_eff_v0_tsf (also sigma_v0_tsf total).  [Terzaghi effective stress principle]

## Notes
Already derived and NON-leaky. No new work needed; it is a model input already. Not on shortlist (already present).

## Evidence papers (8)
- [[W1977727715|Theoretical Soil Mechanics]] (8,569 cites)
- [[W113831185|Critical State Soil Mechanics]] (3,072 cites)
- [[W1490718145|Simplified Procedure for Evaluating Soil Liquefaction Potential]] (2,875 cites)
- [[W2060905767|Fundamentals of Soil Mechanics]] (2,746 cites)
- [[W2166819294|The strength and dilatancy of sands]] (2,672 cites)
- [[W2090155884|A state parameter for sands]] (1,923 cites)
- [[W1577484784|Principles of Geotechnical Engineering]] (1,830 cites)
- [[W1698701323|Soil behaviour and critical state soil mechanics]] (1,274 cites)
