---
title: "fines content"
category: classification
influence_score: 160.61
n_papers: 25
already_derived: False
leaky_for_spt: False
derivable_from_inputs: True
tags: [property, geotech]
---
# fines content

**Category:** classification · **Influence (citation-weighted):** 160.61 across 25 papers

**Status:** 🆕 not yet derived · 🟢 non-leaky · 🔧 derivable from our inputs

*Also called:* Fines content; Fines / silt content; Fines content (silty sands); Fines content / grain characteristics

## Derivation
FC (% passing #200) = plasticity_for(USCS).fines_pct from builtin_plasticity (GW 2, SP 4, SM 18, SC 30, ML 75, CL 85, CH 92, ...). Already computed internally (used for n1_60_cs) but only emitted indirectly via crr; expose FC as its own column.  [USCS definition (Casagrande); FHWA index defaults]

## Notes
⭐ Phase-A shortlist. Influence 160.6, NON-leaky, table already exists in config.rs. Coarse/fine boundary signal complementary to USCS embedding and continuous-valued. On shortlist.

## Evidence papers (8)
- [[W2120311801|Liquefaction Resistance of Soils: Summary Report from the 1996 NCEER and 1998 NC]] (1,973 cites)
- [[W2090155884|A state parameter for sands]] (1,923 cites)
- [[W2055836363|Liquefaction and flow failure during earthquakes]] (1,811 cites)
- [[W2041893233|Influence of SPT Procedures in Soil Liquefaction Resistance Evaluations]] (1,325 cites)
- [[W2043681414|Evaluating cyclic liquefaction potential using the cone penetration test]] (1,198 cites)
- [[W1981474160|Evaluation of Liquefaction Potential Using Field Performance Data]] (915 cites)
- [[W2015320417|Residual strength of clays in landslides, folded strata and the laboratory]] (890 cites)
- [[W4240036724|Liquefaction Resistance of Soils: Summary Report from the 1996 NCEER and 1998 NC]] (796 cites)
