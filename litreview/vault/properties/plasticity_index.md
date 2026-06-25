---
title: "plasticity index"
category: classification
influence_score: 172.97
n_papers: 26
already_derived: False
leaky_for_spt: False
derivable_from_inputs: True
tags: [property, geotech]
---
# plasticity index

**Category:** classification · **Influence (citation-weighted):** 172.97 across 26 papers

**Status:** 🆕 not yet derived · 🟢 non-leaky · 🔧 derivable from our inputs

*Also called:* Plasticity index; Atterberg limits / plasticity index; Atterberg limits / plasticity; Atterberg limits (plasticity)

## Derivation
PI = plasticity_for(USCS).pi from the engine's builtin_plasticity table (CL 15, CH 40, ML 4, MH 20, OL 8, OH 25, SC 12, SM 3, GC 10, GM 2, clean S/G 0). Already computed internally and consumed by Su; just emit it as a column.  [Casagrande plasticity chart; FHWA-NHI-06-089 index defaults]

## Notes
⭐ Phase-A shortlist. High influence (173), NON-leaky, and the engine ALREADY has the USCS->PI table in config.rs (used for Su) but never writes PI out. Cheapest high-value add: a real per-interval index feature. On shortlist.

## Evidence papers (8)
- [[W1490066004|soil mechanics in engineering practice]] (4,980 cites)
- [[W1693096119|Fundamentals of Soil Behavior]] (3,023 cites)
- [[W4296421648|Soil mechanics in engineering practice]] (2,539 cites)
- [[W2000534137|Fundamentals of Soil Behavior]] (2,384 cites)
- [[W1577484784|Principles of Geotechnical Engineering]] (1,830 cites)
- [[W2056990691|Effect of Soil Plasticity on Cyclic Response]] (1,809 cites)
- [[W1509784305|Principles of Foundation Engineering]] (1,245 cites)
- [[W1508673420|Manual on estimating soil properties for foundation design]] (1,218 cites)
