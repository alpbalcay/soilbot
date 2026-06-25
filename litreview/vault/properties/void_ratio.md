---
title: "void ratio"
category: state
influence_score: 374.76
n_papers: 58
already_derived: False
leaky_for_spt: False
derivable_from_inputs: True
tags: [property, geotech]
---
# void ratio

**Category:** state · **Influence (citation-weighted):** 374.76 across 58 papers

**Status:** 🆕 not yet derived · 🟢 non-leaky · 🔧 derivable from our inputs

*Also called:* Void ratio; Porosity; Void ratio / porosity; void ratio

## Derivation
Only as a coarse USCS+stress estimate: pick e0 from a USCS-keyed table (GW~0.3, GP~0.4, SW~0.5, SP~0.6, SM~0.6, SC~0.7, ML~0.8, CL~0.9, MH~1.3, CH~1.2, OL/OH~1.5, PT~3-10), optionally compressed with depth via e = e0 - Cc*log10(sigma'_v0/sigma'_ref) using a USCS Cc. Without lab water content / specific gravity it is essentially a constant per USCS class.  [Lambe & Whitman (1969) typical e ranges; Holtz & Kovacs (1981) USCS index tables]

## Notes
Rank-1 influence but for the SOIL-TYPE GNN a USCS-keyed e0 carries no information beyond the USCS one-hot/embedding the model already has. Real value would need lab w% or Gs which the OCR DB lacks. Low marginal info; not on shortlist.

## Evidence papers (8)
- [[W1490066004|soil mechanics in engineering practice]] (4,980 cites)
- [[W113831185|Critical State Soil Mechanics]] (3,072 cites)
- [[W1693096119|Fundamentals of Soil Behavior]] (3,023 cites)
- [[W2060905767|Fundamentals of Soil Mechanics]] (2,746 cites)
- [[W2000534137|Fundamentals of Soil Behavior]] (2,384 cites)
- [[W2090155884|A state parameter for sands]] (1,923 cites)
- [[W1577484784|Principles of Geotechnical Engineering]] (1,830 cites)
- [[W1964306993|On the compressibility and shear strength of natural clays]] (1,723 cites)
