# Classification knowledge from the literature → improving our soil-classification data

*A second swarm read the harvested geotechnical corpus through a different lens — not "which
properties are influential" but "what does the classification literature tell us to fix in our
soil-classification data." 46 classification-relevant papers, 11 with usable content, 32 extracted
rules, synthesized into prioritized actions and applied to our `strata` table.*

## Headline finding — our USCS distribution is a parser artifact, and the literature proves it

Our 116,142 USCS-labelled intervals contain **only 12 tokens**, and the distribution is
geologically impossible:

| token | count | | token | count |
|---|--:|---|---|--:|
| ML | 32,780 | | PT | 7,346 |
| SP | 32,576 | | FILL | 5,282 |
| GP | 14,877 | | GM | 2,120 |
| SM | 12,433 | | OL | 484 |
| CL | 7,618 | | SC | 441 |
|  |  | | GC | 184 |
|  |  | | **CH** | **1** |
|  |  | | **MH / SW / GW / OH** | **0** |

The classification literature (Casagrande 1948, *Classification and Identification of Soils*,
`[[W2469166441]]`) is explicit: fine-grained soils split into clay/silt/organic by the **A-line**
(PI = 0.73·(LL−20)) and into low/high plasticity at **LL = 50**; coarse soils split into
well-graded/poorly-graded by gradation. A real multi-boring soil population — and NJ's coastal plain
in particular — **must** contain a spread of H-suffix (CH, MH, OH) and W-suffix (SW, GW) classes. A
distribution with **CH = 1 and a clean zero for MH/SW/GW/OH is statistically impossible for natural
geology** — it is the fingerprint of a classifier that cannot emit those symbols.

**Confirmed mechanism (code-level).** Most of our USCS values do not come from clean codes on the
logs; NJDOT logs are *descriptive* ("moist, m.-f. SAND, some silt"), so `pipeline/parse_logs.py`
maps phrases to USCS via `_phrase_to_uscs()`. That heuristic structurally **can only emit
poorly-graded / low-plasticity classes**: sand → SP/SM (never SW), gravel → GP/GM (never GW), silt →
ML (never MH), clay → CL unless the text literally says "fat"/"high plasticity" (→ the lone CH). The
W (well-graded) and H (high-plasticity) distinctions require gradation/Atterberg data the
descriptions rarely state, so they collapse to P and L. The literature's predicted collapse —
SW→SP, GW→GP, MH→ML, CH→CL, OH→OL — is exactly what the code does.

**Scale of the artifact** (`scripts/classify_audit.py` → `strata_quality` table):
- **88,335 / 116,142 (76%)** USCS intervals are the degenerate low/poorly-graded variants
  (ML/SP/GP/CL/OL) that absorb the missing classes' mass.
- **1** interval in all 116k carries an H/W/O suffix.

This is the single most important classification-data improvement the literature surfaced: our USCS
field conflates plasticity/gradation it never measured, so any model using it (and our B1 SPT-N model
keys σ′v0/γ off USCS) sees a degraded label. **The honest fix is not to rewrite classes** (we lack
the underlying gradation/Atterberg to know which CL is really CH) but to **improve the parser** and
re-OCR, and meanwhile to **flag** the affected intervals.

## Applied checks (machine-applicable) — results on our data
Written to a non-destructive `strata_quality` table (one row per interval, with boolean flags):

| check | rule & source | flagged |
|---|---|--:|
| **Illegal USCS token** | must be an ASTM-D2487 symbol (Casagrande `[[W2469166441]]`) | 0 (all legal; `FILL` is a project token) |
| **Fine-grained + high SPT-N** | fine class (ML/CL/OL/CH/MH/OH) with N>50 → likely weathered rock / drilling refusal mislabel | **508** (81 with N>100) |
| **Suffix-ambiguous** | class in ML/SP/GP/CL/OL where W/H/O may be collapsed | **88,335** |
| **Liquefiable saturated loose granular** | granular + below groundwater + (N1)60≈<15 (Vaid/Robertson `[[W2055836363]]`) | 15 (blocked by sparse groundwater) |
| **Groundwater fill-down** | one water table per boring → propagate min gw to its intervals | +591 intervals recoverable |

## The extracted rule set (32 rules, by category)
- **USCS decision criteria** (Casagrande `[[W2469166441]]`): A-line PI=0.73(LL−20) separates
  clay/silt; LL=50 separates low/high plasticity; ~50% passing No. 200 separates coarse/fine; coarse
  named by gradation, fine by plasticity. The **U-line** PI=0.9(LL−8) bounds physically plausible
  Atterberg data (flag points above it as test/transcription error).
- **CPT soil-behaviour-type** (Robertson `[[W2032674078]]`, `[[W2527105207]]`, `[[W2045572265]]`):
  SBT is a *behavioural* classification (Qt–Fr–Bq / Ic), distinct from but correlated with USCS
  grain-size class; the 9-zone Qt–Fr crosswalk maps to sensitive-fine → clays → silt-mixtures →
  sand-mixtures → sands → gravelly-sand. Not applicable now (no CPT), recorded as a future-ingestion
  target.
- **Index-property correlations** (Wroth & Wood `[[W2004941813]]`): undrained strength ~1.7 kPa at LL
  and ~100× that at PL; Cc ∝ PI — sanity checks once Atterberg is captured.
- **Behavioural / liquefaction** (`[[W2055836363]]`): saturated loose granular below the water table
  is contractive/liquefiable; the same dense is not — a drainage/SPT consistency check.

## Recommendations (prioritized)
1. **[HIGH] Fix the description→USCS heuristic** in `pipeline/parse_logs.py::_phrase_to_uscs`:
   detect "well[- ]graded"/"poorly[- ]graded" → W/P suffix, "fat"/"high[- ]plasticity"/"CH" → H,
   "organic"/"peaty" → O prefix, and emit dual symbols on borderline fines. Re-OCR (or re-parse
   cached text) to populate the missing CH/MH/SW/GW/OH. This is the root-cause fix; everything else is
   mitigation.
2. **[HIGH] Carry the `strata_quality` flags into modeling** — exclude or down-weight the 508
   fine+high-SPT intervals (weathered rock/refusal) and treat `suffix_ambiguous` USCS as the coarse
   parent class (sand/silt/clay/gravel) rather than the spurious P/L precision.
3. **[HIGH] Capture Atterberg / %-fines / gradation** as OCR targets when present on logs — they are
   the inputs every USCS decision criterion needs and the only way to recover true H/W classes.
4. **[MED] Propagate groundwater per boring** (+591 intervals now; add the SSURGO water-table
   crosswalk for borings with no measured gw) to unblock the liquefaction/drainage checks.
5. **[MED] Build an NJDOT-engineering-code ↔ USCS crosswalk** from co-located labels to cross-validate
   OCR'd classes (e.g. marsh/organic codes ↔ PT/OH).

## Honest scope
The degenerate-distribution finding is the real prize and is **verified at code level**, not merely
inferred. What is *fixable today* is flagging (`strata_quality`) and the groundwater fill-down; the
*root-cause* fix (recovering CH/MH/SW/GW/OH) needs an improved parser + re-OCR, because we cannot
invent the gradation/Atterberg that distinguishes the collapsed classes. No classes were rewritten —
that would fabricate measurements we never made.

## Reproduce
```
# swarm: Workflow scripts/classification_swarm.workflow.js  (args = our USCS distribution)
python scripts/classify_audit.py     # -> strata_quality flag table + summary counts
```
