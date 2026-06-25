# Literature-driven soil-property discovery → GNN information-gain test

*An orchestrated experiment: web-scrape foundational geotechnical papers, rank the most influential
soil properties, derive the non-redundant ones, and test whether they add information to the soil-type
graph network. The headline result is an honest **null**: they do not.*

## Pipeline
1. **Harvest** (`pipeline/litreview.py`, phase 7) — OpenAlex, ~40 canonical seed queries (Terzaghi,
   Casagrande, Skempton, Peck, Seed-Idriss, Robertson, Bolton, Mesri, Hazen, …), one hop of
   co-citation expansion, ranked by `cited_by_count`. A topical relevance gate (allow geotech terms,
   veto seismology/InSAR/ML/chemistry) pruned 98 high-citation off-domain blockbusters that loose
   full-text search dragged in. **Result: 202 papers, 450 in-set citation edges, 20 open-access full
   texts** (most foundational papers predate OA → extraction leaned on abstracts + OpenAlex concepts).
2. **Swarm** (`scripts/lit_swarm.workflow.js`) — 60 agents (batched ~5 papers each) extracted the soil
   properties each paper concerns; a deterministic citation-weighted ranking (alias-folded to a
   canonical vocabulary) produced **27 ranked properties**; a gap-analysis agent classified each
   against our `strata_derived` columns and the SAFE/LEAKY split.
3. **Vault** (`litreview/vault/`, committed) — 202 paper notes + 27 property notes + index, linked by
   `[[wikilinks]]` (citation graph + paper↔property edges) for Obsidian's graph view.
4. **Derive + ablate** (`ml/geotech_features.py`, `ml/train.py --geotech`) — the shortlisted
   properties, computed per boring from its OCR'd USCS profile + depth + groundwater, added as
   Phase-A boring-node features; spatial-block 5-fold CV with vs without.

## Most influential soil properties (citation-weighted)
| # | property | category | influence | papers | already derived | leaky for SPT | derivable |
|--:|---|---|--:|--:|:-:|:-:|:-:|
| 1 | void ratio | state | 374.8 | 58 | – | – | ✓ |
| 2 | friction angle φ′ | strength | 347.8 | 52 | ✓ | ✓ | ✓ |
| 3 | undrained shear strength Su | strength | 282.2 | 43 | ✓ | ✓ | ✓ |
| 4 | permeability k | hydraulic | 276.8 | 41 | – | – | ✓ |
| 5 | effective stress σ′v | stress | 264.4 | 41 | ✓ | – | ✓ |
| 6 | compressibility | compressibility | 230.3 | 34 | – | – | ✓ |
| 7 | relative density Dr | state | 215.4 | 33 | ✓ | ✓ | – |
| 8 | shear modulus Gmax | dynamic | 214.7 | 33 | – | – | ✓ |
| 9 | OCR / stress history | stress | 209.6 | 32 | – | – | ✓ |
| 10 | pore pressure | hydraulic | 191.4 | 30 | ✓ | – | ✓ |
| 11 | plasticity index PI | classification | 173.0 | 26 | – | – | ✓ |
| 13 | fines content | classification | 160.6 | 25 | – | – | ✓ |

The field is **SPT-saturated**: the top strength/state properties (φ′, Su, Dr, (N₁)₆₀, CRR) are all
already derived in `strata_derived` **and** are functions of measured SPT-N — leaky for the SPT-N task.
The user-chosen **soil-type** target makes them non-leaky (they sit on boring nodes, the target on
label nodes), so the experiment instead asks which properties carry *new* signal.

## Gap-analysis shortlist (non-leaky, not-yet-derived, derivable)
**plasticity index · fines content · permeability (log₁₀k) · K₀ · liquid limit · recompression index.**
The gap agent's own honest caveat: these are **USCS-keyed constants** (lookup by soil class), so
*"void ratio, compression index, OCR and friction angle are derivable only as USCS-keyed constants or
stress heuristics — they add little beyond a USCS one-hot the GNN already sees."* Implemented in
`ml/geotech_features.py` from standard references the harvest surfaced (Terzaghi-Peck-Mesri permeability
table; Hazen; USCS plasticity/fines defaults; Jaky/Brooker-Ireland K₀), as 10 per-boring aggregates
(thickness-weighted whole-profile + near-surface means) over the 27,580 borings with a USCS profile.

## Information-gain result — Phase-A soil-type, spatial-block 5-fold CV
| metric | baseline (geology+SSURGO) | + literature geotech | Δ |
|---|--:|--:|--:|
| macro-F1 | 0.2684 | 0.2682 | **−0.0002** |
| balanced accuracy | 0.3234 | 0.3182 | −0.0052 |
| accuracy | 0.4038 | 0.4108 | +0.0069 |
| NLL | 1.7834 | 1.7774 | −0.0060 |
| ECE | 0.0247 | 0.0205 | −0.0043 |

**Verdict: no discriminative information added.** Macro-F1 is flat to three decimals; per-fold deltas
are firmly within noise (geotech wins folds 0 & 4, loses fold 3, ties elsewhere). The only consistent
movement is a **small calibration improvement** (NLL −0.006, ECE −0.004) — the geotech block nudges the
posterior slightly sharper but no better at ranking classes.

## Why — and why this is the *expected* honest outcome
The literature-influential properties that are **not SPT-derived** reduce, for our data, to
**USCS-keyed constants**. The Phase-A GNN already ingests surficial/bedrock geology and SSURGO
component/drainage/hydrologic-group per node, which are strongly correlated with soil class — so a
property that is a deterministic function of USCS (itself correlated with geology) is largely
**redundant** with features the model already has. The experiment was designed to detect exactly this,
and it did: the *most influential* geotech properties are either already derived, leaky for SPT, or
informationally redundant with geology for soil-type. The genuine non-redundant signal geology can't
provide remains the **depth-resolved SPT-N** profile (the B1 model), not these aggregates.

## Caveats
- Influence = raw OpenAlex citation count (not field-normalized); textbooks/blockbusters rank high.
- OA full text only 20/202; property extraction leaned on abstracts + concept tags.
- Geotech aggregates use USCS-keyed literature defaults, not measured lab Atterberg/grain-size (NJDOT
  logs rarely report them) — so they encode the USCS profile, not independent lab measurements.
- A single alias-folding pass (canonical vocabulary) underlies the ranking; a substring-collision bug
  was found and fixed (symbols matched as whole tokens, ≥2 chars) before the reported run.

## Reproduce
```
python -m pipeline.run --phase 7                 # harvest -> lit_* tables + metadata/pdf caches
# (swarm) Workflow scripts/lit_swarm.workflow.js  -> ranking + gap analysis
python scripts/persist_props.py /tmp/lit_swarm_out.json   # -> lit_properties / lit_property_links
python scripts/build_vault.py                    # -> litreview/vault/ (Obsidian)
python -c "from pipeline.config import Config; from ml.data import build_and_cache; build_and_cache(Config.load(None))"
python -m ml.train --mode a3 --folds 5 --tag a3_geo_base   # baseline
python -m ml.train --mode a3 --folds 5 --geotech --tag a3_geotech   # + literature geotech
```
