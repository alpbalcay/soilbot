---
title: Geotechnical Literature Vault
tags: [index]
---
# Geotechnical Literature → Soil Properties

Harvested **202 foundational geotechnical papers** (OpenAlex, citation-ranked) and the **27 soil properties** they establish. Links below open the Obsidian graph view.

## Most-cited papers
- [[W2162604832|A Closed‐form Equation for Predicting the Hydraulic Conductivity of Unsaturated ]] — 27,363 cites (1980)
- [[W2055072688|General Theory of Three-Dimensional Consolidation]] — 9,578 cites (1941)
- [[W1977727715|Theoretical Soil Mechanics]] — 8,569 cites (1943)
- [[W1490066004|soil mechanics in engineering practice]] — 4,980 cites (2014)
- [[W2314167995|Geotechnical Earthquake Engineering]] — 3,809 cites (1997)
- [[W113831185|Critical State Soil Mechanics]] — 3,072 cites (1968)
- [[W2096885477|Geotechnical earthquake engineering]] — 3,067 cites (2008)
- [[W1693096119|Fundamentals of Soil Behavior]] — 3,023 cites (2025)
- [[W1490718145|Simplified Procedure for Evaluating Soil Liquefaction Potential]] — 2,875 cites (1971)
- [[W2060905767|Fundamentals of Soil Mechanics]] — 2,746 cites (1948)
- [[W2166819294|The strength and dilatancy of sands]] — 2,672 cites (1986)
- [[W4296421648|Soil mechanics in engineering practice]] — 2,539 cites (1996)
- [[W2137407452|A constitutive model for partially saturated soils]] — 2,442 cites (1990)
- [[W2000534137|Fundamentals of Soil Behavior]] — 2,384 cites (1994)
- [[W1560408291|Nonlinear Analysis of Stress and Strain in Soils]] — 2,376 cites (1970)
- [[W2145185680|Characterization of geotechnical variability]] — 2,331 cites (1999)
- [[W1491085163|Shear Modulus and Damping in Soils: Design Equations and Curves]] — 2,254 cites (1972)
- [[W1978304314|Fundamentals of Soil Behavior]] — 2,085 cites (1976)
- [[W1587662141|Pile foundation analysis and design]] — 1,993 cites (1981)
- [[W2120311801|Liquefaction Resistance of Soils: Summary Report from the 1996 NCEER and 1998 NC]] — 1,973 cites (2001)
- [[W2090155884|A state parameter for sands]] — 1,923 cites (1985)
- [[W3019775124|Foundation Analysis and Design]] — 1,903 cites (2006)
- [[W2070394663|Microbial Carbonate Precipitation as a Soil Improvement Technique]] — 1,879 cites (2007)
- [[W1577484784|Principles of Geotechnical Engineering]] — 1,830 cites (2020)
- [[W2055836363|Liquefaction and flow failure during earthquakes]] — 1,811 cites (1993)

## Soil properties by literature influence
- [[void_ratio|void ratio]] — influence 374.76 / 58 papers · **⭐ shortlist**
- [[friction_angle|friction angle]] — influence 347.8 / 52 papers · already derived
- [[undrained_shear_strength|undrained shear strength]] — influence 282.24 / 43 papers · already derived
- [[permeability|permeability]] — influence 276.82 / 41 papers · **⭐ shortlist**
- [[effective_stress|effective stress]] — influence 264.38 / 41 papers · already derived
- [[compressibility|compressibility]] — influence 230.27 / 34 papers · **⭐ shortlist**
- [[relative_density|relative density]] — influence 215.36 / 33 papers · already derived
- [[shear_modulus|shear modulus]] — influence 214.69 / 33 papers · **⭐ shortlist**
- [[ocr|ocr]] — influence 209.58 / 32 papers · **⭐ shortlist**
- [[pore_pressure|pore pressure]] — influence 191.43 / 30 papers · already derived
- [[plasticity_index|plasticity index]] — influence 172.97 / 26 papers · **⭐ shortlist**
- [[spt_n|spt n]] — influence 165.84 / 26 papers
- [[fines_content|fines content]] — influence 160.61 / 25 papers · **⭐ shortlist**
- [[liquefaction_resistance|liquefaction resistance]] — influence 142.53 / 21 papers · already derived
- [[shear_wave_velocity|shear wave velocity]] — influence 89.4 / 13 papers · **⭐ shortlist**
- [[compression_index|compression index]] — influence 87.09 / 12 papers · **⭐ shortlist**
- [[k0|k0]] — influence 86.47 / 13 papers · **⭐ shortlist**
- [[preconsolidation_stress|preconsolidation stress]] — influence 81.28 / 12 papers · **⭐ shortlist**
- [[coefficient_consolidation|coefficient consolidation]] — influence 74.77 / 10 papers · **⭐ shortlist**
- [[dilatancy|dilatancy]] — influence 68.39 / 11 papers · **⭐ shortlist**
- [[unit_weight|unit weight]] — influence 66.58 / 10 papers · already derived
- [[cpt_qc|cpt qc]] — influence 66.35 / 10 papers
- [[damping_ratio|damping ratio]] — influence 62.19 / 9 papers · **⭐ shortlist**
- [[youngs_modulus|youngs modulus]] — influence 59.4 / 9 papers · already derived
- [[bearing_capacity|bearing capacity]] — influence 51.84 / 8 papers · already derived
- [[liquid_limit|liquid limit]] — influence 35.85 / 6 papers · **⭐ shortlist**
- [[recompression_index|recompression index]] — influence 27.25 / 4 papers · **⭐ shortlist**

## ⭐ Phase-A information-gain shortlist
Non-leaky, not-yet-derived, derivable-from-our-inputs — candidates to add to the soil-type GNN:
- [[void_ratio|void ratio]] — Only as a coarse USCS+stress estimate: pick e0 from a USCS-keyed table (GW~0.3, GP~0.4, SW~0.5, SP~0.6, SM~0.6, SC~0.7, ML~0.8, CL~0.9, MH~1.3, CH~1.2, OL/OH~1.5, PT~3-10), optionally compressed with depth via e = e0 - Cc*log10(sigma'_v0/sigma'_ref) using a USCS Cc. Without lab water content / specific gravity it is essentially a constant per USCS class.  [Lambe & Whitman (1969) typical e ranges; Holtz & Kovacs (1981) USCS index tables]
- [[permeability|permeability]] — USCS-keyed hydraulic conductivity (log10 k, m/s), a robust physically real per-class signal: GW -1, GP -1.5, GM/GC -6, SW -3, SP -3.5, SM -5, SC -7, ML -7, CL -9, CH -10, OL/OH -8, PT -6. If clean coarse-grained and D10 is estimable, refine sands with Hazen k(cm/s)=C*D10(mm)^2, C~100. Emit log10_k as the feature (spans ~10 orders of magnitude).  [Terzaghi, Peck & Mesri (1996) Table; Hazen (1911); USACE EM-1110-2-1901]
- [[compressibility|compressibility]] — Represent via compression index Cc (see compression_index): Cc from USCS or Cc=0.009*(LL-10) (Terzaghi-Peck) using USCS-keyed LL, or mv = Cc/(2.3*sigma'_v0*(1+e0)). Constrained modulus M is the inverse but the engine's M (m_constrained_tsf) is the leaky SPT version.  [Terzaghi & Peck (1967); Skempton (1944) Cc-LL]
- [[shear_modulus|shear modulus]] — Small-strain Gmax = rho*Vs^2 with Vs from a USCS+stress correlation, or empirical Gmax = A*(OCR^k)*sigma'_m^n / (0.3+0.7e^2) (Hardin & Black). Without Vs, e, or OCR it reduces to Gmax ~ f(sigma'_v0) with USCS constants.  [Hardin & Black (1968); Seed & Idriss (1970)]
- [[ocr|ocr]] — Only as a depth/geology heuristic: OCR = sigma'_p / sigma'_v0 with sigma'_p estimated from a constant crust preconsolidation or a regional unloading offset; or Mayne's OCR=0.5*(N60/sigma'_v0)... (that is SPT). Non-SPT route requires assuming a geologic preconsolidation profile not available per-interval in NJ.  [Mayne (2007) for SPT/CPT; Mesri for crust]
- [[plasticity_index|plasticity index]] — PI = plasticity_for(USCS).pi from the engine's builtin_plasticity table (CL 15, CH 40, ML 4, MH 20, OL 8, OH 25, SC 12, SM 3, GC 10, GM 2, clean S/G 0). Already computed internally and consumed by Su; just emit it as a column.  [Casagrande plasticity chart; FHWA-NHI-06-089 index defaults]
- [[fines_content|fines content]] — FC (% passing #200) = plasticity_for(USCS).fines_pct from builtin_plasticity (GW 2, SP 4, SM 18, SC 30, ML 75, CL 85, CH 92, ...). Already computed internally (used for n1_60_cs) but only emitted indirectly via crr; expose FC as its own column.  [USCS definition (Casagrande); FHWA index defaults]
- [[shear_wave_velocity|shear wave velocity]] — Vs = a*sigma'_v0^b with USCS-typical a,b, or Vs from N60 (Vs=97*N60^0.314, leaky). Non-SPT route is only Vs ~ f(sigma'_v0, USCS-constant) which duplicates effective_stress+USCS.  [Andrus & Stokoe; Ohta & Goto (1978)]
- [[compression_index|compression index]] — Cc = 0.009*(LL-10) (Terzaghi & Peck) using USCS-keyed LL, or USCS-typical Cc (SP/SW ~0.02, ML ~0.1, CL ~0.25, CH ~0.5, OH/PT ~0.6-1.0). For overconsolidated/recompression use Cr~0.1-0.2*Cc.  [Terzaghi & Peck (1967); Skempton (1944)]
- [[k0|k0]] — Jaky NC: K0 = 1 - sin(phi'), with phi' from a USCS-typical drained friction angle (SW/SP ~34-36 -> K0~0.42-0.44; CL ~28 -> K0~0.53; CH ~22 -> K0~0.63). If an OCR estimate existed, K0_OC = (1-sin phi')*OCR^sin(phi') (Mayne & Kulhawy), but use the NC form given no reliable OCR.  [Jaky (1944); Mayne & Kulhawy (1982)]
- [[preconsolidation_stress|preconsolidation stress]] — sigma'_p = OCR * sigma'_v0; non-SPT only via an assumed crust/geology OCR profile (same obstacle as OCR). Mayne SPT form sigma'_p=0.47*N60^m*Pa is leaky.  [Mayne (2007); Mesri crust models]
- [[coefficient_consolidation|coefficient consolidation]] — cv = k/(mv*gamma_w) = k*(1+e0)/(av*gamma_w). Buildable from USCS-keyed k (permeability) and Cc/mv, but it is a deterministic combination of permeability + compressibility, both already USCS-keyed.  [Terzaghi 1D consolidation]
- [[dilatancy|dilatancy]] — psi = phi' - phi'_cv (Bolton 1986: psi proportional to relative dilatancy index IR), needs Dr/relative density which is SPT-bound. Non-SPT version is a USCS-typical psi constant (dense sand ~5-15 deg, clay ~0).  [Bolton (1986)]
- [[damping_ratio|damping ratio]] — Small-strain damping Dmin from a USCS/PI-keyed table (sand ~0.5-1%, high-PI clay ~2-3%); curves keyed to PI (Darendeli 2001).  [Darendeli (2001); Vucetic & Dobry (1991)]
- [[liquid_limit|liquid limit]] — LL from USCS class via PI and the A-line: for cohesive soils LL ~= PI/0.73 + 20 (A-line PI=0.73(LL-20)); or direct USCS-typical LL (CL ~35, CH ~60, ML ~28, MH ~55, OL ~35, OH ~65, clean sands/gravels NP/0). Uses the engine's existing USCS->PI table.  [Casagrande A-line / plasticity chart]
- [[recompression_index|recompression index]] — Cr (swell/recompression index) ~ 0.1-0.2*Cc, with Cc from USCS or Cc=0.009*(LL-10). USCS-typical Cr: clean sand ~0, CL ~0.03-0.05, CH ~0.06-0.1, OH/PT higher.  [Terzaghi, Peck & Mesri (1996); Lambe & Whitman]