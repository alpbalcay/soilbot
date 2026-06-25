export const meta = {
  name: 'lit-property-swarm',
  description: 'Swarm-extract soil properties from harvested geotech papers, citation-weight rank, gap-analyze vs strata_derived for the Phase-A information-gain test',
  phases: [
    { title: 'Load', detail: 'query lit_papers for the id + citation list' },
    { title: 'Extract', detail: 'fan out agents over paper batches -> property mentions' },
    { title: 'GapAnalysis', detail: 'classify ranked properties vs our derived columns + leakage' },
  ],
}

// args = { derived_cols: [...], safe_cols: [...], leaky_cols: [...] }  // from strata_derived / ml/data3d.py
const DERIVED = (args && args.derived_cols) || []
const SAFE = (args && args.safe_cols) || []
const LEAKY = (args && args.leaky_cols) || []

// ---- Load: one agent dumps the harvested paper id+citation list from DuckDB ----------------
const LOAD_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    papers: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: { id: { type: 'string' }, cites: { type: 'integer' } },
        required: ['id', 'cites'],
      },
    },
  },
  required: ['papers'],
}
const loaded = await agent(
  `Run this exact command from the repo root and return its output verbatim as the schema `
  + `(it prints a JSON object {"papers":[{"id","cites"},...]} for all harvested papers):\n\n`
  + `.venv/bin/python -c "import json,duckdb; c=duckdb.connect('data/soilbot.duckdb',read_only=True); `
  + `print(json.dumps({'papers':[{'id':r[0],'cites':r[1]} for r in `
  + `c.execute('SELECT openalex_id, cited_by_count FROM lit_papers ORDER BY cited_by_count DESC').fetchall()]}))"`,
  { label: 'load-papers', phase: 'Load', schema: LOAD_SCHEMA }
)
const PAPERS = (loaded && loaded.papers) || []
if (!PAPERS.length) { return { error: 'loader returned no papers' } }
log(`loaded ${PAPERS.length} papers`)

// ---- canonical soil-property vocabulary (deterministic alias folding) ----------------------
// Maps the many ways papers name a property to one canonical key so citation-weighted ranking
// aggregates correctly. Matching: descriptive PHRASES are substring-matched on the property NAME;
// SYMBOLS must equal the cleaned `symbol` field as a whole token and be >= 2 chars (so 'dr' can't
// match "hy-dr-aulic" and 'id' can't match "consol-id-ation"). ORDER is significant — more specific
// keys come first (recompression before compression; ocr/preconsolidation before consolidation).
const CANON = [
  ['friction_angle', {phrases:['friction angle','angle of internal friction','angle of shearing resistance','shearing resistance angle','effective friction','drained friction'], symbols:['phi']}],
  ['relative_density', {phrases:['relative density','density index'], symbols:['dr']}],
  ['undrained_shear_strength', {phrases:['undrained shear strength','undrained strength','undrained cohesion','shear strength of clay','undrained su'], symbols:['su','cu']}],
  ['recompression_index', {phrases:['recompression index','recompression','swelling index','swell index'], symbols:['cr','cs']}],
  ['compression_index', {phrases:['compression index','virgin compression','compressibility index'], symbols:['cc']}],
  ['compressibility', {phrases:['compressibility','volume change','volumetric compress'], symbols:['mv']}],
  ['coefficient_consolidation', {phrases:['coefficient of consolidation','consolidation coefficient'], symbols:['cv']}],
  ['ocr', {phrases:['overconsolidation ratio','overconsolidation','stress history'], symbols:['ocr']}],
  ['preconsolidation_stress', {phrases:['preconsolidation','yield stress','preconsolidation pressure'], symbols:['pc']}],
  ['liquefaction_resistance', {phrases:['cyclic resistance ratio','cyclic stress ratio','liquefaction resistance','cyclic strength','liquefaction triggering','cyclic shear stress'], symbols:['crr','csr']}],
  ['spt_n', {phrases:['standard penetration','penetration resistance','blow count','spt n','n-value','n value','corrected blow'], symbols:['n60','spt']}],
  ['cpt_qc', {phrases:['cone resistance','cone tip resistance','cone penetration resistance','tip resistance'], symbols:['qc','qt']}],
  ['plasticity_index', {phrases:['plasticity index','plasticity chart','atterberg'], symbols:['pi','ip']}],
  ['liquid_limit', {phrases:['liquid limit'], symbols:['ll','wl']}],
  ['fines_content', {phrases:['fines content','percent fines','passing 200','passing #200','fines fraction','silt and clay content','clay fraction'], symbols:['fc']}],
  ['void_ratio', {phrases:['void ratio','porosity'], symbols:['e0']}],
  ['unit_weight', {phrases:['unit weight','bulk density','dry density','specific weight','soil density'], symbols:['gamma']}],
  ['k0', {phrases:['earth pressure at rest','coefficient of earth pressure','lateral earth pressure coefficient','at-rest','lateral stress ratio'], symbols:['k0']}],
  ['effective_stress', {phrases:['effective stress','effective overburden','vertical effective stress'], symbols:[]}],
  ['pore_pressure', {phrases:['pore pressure','pore water pressure','pore-water','excess pore'], symbols:['ru']}],
  ['shear_modulus', {phrases:['shear modulus','small strain stiffness','small-strain stiffness','maximum shear modulus'], symbols:['gmax','g0']}],
  ['shear_wave_velocity', {phrases:['shear wave velocity','shear-wave velocity','s-wave velocity'], symbols:['vs']}],
  ['youngs_modulus', {phrases:["young's modulus",'elastic modulus','deformation modulus','elastic stiffness'], symbols:['es']}],
  ['permeability', {phrases:['permeability','hydraulic conductivity','coefficient of permeability'], symbols:['ksat']}],
  ['damping_ratio', {phrases:['damping ratio','material damping'], symbols:[]}],
  ['bearing_capacity', {phrases:['bearing capacity','allowable bearing','ultimate bearing'], symbols:['qult']}],
  ['dilatancy', {phrases:['dilatancy','dilation angle','angle of dilation'], symbols:[]}],
]
function canonicalize(name, symbol) {
  const n = (name || '').toLowerCase()
  const sym = (symbol || '').toLowerCase().replace(/[\s().,'’′`*]/g, '')
  for (const [key, def] of CANON) {            // 1) phrase match on the NAME (ordered, specific-first)
    if (def.phrases.some(p => n.includes(p))) return key
  }
  for (const [key, def] of CANON) {            // 2) whole-token symbol match (>= 2 chars)
    if ((def.symbols || []).some(s => s.length >= 2 && s === sym)) return key
  }
  return null  // unmapped -> ignored in ranking (keeps the vocabulary disciplined)
}

const EXTRACT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    papers: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          id: { type: 'string' },
          relevant: { type: 'boolean', description: 'true if the paper concerns soil properties/behavior' },
          one_line: { type: 'string', description: 'one-sentence plain summary of the paper\'s contribution' },
          properties: {
            type: 'array',
            items: {
              type: 'object', additionalProperties: false,
              properties: {
                name: { type: 'string' },
                symbol: { type: 'string' },
                category: { type: 'string', enum: ['strength','stiffness','compressibility','classification','state','hydraulic','stress','dynamic','liquefaction','other'] },
                role: { type: 'string', description: 'what the paper establishes about this property (a correlation, a method, a theory)' },
              },
              required: ['name','category','role'],
            },
          },
        },
        required: ['id','relevant','one_line','properties'],
      },
    },
  },
  required: ['papers'],
}

// ---- Extract: batch papers (~5/agent) and fan out ------------------------------------------
const BATCH = 5
const batches = []
for (let i = 0; i < PAPERS.length; i += BATCH) batches.push(PAPERS.slice(i, i + BATCH))
log(`extracting from ${PAPERS.length} papers in ${batches.length} batches`)

const extractResults = await parallel(batches.map((batch, bi) => () =>
  agent(
    `You are a geotechnical-engineering research assistant. For EACH paper below, read its cached `
    + `metadata at litreview/metadata/<id>.json (fields: title, abstract, concepts, year) and, if it `
    + `exists, the extracted full text at litreview/fulltext/<id>.txt (use Read; skip if missing).\n\n`
    + `Identify the SOIL PROPERTIES the paper concerns — the measurable geotechnical quantities it `
    + `defines, correlates, or builds a method around (e.g. undrained shear strength, friction angle, `
    + `compression index, relative density, K0, Gmax, permeability, OCR, plasticity/liquid limit, `
    + `fines content, void ratio, CRR). For each, give name, common symbol, a category, and the role `
    + `the paper assigns it. Mark relevant=false (empty properties) if the paper is not about soil `
    + `properties/behavior. Be precise; do not invent properties not discussed.\n\n`
    + `Paper ids (read each one's metadata/full-text file):\n` + batch.map(p => `- ${p.id}`).join('\n'),
    { label: `extract:b${bi}`, phase: 'Extract', schema: EXTRACT_SCHEMA }
  ).then(r => (r && r.papers) || [])
))

const perPaper = extractResults.filter(Boolean).flat()
const citeById = Object.fromEntries(PAPERS.map(p => [p.id, p.cites || 0]))
log(`extracted ${perPaper.length} paper records`)

// ---- Rank: citation-weighted influence per canonical property ------------------------------
const agg = {}  // canon -> {score, papers:Set, names:Set, categories:Counter}
for (const rec of perPaper) {
  if (!rec || !rec.relevant) continue
  const w = Math.log(1 + (citeById[rec.id] || 0))
  const seen = new Set()
  for (const pr of (rec.properties || [])) {
    const key = canonicalize(pr.name, pr.symbol)
    if (!key || seen.has(key)) continue
    seen.add(key)
    if (!agg[key]) agg[key] = { score: 0, papers: [], names: {}, categories: {} }
    agg[key].score += w
    agg[key].papers.push({ id: rec.id, role: pr.role })
    agg[key].names[pr.name] = (agg[key].names[pr.name] || 0) + 1
    agg[key].categories[pr.category] = (agg[key].categories[pr.category] || 0) + 1
  }
}
const ranking = Object.entries(agg)
  .map(([key, v]) => ({
    property: key,
    influence_score: Math.round(v.score * 100) / 100,
    n_papers: v.papers.length,
    category: Object.entries(v.categories).sort((a, b) => b[1] - a[1])[0]?.[0] || 'other',
    top_names: Object.entries(v.names).sort((a, b) => b[1] - a[1]).slice(0, 4).map(x => x[0]),
    evidence: v.papers.sort((a, b) => (citeById[b.id] || 0) - (citeById[a.id] || 0)).slice(0, 8),
  }))
  .sort((a, b) => b.influence_score - a.influence_score)
log(`ranked ${ranking.length} canonical properties; top: ${ranking.slice(0, 8).map(r => r.property).join(', ')}`)

// ---- GapAnalysis: classify vs our derived columns + leakage + derivability -----------------
const GAP_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    classified: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          property: { type: 'string' },
          already_derived: { type: 'boolean', description: 'present in strata_derived columns' },
          leaky_for_spt: { type: 'boolean', description: 'a function of measured SPT-N (LEAKY family)' },
          derivable_from_inputs: { type: 'boolean', description: 'computable from depth + USCS + groundwater + grain-size/Atterberg defaults + geology, WITHOUT SPT-N' },
          formula: { type: 'string', description: 'a concrete correlation/equation to derive it from our inputs, or "" if not derivable' },
          source_hint: { type: 'string', description: 'author/year of the correlation if known' },
          notes: { type: 'string' },
        },
        required: ['property','already_derived','leaky_for_spt','derivable_from_inputs','formula','notes'],
      },
    },
    shortlist: {
      type: 'array', description: 'non-leaky, NOT-already-derived, derivable high-influence properties to implement for the Phase-A test, best first',
      items: { type: 'string' },
    },
    summary: { type: 'string' },
  },
  required: ['classified','shortlist','summary'],
}

const gap = await agent(
  `You are a geotechnical modeling expert advising on a feature-engineering experiment.\n\n`
  + `We predict NJ soil-type (Phase-A GNN) and have an OCR'd boring database with, per depth interval: `
  + `top/bottom depth, USCS class, SPT-N, groundwater depth. We already derive these columns in `
  + `'strata_derived': ${DERIVED.join(', ')}.\n`
  + `Of those, NON-LEAKY model inputs (depend only on depth/USCS/groundwater, NOT SPT-N) are: `
  + `${SAFE.join(', ')}. LEAKY columns (functions of measured SPT-N) are: ${LEAKY.join(', ')}.\n\n`
  + `For the soil-type target, geotech properties attach to BORING nodes and inform soil-label nodes `
  + `via the graph, so SPT-derived properties are NOT leaky here — but properties we can compute WITHOUT `
  + `SPT-N (from grain-size/Atterberg/geology/stress-history) are the most valuable NEW signal because `
  + `they are independent of what we already feed.\n\n`
  + `Here are the literature-ranked soil properties (citation-weighted influence, descending):\n`
  + JSON.stringify(ranking.map(r => ({ property: r.property, influence: r.influence_score, n_papers: r.n_papers, names: r.top_names })), null, 1)
  + `\n\nFor EACH ranked property: set already_derived (is it in strata_derived?), leaky_for_spt (is it a `
  + `function of SPT-N?), derivable_from_inputs (can we compute it from depth+USCS+groundwater+standard `
  + `USCS-keyed Atterberg/grain-size defaults+geology, withOUT SPT-N?), and if derivable give a concrete `
  + `formula + source. Then produce a 'shortlist' of the best NON-leaky, NOT-already-derived, derivable, `
  + `high-influence properties worth implementing to test whether they add information to the soil-type GNN.`,
  { label: 'gap-analysis', phase: 'GapAnalysis', schema: GAP_SCHEMA }
)

return { ranking, gap, n_papers_extracted: perPaper.length, n_relevant: perPaper.filter(p => p && p.relevant).length }
