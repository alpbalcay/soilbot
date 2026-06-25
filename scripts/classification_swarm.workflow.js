export const meta = {
  name: 'classification-knowledge-swarm',
  description: 'Mine harvested geotech papers for actionable knowledge to improve our USCS + soil-type classification data; synthesize a prioritized, machine-applicable rule set grounded in the columns we actually have',
  phases: [
    { title: 'Load', detail: 'select classification-relevant papers' },
    { title: 'Extract', detail: 'fan out -> classification rules/criteria/checks per paper' },
    { title: 'Synthesize', detail: 'curate prioritized actions applicable to our strata data' },
  ],
}

// args = { data_context: {...} }  // our classification-data reality
const DC = (args && args.data_context) || {}

// ---- Load: classification-relevant papers (keyword filter + top-cited general texts) -------
const LOAD_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    papers: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { id: { type: 'string' }, cites: { type: 'integer' } },
      required: ['id', 'cites'] } },
  },
  required: ['papers'],
}
const loaded = await agent(
  `Run this exact command from the repo root and return its stdout verbatim as the schema. It selects `
  + `classification-relevant papers (and the most-cited general soil-mechanics texts, which contain `
  + `classification chapters):\n\n`
  + `.venv/bin/python -c "import json,duckdb; c=duckdb.connect('data/soilbot.duckdb',read_only=True); `
  + `rows=c.execute(\\"SELECT openalex_id,cited_by_count FROM lit_papers WHERE `
  + `lower(title||' '||coalesce(concepts,'')||' '||coalesce(abstract,'')) LIKE '%classif%' `
  + `OR lower(title||coalesce(concepts,'')) LIKE '%atterberg%' OR lower(title||coalesce(concepts,'')) LIKE '%plasticity%' `
  + `OR lower(title||coalesce(concepts,'')) LIKE '%grain%' OR lower(title||coalesce(concepts,'')) LIKE '%particle size%' `
  + `OR lower(title||coalesce(concepts,'')) LIKE '%fines%' OR lower(title||coalesce(concepts,'')) LIKE '%consistency%' `
  + `OR lower(title||coalesce(concepts,'')) LIKE '%soil behaviou%' OR lower(title||coalesce(concepts,'')) LIKE '%index propert%' `
  + `OR lower(title) LIKE '%soil mechanics%' OR lower(title) LIKE '%soil behavior%' `
  + `ORDER BY cited_by_count DESC\\").fetchall(); `
  + `print(json.dumps({'papers':[{'id':r[0],'cites':r[1]} for r in rows]}))"`,
  { label: 'load-papers', phase: 'Load', schema: LOAD_SCHEMA }
)
const PAPERS = (loaded && loaded.papers) || []
if (!PAPERS.length) { return { error: 'no classification-relevant papers loaded' } }
log(`loaded ${PAPERS.length} classification-relevant papers`)

const EXTRACT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    papers: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        id: { type: 'string' },
        relevant: { type: 'boolean', description: 'true if it contains knowledge usable to classify/validate/correct soil classification' },
        rules: { type: 'array', items: {
          type: 'object', additionalProperties: false,
          properties: {
            statement: { type: 'string', description: 'the actionable rule, criterion, or check, stated concretely' },
            kind: { type: 'string', enum: ['decision_criterion','validation_check','correction_heuristic','crosswalk','sanity_range','taxonomy'] },
            applies_to: { type: 'string', enum: ['uscs','atterberg','fines_content','particle_size','engineering_code','spt','drainage','soil_behavior_type','general'] },
            threshold_or_formula: { type: 'string', description: 'concrete numbers/equation if any, else ""' },
            machine_applicable: { type: 'boolean', description: 'codeable against per-interval USCS + depth + SPT-N + groundwater (NO lab Atterberg/grain-size available)' },
          },
          required: ['statement','kind','applies_to','threshold_or_formula','machine_applicable'],
        } },
      },
      required: ['id','relevant','rules'],
    } },
  },
  required: ['papers'],
}

// ---- Extract: fan out over batches ---------------------------------------------------------
const BATCH = 4
const batches = []
for (let i = 0; i < PAPERS.length; i += BATCH) batches.push(PAPERS.slice(i, i + BATCH))
log(`extracting classification knowledge from ${PAPERS.length} papers in ${batches.length} batches`)

const extractResults = await parallel(batches.map((batch, bi) => () =>
  agent(
    `You are a geotechnical-engineering expert mining papers for knowledge to IMPROVE SOIL `
    + `CLASSIFICATION DATA. For EACH paper below, read litreview/metadata/<id>.json (title, abstract, `
    + `concepts) and litreview/fulltext/<id>.txt if it exists (use Read; skip if missing).\n\n`
    + `Extract concrete, actionable classification knowledge: USCS/ASTM-D2487 decision criteria `
    + `(e.g. A-line PI=0.73(LL-20) separating CL/ML and CH/MH; fines thresholds 5%/12%/50%; dual `
    + `symbols), Atterberg/plasticity-chart rules, validation checks (consistency between class, `
    + `plasticity, fines, SPT), correction heuristics for mislabeled classes, crosswalks (engineering/`
    + `AASHTO soil codes <-> USCS), and per-class sanity ranges (typical SPT-N, unit weight, fines by `
    + `USCS class). For each rule set machine_applicable=true ONLY if it can be coded against our data, `
    + `which per interval has: USCS class, top/bottom depth, SPT-N, groundwater depth, sample type — `
    + `NO measured Atterberg limits or grain-size. Set relevant=false (empty rules) for papers with no `
    + `classification content. Be concrete and quote thresholds; do not invent.\n\n`
    + `Paper ids:\n` + batch.map(p => `- ${p.id}`).join('\n'),
    { label: `extract:b${bi}`, phase: 'Extract', schema: EXTRACT_SCHEMA }
  ).then(r => (r && r.papers) || [])
))
const perPaper = extractResults.filter(Boolean).flat()
const citeById = Object.fromEntries(PAPERS.map(p => [p.id, p.cites || 0]))
const allRules = []
for (const rec of perPaper) {
  if (!rec || !rec.relevant) continue
  for (const r of (rec.rules || [])) allRules.push({ ...r, paper: rec.id, cites: citeById[rec.id] || 0 })
}
log(`extracted ${allRules.length} candidate rules from ${perPaper.filter(p => p && p.relevant).length} relevant papers`)

// ---- Synthesize: curate prioritized, applicable actions ------------------------------------
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    actions: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        title: { type: 'string' },
        kind: { type: 'string', enum: ['validation_check','correction_heuristic','crosswalk','sanity_range','distribution_audit','decision_criterion'] },
        rationale: { type: 'string', description: 'what classification problem it fixes, citing the principle' },
        applies_to: { type: 'string' },
        our_data_applicable: { type: 'boolean' },
        method: { type: 'string', description: 'concrete SQL/pseudocode/procedure against strata (uscs_class, top/bottom_depth, spt_n, gw_depth) or soil_labels' },
        expected_impact: { type: 'string' },
        priority: { type: 'string', enum: ['high','medium','low'] },
        source_papers: { type: 'array', items: { type: 'string' } },
      },
      required: ['title','kind','rationale','applies_to','our_data_applicable','method','expected_impact','priority','source_papers'],
    } },
    distribution_assessment: { type: 'string', description: 'assess our USCS distribution vs what the classification literature predicts; explain the missing CH/MH/SW/GW/OH and ML dominance' },
    summary: { type: 'string' },
  },
  required: ['actions','distribution_assessment','summary'],
}

const synth = await agent(
  `You are a geotechnical data-quality expert. We have ${allRules.length} classification rules `
  + `extracted from foundational papers (below). Curate them into a PRIORITIZED, DE-DUPLICATED set of `
  + `concrete ACTIONS to improve OUR soil-classification data.\n\n`
  + `OUR DATA REALITY:\n`
  + `- strata table per depth interval: uscs_class, top_depth, bottom_depth, spt_n, gw_depth, sample_type. `
  + `NO measured Atterberg limits or grain-size distribution (NJDOT logs rarely report them).\n`
  + `- soil_labels.primary_label = NJDOT ENGINEERING soil codes (not USCS); drainage class present.\n`
  + `- Per-boring geology (surficial/bedrock) + SSURGO component/drainage covariates available.\n`
  + `- OUR USCS DISTRIBUTION (counts): ${JSON.stringify(DC.uscs_dist || {})}. Note the anomaly: `
  + `only 1 'CH', and ZERO MH/SW/GW/OH, with 'ML' dominant — likely an OCR/parser-vocabulary artifact, `
  + `not real NJ geology (which has coastal-plain high-plasticity clays).\n\n`
  + `RULES (statement | kind | applies_to | threshold | machine_applicable | paper | cites):\n`
  + allRules.slice(0, 120).map(r => `- ${r.statement} | ${r.kind} | ${r.applies_to} | ${r.threshold_or_formula} | ${r.machine_applicable} | ${r.paper} | ${r.cites}`).join('\n')
  + `\n\nProduce: (1) 'actions' — each a concrete, prioritized improvement with a method coded against `
  + `OUR columns (mark our_data_applicable=false honestly where it needs Atterberg/grain-size we lack, `
  + `but still record it as a recommendation for future OCR). Favor: USCS token validation, the missing `
  + `CH/MH/SW/GW/OH distribution audit + likely OCR confusions (CH->CL, MH->ML, SW->SP, GW->GP), `
  + `SPT-N-vs-USCS sanity ranges, engineering-code<->USCS crosswalk, depth/groundwater consistency. `
  + `(2) 'distribution_assessment' explaining our USCS distribution vs literature expectation. `
  + `Cite source_papers by id.`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return {
  n_papers: PAPERS.length,
  n_relevant: perPaper.filter(p => p && p.relevant).length,
  n_rules: allRules.length,
  rules: allRules,
  synthesis: synth,
}
