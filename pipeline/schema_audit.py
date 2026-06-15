"""Phase 1 — discovery & schema audit.

Re-verifies each target layer live (`?f=json`), classifies every field as spatial /
attribute / document-link, checks counts against the audited expectations in config, and
writes schema_audit.md. The stratigraphy verdict is DATA-DRIVEN: it scans all fields for
SPT/USCS/depth/groundwater columns and concludes OCR-only when none exist.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .arcgis import ArcGISClient
from .config import Config
from .util import atomic_write_text

# Field-name/alias tokens that would indicate STRUCTURED stratigraphy if present.
_STRATA_TOKENS = ("spt", "nval", "n_value", "blow", "uscs", "aashto", "strat",
                   "depth", "groundwater", "gwt", "gw_", "sample", "elev", "horizon")
_DOCLINK_TOKENS = ("url", "weburl", "link", "doc", "pdf", "hyperlink", "image", "scan", "filename")
_SPATIAL_TOKENS = ("point_x", "point_y", "shape__", "shape_", "longitude", "latitude")


def classify_field(name: str, ftype: str) -> str:
    n = name.lower()
    if ftype in ("esriFieldTypeOID", "esriFieldTypeGeometry"):
        return "spatial"
    if any(t in n for t in _SPATIAL_TOKENS):
        return "spatial"
    if any(t in n for t in _DOCLINK_TOKENS):
        return "doclink"
    return "attribute"


def _audit_layer(client: ArcGISClient, layer_url: str, expected: int | None) -> dict:
    meta = client.layer_metadata(layer_url)
    count = client.feature_count(layer_url)
    fields = []
    has_strata_field = False
    for f in meta.get("fields", []) or []:
        name, ftype, alias = f.get("name"), f.get("type"), f.get("alias")
        cls = classify_field(name, ftype)
        fields.append({"name": name, "type": ftype, "alias": alias, "class": cls})
        if any(t in (name or "").lower() for t in _STRATA_TOKENS):
            # POINT_X/Y contain 'point' not strata; exclude obvious coordinate fields.
            if not any(s in (name or "").lower() for s in ("point_x", "point_y", "shape")):
                has_strata_field = True
    sr = (meta.get("spatialReference", {}) or {})
    return {
        "url": layer_url,
        "name": meta.get("name"),
        "geometry_type": meta.get("geometryType"),
        "wkid": sr.get("latestWkid") or sr.get("wkid"),
        "max_record_count": meta.get("maxRecordCount"),
        "has_attachments": bool(meta.get("hasAttachments")),
        "supports_query_attachments": ArcGISClient.supports_query_attachments(meta),
        "object_id_field": meta.get("objectIdField"),
        "count": count,
        "expected": expected,
        "count_matches": (expected is None or count == expected),
        "fields": fields,
        "has_structured_strata": has_strata_field,
    }


def audit(config: Config, client: ArcGISClient, log) -> dict:
    log.info("phase1_start")
    services = []
    try:
        svc = client.list_services()
        services = [{"name": s.get("name"), "type": s.get("type")}
                    for s in svc.get("services", []) or []]
    except Exception as exc:  # noqa: BLE001 - audit must not abort on service listing
        log.warning("service_list_failed", error=str(exc))

    layers = {}
    for key in ("borings", "boring_plan", "soil_label"):
        url = config.layer_url(key)
        expected = config.layer(key).get("expected_count")
        layers[key] = _audit_layer(client, url, expected)
        a = layers[key]
        log.info("layer_audited", key=key, count=a["count"], expected=expected,
                 matches=a["count_matches"], fields=len(a["fields"]))
    any_strata = any(l["has_structured_strata"] for l in layers.values())
    return {"services": services, "layers": layers, "any_structured_strata": any_strata}


def _md_field_table(fields: list[dict]) -> str:
    lines = ["| Field | Type | Alias | Class |", "|---|---|---|---|"]
    for f in fields:
        lines.append(f"| {f['name']} | {f['type']} | {f.get('alias') or ''} | {f['class']} |")
    return "\n".join(lines)


def write_markdown(config: Config, data: dict, run_id: str) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = [
        "# schema_audit.md — NJDOT GDMS layer audit",
        "",
        f"_Generated {ts} (run `{run_id}`) by `pipeline.schema_audit`. All endpoints anonymous._",
        "",
        f"Feature server: `{config.feature_server}`",
        "",
        "## Services discovered",
    ]
    if data["services"]:
        out.append("| Service | Type |")
        out.append("|---|---|")
        for s in data["services"]:
            out.append(f"| {s['name']} | {s['type']} |")
    else:
        out.append("_Service listing unavailable (queried layers directly)._")
    out.append("")

    for key, a in data["layers"].items():
        doclinks = [f["name"] for f in a["fields"] if f["class"] == "doclink"]
        spatial = [f["name"] for f in a["fields"] if f["class"] == "spatial"]
        flag = "✅" if a["count_matches"] else "⚠️ DRIFT"
        out += [
            f"## Layer `{key}` — {a['name']}",
            "",
            f"- URL: `{a['url']}`",
            f"- Geometry: `{a['geometry_type']}` · SR (WKID): `{a['wkid']}` · maxRecordCount: `{a['max_record_count']}`",
            f"- Feature count: **{a['count']:,}** (expected {a['expected']:,}) {flag}"
            if a["expected"] else f"- Feature count: **{a['count']:,}**",
            f"- Attachments: `{a['has_attachments']}` · supportsQueryAttachments: `{a['supports_query_attachments']}` · OID field: `{a['object_id_field']}`",
            f"- Spatial fields: {', '.join(spatial) or '—'}",
            f"- Document-link fields: **{', '.join(doclinks) or '—'}**",
            "",
            _md_field_table(a["fields"]),
            "",
        ]

    verdict = (
        "**STRUCTURED stratigraphy fields were found** — review the field tables above."
        if data["any_structured_strata"] else
        "**No structured stratigraphy fields exist in ANY layer.** SPT N-values, USCS class, "
        "depth intervals, groundwater depth, sample type and elevation live ONLY inside the "
        "scanned PDF boring logs (Layer-0 attachments). OCR is the sole path to them, so the "
        "`strata` table starts EMPTY and is populated only when `--ocr` runs. The only structured "
        "soil signal without OCR is `PRIMARY_LABEL`/`SECONDARY_LABEL`/`DRAINAGE` on the soil-label "
        "layer (a coarse engineering-soil class, not full stratigraphy)."
    )
    out += [
        "## Stratigraphy verdict (structured vs OCR-only)",
        "",
        verdict,
        "",
        "## Document access",
        "",
        "- Scanned boring logs = ArcGIS **attachments** on Layer 0: "
        "`<layer0>/<OBJECTID>/attachments/<ATTACHMENT_ID>` (application/pdf, anonymous).",
        "- Secondary: county roll-up PDFs referenced by the soil-label layer `WEBURL` "
        "(`http://wading01.oit.state.nj.us/gdms/PDFs/<county>.pdf#page=N`) — plain HTTP, opt-in only.",
        "",
    ]
    text = "\n".join(out) + "\n"
    path = config.path("schema_audit")
    atomic_write_text(path, text)
    return str(path)


def run(config: Config, log) -> dict:
    client = ArcGISClient(config, log)
    data = audit(config, client, log)
    path = write_markdown(config, data, log.run_id)
    log.info("phase1_done", schema_audit=path,
             structured_strata=data["any_structured_strata"])
    return data
