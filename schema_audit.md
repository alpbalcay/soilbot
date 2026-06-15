# schema_audit.md — NJDOT GDMS layer audit

_Generated 2026-06-11T00:14:22+00:00 (run `r-6c237db9`) by `pipeline.schema_audit`. All endpoints anonymous._

Feature server: `https://services.arcgis.com/HggmsDF7UJsNN1FK/arcgis/rest/services/Soil_Borings_Map/FeatureServer`

## Services discovered
_Service listing unavailable (queried layers directly)._

## Layer `borings` — GDMS BORING LOG

- URL: `https://services.arcgis.com/HggmsDF7UJsNN1FK/arcgis/rest/services/Soil_Borings_Map/FeatureServer/0`
- Geometry: `esriGeometryPoint` · SR (WKID): `3424` · maxRecordCount: `2000`
- Feature count: **49,152** (expected 49,152) ✅
- Attachments: `True` · supportsQueryAttachments: `True` · OID field: `OBJECTID`
- Spatial fields: OBJECTID, POINT_X, POINT_Y
- Document-link fields: **FILENAME**

| Field | Type | Alias | Class |
|---|---|---|---|
| OBJECTID | esriFieldTypeOID | OBJECTID | spatial |
| LID | esriFieldTypeString | LID | attribute |
| LABEL | esriFieldTypeString | LABEL | attribute |
| FILENAME | esriFieldTypeString | FILENAME | doclink |
| PID | esriFieldTypeString | PID | attribute |
| BCONTR | esriFieldTypeString | BCONTR | attribute |
| SRCE | esriFieldTypeSmallInteger | SRCE | attribute |
| LOCALN | esriFieldTypeString | LOCALN | attribute |
| OVERSIZE | esriFieldTypeString | OVERSIZE | attribute |
| POINT_X | esriFieldTypeDouble | POINT_X | spatial |
| POINT_Y | esriFieldTypeDouble | POINT_Y | spatial |
| CREATEDATE | esriFieldTypeDate | CREATEDATE | attribute |
| CREATEDBY | esriFieldTypeInteger | CREATEDBY | attribute |
| CHANGEDATE | esriFieldTypeDate | CHANGEDATE | attribute |
| CHANGEDBY | esriFieldTypeInteger | CHANGEDBY | attribute |
| MATCHID | esriFieldTypeInteger | MATCHID | attribute |

## Layer `boring_plan` — GDMS BORING PLAN

- URL: `https://services.arcgis.com/HggmsDF7UJsNN1FK/arcgis/rest/services/Soil_Borings_Map/FeatureServer/1`
- Geometry: `esriGeometryPolygon` · SR (WKID): `3424` · maxRecordCount: `2000`
- Feature count: **3,829** (expected 3,829) ✅
- Attachments: `True` · supportsQueryAttachments: `True` · OID field: `OBJECTID`
- Spatial fields: OBJECTID, Shape__Area, Shape__Length
- Document-link fields: **FILENAME**

| Field | Type | Alias | Class |
|---|---|---|---|
| OBJECTID | esriFieldTypeOID | OBJECTID | spatial |
| PID | esriFieldTypeString | PID | attribute |
| FILENAME | esriFieldTypeString | FILENAME | doclink |
| ROUTE | esriFieldTypeString | ROUTE | attribute |
| SECT | esriFieldTypeString | SECT | attribute |
| PCONTR | esriFieldTypeString | PCONTR | attribute |
| PDATE | esriFieldTypeDate | PDATE | attribute |
| UPC | esriFieldTypeString | UPC | attribute |
| CREATEDATE | esriFieldTypeDate | CREATEDATE | attribute |
| CREATEDBY | esriFieldTypeInteger | CREATEDBY | attribute |
| CHANGEDATE | esriFieldTypeDate | CHANGEDATE | attribute |
| CHANGEDBY | esriFieldTypeInteger | CHANGEDBY | attribute |
| MATCHID | esriFieldTypeInteger | MATCHID | attribute |
| Shape__Area | esriFieldTypeDouble | Shape__Area | spatial |
| Shape__Length | esriFieldTypeDouble | Shape__Length | spatial |

## Layer `soil_label` — GIS.Geol_soil_egr_label

- URL: `https://services.arcgis.com/HggmsDF7UJsNN1FK/arcgis/rest/services/Soil_Borings_Map/FeatureServer/2`
- Geometry: `esriGeometryPoint` · SR (WKID): `3424` · maxRecordCount: `2000`
- Feature count: **20,255** (expected 20,255) ✅
- Attachments: `False` · supportsQueryAttachments: `False` · OID field: `OBJECTID`
- Spatial fields: OBJECTID
- Document-link fields: **PDF_PG_NUM, URL, WEBURL**

| Field | Type | Alias | Class |
|---|---|---|---|
| OBJECTID | esriFieldTypeOID | OBJECTID | spatial |
| LABEL_TYPE | esriFieldTypeString | LABEL_TYPE | attribute |
| PRIMARY_LABEL | esriFieldTypeString | PRIMARY_LABEL | attribute |
| SECONDARY_LABEL | esriFieldTypeString | SECONDARY_LABEL | attribute |
| DRAINAGE | esriFieldTypeString | DRAINAGE | attribute |
| PDF_PG_NUM | esriFieldTypeString | PDF_PG_NUM | doclink |
| URL | esriFieldTypeString | URL | doclink |
| WEBURL | esriFieldTypeString | WebURL | doclink |

## Stratigraphy verdict (structured vs OCR-only)

**No structured stratigraphy fields exist in ANY layer.** SPT N-values, USCS class, depth intervals, groundwater depth, sample type and elevation live ONLY inside the scanned PDF boring logs (Layer-0 attachments). OCR is the sole path to them, so the `strata` table starts EMPTY and is populated only when `--ocr` runs. The only structured soil signal without OCR is `PRIMARY_LABEL`/`SECONDARY_LABEL`/`DRAINAGE` on the soil-label layer (a coarse engineering-soil class, not full stratigraphy).

## Document access

- Scanned boring logs = ArcGIS **attachments** on Layer 0: `<layer0>/<OBJECTID>/attachments/<ATTACHMENT_ID>` (application/pdf, anonymous).
- Secondary: county roll-up PDFs referenced by the soil-label layer `WEBURL` (`http://wading01.oit.state.nj.us/gdms/PDFs/<county>.pdf#page=N`) — plain HTTP, opt-in only.

