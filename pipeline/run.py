"""soilbot pipeline CLI.

    python -m pipeline.run --phase 1        # discovery & schema audit -> schema_audit.md
    python -m pipeline.run --phase 2        # bulk extract layers 0/1/2 -> DuckDB + GeoPackage
    python -m pipeline.run --phase 3 --download-logs [--ocr]   # GATED scanned-log crawl / OCR
    python -m pipeline.run --phase 4 [--dem]                   # covariates (geology, SSURGO[, DEM])
    python -m pipeline.run --phase 5        # node_features + edges.parquet + REPORT.md
    python -m pipeline.run --phase 6        # geotechnical soil-equation engine -> strata_derived
    python -m pipeline.run --phase all      # default scope: 1, 2, 4, 5 (heavy steps stay gated)
"""
from __future__ import annotations

import argparse

from . import covariates, db, derive, extract, graph, parse_logs, report, schema_audit
from .arcgis import ArcGISClient
from .config import Config
from .logging_setup import new_run_id, setup


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.run",
        description="Extract NJDOT GDMS geotechnical boring data into a GNN-ready store.")
    p.add_argument("--phase", required=True, choices=["1", "2", "3", "4", "5", "6", "7", "all"],
                   help="pipeline phase to run (or 'all' for 1,2,4,5; 7=literature review)")
    p.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    p.add_argument("--use-rust", action="store_true",
                   help="force the soilbot_rs path for graph/feature build (overrides config flags)")
    p.add_argument("--no-gpkg", action="store_true", help="phase 2: skip GeoPackage export")
    p.add_argument("--download-logs", action="store_true",
                   help="phase 3: crawl scanned-log PDFs (heavy, ~3-4 GB)")
    p.add_argument("--ocr", action="store_true", help="phase 3: OCR-parse downloaded logs into strata")
    p.add_argument("--dem", action="store_true",
                   help="phase 4: per-boring 3DEP elevation (~49k calls)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap downloads / parses / DEM samples this run (for batching/testing)")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    config = Config.load(args.config)
    if args.use_rust:  # CLI override: flip the soilbot_rs flags on for this run
        config.d.setdefault("graph", {})["use_rust"] = True
        config.d.setdefault("ml", {})["use_rust"] = True
    run_id = new_run_id()

    def logger(filename: str, phase: str):
        return setup(config.path("log_dir"), filename, run_id, phase, console=True)

    def phase1():
        schema_audit.run(config, logger("extract.log", "phase1"))

    def phase2():
        extract.run(config, logger("extract.log", "phase2"), export_gpkg=not args.no_gpkg)

    def phase3():
        ran = False
        if args.download_logs:
            log = logger("download.log", "phase3")
            con = db.connect(config)
            db.bootstrap(con)
            attachments_client = ArcGISClient(config, log, rate_group="attachments")
            from . import attachments
            attachments.download_logs(con, attachments_client, config, log, max_downloads=args.limit)
            con.close()
            ran = True
        if args.ocr or not ran:
            parse_logs.run(config, logger("parse.log", "phase3"), ocr=args.ocr, limit=args.limit)

    def phase4():
        covariates.run(config, logger("extract.log", "phase4"), dem=args.dem, dem_limit=args.limit)

    def phase5():
        graph.run(config, logger("extract.log", "phase5"))
        report.run(config, logger("extract.log", "report"))

    def phase6():
        derive.run(config, logger("parse.log", "phase6"))

    def phase7():
        from . import litreview
        litreview.run(config, logger("litreview.log", "phase7"), limit=args.limit)

    plan = {"1": [phase1], "2": [phase2], "3": [phase3], "4": [phase4], "5": [phase5],
            "6": [phase6], "7": [phase7], "all": [phase1, phase2, phase4, phase5]}
    for fn in plan[args.phase]:
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
