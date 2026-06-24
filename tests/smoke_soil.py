"""Smoke + sanity test for the geotechnical soil-equation engine (pipeline.derive + soilbot_rs).

Builds `strata_derived` from the real `strata` table and asserts physical plausibility of the
derived properties. Run:  SOILBOT_DB=<writable copy> .venv/bin/python tests/smoke_soil.py
(the live DB is single-writer locked while a pipeline job runs — test against a copy).

Skips (exit 0) if soilbot_rs is not built or there are no parsed strata yet.
"""
from __future__ import annotations

import os
import sys

try:
    import soilbot_rs  # noqa: F401
except ImportError:
    print("SKIP: soilbot_rs not built")
    sys.exit(0)

import duckdb

from pipeline import db, derive
from pipeline.config import Config


class _Log:
    def info(self, ev, **k):
        print("  info", ev, k)

    def warning(self, ev, **k):
        print("  warn", ev, k)


def main():
    config = Config.load()
    dbpath = os.environ.get("SOILBOT_DB", str(config.duckdb_path))
    # This test WRITES strata_derived, so it must own the single-writer lock. Point SOILBOT_DB at a
    # copy when the live DB may be in use; if the lock is held, skip cleanly rather than crash.
    try:
        con = duckdb.connect(dbpath, read_only=False)
    except duckdb.Error as exc:
        print(f"SKIP: cannot open {dbpath} read-write ({str(exc)[:80]}); "
              "point SOILBOT_DB at a writable copy")
        sys.exit(0)
    con.execute(f"SET extension_directory='{config.extension_dir}'")
    con.execute("LOAD spatial")

    n_strata = con.execute("SELECT COUNT(*) FROM strata").fetchone()[0]
    if n_strata == 0:
        print("SKIP: strata table empty (OCR not run yet)")
        sys.exit(0)
    print(f"strata rows: {n_strata}")

    db.bootstrap(con)
    # force-enable the engine for the test regardless of config gate
    config.d.setdefault("soil_engine", {})["enabled"] = True
    stats = derive.build_derived(con, config, _Log())

    ok = True
    if stats["rows"] <= 0:
        print("FAIL: no derived rows"); ok = False
    if stats["with_stress"] <= 0:
        print("FAIL: no effective-stress rows"); ok = False
    # positive-coverage: with SPT present, the engine must actually derive strength properties
    # (guards against a silent all-NULL regression that the bounds checks below would pass).
    if stats["with_spt"] > 0 and (stats["with_phi"] + stats["with_su"]) <= 0:
        print("FAIL: SPT present but no phi/Su derived (engine emitted all-NULL)"); ok = False

    # physical-sanity battery over the populated columns
    # σ'v0 must be > 0 and must not exceed total stress (above the shallow 0.01 tsf guard floor).
    bad_stress = con.execute("""
        SELECT COUNT(*) FROM strata_derived
        WHERE sigma_eff_v0_tsf IS NOT NULL
          AND (sigma_eff_v0_tsf <= 0
               OR (sigma_v0_tsf > 0.01 AND sigma_eff_v0_tsf > sigma_v0_tsf + 1e-6))
    """).fetchone()[0]
    # σ'v0 must be non-decreasing with depth within a boring (monotone effective-stress profile)
    bad_mono = con.execute("""
        WITH p AS (
            SELECT boring_id, depth_ft, sigma_eff_v0_tsf,
                   LAG(sigma_eff_v0_tsf) OVER (PARTITION BY boring_id ORDER BY depth_ft) AS prev
            FROM strata_derived WHERE sigma_eff_v0_tsf IS NOT NULL
        ) SELECT COUNT(*) FROM p WHERE prev IS NOT NULL AND sigma_eff_v0_tsf < prev - 1e-6
    """).fetchone()[0]
    bad_dr = con.execute(
        "SELECT COUNT(*) FROM strata_derived WHERE dr_pct < 0 OR dr_pct > 100").fetchone()[0]
    bad_phi = con.execute(
        "SELECT COUNT(*) FROM strata_derived WHERE phi_peck_deg < 20 OR phi_peck_deg > 45"
    ).fetchone()[0]
    # Su = f1·N60 is ≥ 0; the engine emits NULL (not a misleading 0) at spt_n = 0 where the Stroud
    # correlation degenerates, so any non-NULL Su should be > 0. Negative Su would signal a defect.
    bad_su = con.execute(
        "SELECT COUNT(*) FROM strata_derived WHERE su_tsf IS NOT NULL AND su_tsf <= 0").fetchone()[0]
    for name, bad in [("eff>total or eff<=0", bad_stress), ("σ'v0 non-monotone", bad_mono),
                      ("Dr out of [0,100]", bad_dr),
                      ("phi out of [20,45]", bad_phi), ("Su<=0", bad_su)]:
        status = "ok  " if bad == 0 else "FAIL"
        if bad:
            ok = False
        print(f"  {status} sanity[{name}] violations={bad}")

    # show a sample
    sample = con.execute("""
        SELECT boring_id, depth_ft, sigma_eff_v0_tsf, n1_60, phi_peck_deg, dr_pct, su_tsf
        FROM strata_derived WHERE n1_60 IS NOT NULL ORDER BY boring_id, interval_index LIMIT 5
    """).fetchall()
    print("  sample (boring, depth, σ'v0, (N1)60, φ, Dr, Su):")
    for r in sample:
        print("   ", tuple(round(x, 3) if isinstance(x, float) else x for x in r))
    con.close()

    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
