"""soilbot — NJDOT GDMS geotechnical boring extraction pipeline.

Extracts publicly available NJDOT Geotechnical Data Management System (GDMS) soil
boring data from ArcGIS REST endpoints into a DuckDB + GeoPackage store, structured
to feed a downstream Bayesian Graph Neural Network. See REPORT.md / schema_audit.md.
"""
__version__ = "0.1.0"
