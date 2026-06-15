"""Load and validate config.yaml, resolving all relative paths against the project root."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of the `pipeline/` package directory.
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _ROOT / "config.yaml"

_REQUIRED_TOP_KEYS = ("org", "layers", "crs", "paging", "covariates",
                      "rate_limits", "duckdb", "paths", "graph", "gates")


class Config:
    """Thin, validated wrapper around the parsed config.yaml."""

    def __init__(self, data: dict, root: Path, source: Path):
        self.d = data
        self.root = root
        self.source = source

    # ---- loading / validation ------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        cfg_path = Path(path or os.environ.get("SOILBOT_CONFIG") or _DEFAULT_CONFIG).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"config not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"config root must be a mapping: {cfg_path}")
        missing = [k for k in _REQUIRED_TOP_KEYS if k not in data]
        if missing:
            raise ValueError(f"config missing required keys: {missing}")
        # Root is the directory containing the config file.
        return cls(data, cfg_path.parent, cfg_path)

    # ---- generic access ------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.d[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.d
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    # ---- path resolution -----------------------------------------------------
    def path(self, name: str) -> Path:
        """Resolve a `paths:` entry to an absolute Path under the project root."""
        rel = self.d["paths"][name]
        return (self.root / rel).resolve()

    def abspath(self, rel: str) -> Path:
        return (self.root / rel).resolve()

    @property
    def duckdb_path(self) -> Path:
        return (self.root / self.d["duckdb"]["path"]).resolve()

    @property
    def extension_dir(self) -> Path:
        return (self.root / self.d["duckdb"]["extension_dir"]).resolve()

    # ---- domain helpers ------------------------------------------------------
    @property
    def feature_server(self) -> str:
        return self.d["org"]["feature_server"].rstrip("/")

    def layer(self, key: str) -> dict:
        return self.d["layers"][key]

    def layer_url(self, key: str) -> str:
        return f"{self.feature_server}/{self.d['layers'][key]['index']}"

    @property
    def user_agent(self) -> str:
        return self.d.get("user_agent", "soilbot/0.1")

    def rate(self, group: str) -> dict:
        return self.d["rate_limits"][group]

    def gate(self, name: str) -> bool:
        return bool(self.d["gates"].get(name, False))
