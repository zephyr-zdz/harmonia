"""Configuration loading: packaged defaults deep-merged with optional user overrides."""

from __future__ import annotations

import copy
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any


def default_config() -> dict[str, Any]:
    with resources.files("harmonia").joinpath("default_config.toml").open("rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = default_config()
    if path is not None:
        with open(path, "rb") as f:
            cfg = _deep_merge(cfg, tomllib.load(f))
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
