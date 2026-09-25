"""Config loading and CLI override merging. Nothing in src/ hardcodes a value."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    settings_path: str | None = None,
    categories_path: str | None = None,
    lexicon_path: str | None = None,
) -> dict[str, Any]:
    cfg = _load_yaml(Path(settings_path) if settings_path else ROOT / "config" / "settings.yaml")
    cfg["_categories"] = _load_yaml(
        Path(categories_path) if categories_path else ROOT / "config" / "categories.yaml"
    )
    cfg["_lexicon"] = _load_yaml(
        Path(lexicon_path) if lexicon_path else ROOT / "config" / "lexicon.yaml"
    )
    cfg["_root"] = str(ROOT)
    return cfg


def set_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """set_path(cfg, 'llm.model', 'x') -> cfg['llm']['model'] = 'x'"""
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply {'llm.model': 'x'} style overrides. None values are ignored so
    that an unset CLI flag never clobbers a config default."""
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        if value is None:
            continue
        set_path(out, dotted, value)
    return out


def resolve_api_key(explicit: str | None) -> str | None:
    """Priority: explicit flag > GOOGLE_FACTCHECK_API_KEY env > .apikey file."""
    if explicit:
        return explicit.strip()
    env = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
    if env:
        return env.strip()
    keyfile = ROOT / ".apikey"
    if keyfile.exists():
        text = keyfile.read_text(encoding="utf-8").strip()
        if text:
            return text
    return None


def category_policy(cfg: dict[str, Any], category: str) -> dict[str, Any]:
    """Merge the category's policy over the defaults block."""
    cats = cfg.get("_categories", {})
    policy = dict(cats.get("defaults", {}))
    policy.update(cats.get("categories", {}).get(category, {}))
    # keyword lists are routing inputs, not policy - strip them from the result
    for k in list(policy):
        if k.startswith("keywords_"):
            policy.pop(k)
    policy["category"] = category
    return policy
