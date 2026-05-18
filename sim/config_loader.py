"""Small config loader with a PyYAML fallback for simple project YAML files."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml:
        return yaml.safe_load(text) or {}
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current_key: str | None = None
    current_item: dict[str, Any] | None = None
    nested_key: str | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0 and line.endswith(":"):
            current_key = line[:-1]
            root[current_key] = None
            current_item = None
            nested_key = None
            continue

        if indent == 0 and ":" in line:
            key, value = line.split(":", 1)
            root[key] = _parse_scalar(value.strip())
            current_key = key
            current_item = None
            nested_key = None
            continue

        if current_key is None:
            continue

        if line.startswith("- "):
            if root.get(current_key) is None:
                root[current_key] = []
            value = line[2:].strip()
            if ":" in value and not value.startswith("["):
                key, raw_value = value.split(":", 1)
                current_item = {key: _parse_scalar(raw_value.strip())}
                root[current_key].append(current_item)
            else:
                root[current_key].append(_parse_scalar(value))
            nested_key = None
            continue

        if root.get(current_key) is None:
            root[current_key] = {}

        if isinstance(root.get(current_key), list) and current_item is not None and ":" in line:
            key, value = line.split(":", 1)
            current_item[key] = _parse_scalar(value.strip())
            continue

        if isinstance(root.get(current_key), list):
            continue

        if isinstance(root.get(current_key), dict) and ":" in line:
            key, value = line.split(":", 1)
            root[current_key][key] = _parse_scalar(value.strip())
            continue

        if line.endswith(":"):
            nested_key = line[:-1]
            root[current_key] = {}
            continue

        if nested_key and ":" in line:
            root[current_key][nested_key] = root[current_key].get(nested_key, {})

    return root


def _parse_scalar(value: str) -> Any:
    if value == "":
        return {}
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")
