# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml==6.0.3"]
# ///
"""Convert local OpenAPI 3.0 and 3.1 path operations to a ToolRank retrieval catalog."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

import yaml

METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
VERSION = re.compile(r"3\.[01]\.\d+(?:[-+][A-Za-z0-9.+-]+)?")


def object_at(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def path_item(document, item, seen=None):
    item = object_at(item, "path item")
    if "$ref" not in item:
        return item
    ref = item["$ref"]
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise ValueError("path item references must be local JSON Pointers")
    seen = set() if seen is None else seen
    if ref in seen:
        raise ValueError(f"cyclic path item reference {ref}")
    seen.add(ref)
    target = document
    for field in unquote(ref[2:]).split("/"):
        field = field.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or field not in target:
            raise ValueError(f"unresolved path item reference {ref}")
        target = target[field]
    fields = {name: value for name, value in item.items() if name != "$ref"}
    inherited = path_item(document, target, seen)
    if set(fields) & set(inherited):
        raise ValueError(f"conflicting fields at path item reference {ref}")
    return {**inherited, **fields}


def name_for(namespace, operation_id, method, route):
    route_name = re.sub(r"\{([^}]+)\}", r"by_\1", route.strip("/"))
    suffix = operation_id or f"{method}_{route_name}"
    name = f"{namespace}__{UNSAFE.sub('_', suffix).strip('_') or method}"
    if len(name) > 128:
        name = f"{name[:115]}_{sha256(name.encode()).hexdigest()[:12]}"
    return name


def catalog_from_openapi(document, namespace):
    """Return namespaced search entries with OpenAPI operation identities."""
    object_at(document, "OpenAPI document")
    version = document.get("openapi")
    if not isinstance(version, str) or not VERSION.fullmatch(version):
        raise ValueError("expected OpenAPI 3.0.x or 3.1.x")
    if not NAME.fullmatch(namespace):
        raise ValueError("namespace must start with a letter or underscore and use ASCII letters, digits, _, . or -")
    paths = object_at(document.get("paths", {}), "paths")
    entries = {}
    for route in sorted(key for key in paths if isinstance(key, str) and key.startswith("/")):
        item = path_item(document, paths[route])
        for method in METHODS:
            if method not in item:
                continue
            label = f"{method.upper()} {route}"
            operation = object_at(item[method], label)
            operation_id = operation.get("operationId")
            if "operationId" in operation and (not isinstance(operation_id, str) or not operation_id.strip()):
                raise ValueError(f"{label} has an invalid operationId")
            name = name_for(namespace, operation_id, method, route)
            if name in entries:
                raise ValueError(f"{label} produces duplicate catalog name {name!r}")
            text = []
            for key in ("summary", "description"):
                value = operation.get(key, "")
                if not isinstance(value, str):
                    raise ValueError(f"{label} has a non-text {key}")
                if value.strip() and value.strip() not in text:
                    text.append(value.strip())
            entries[name] = {
                "name": name,
                "description": "\n".join([*text, label]),
                "_meta": {"toolrank/openapi": {
                    "namespace": namespace, "operationId": operation_id,
                    "method": method.upper(), "path": route,
                }},
            }
    if not entries:
        raise ValueError("specification has no operations under paths")
    return [entries[name] for name in sorted(entries)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", nargs=2, action="append", metavar=("NAMESPACE", "SPEC"), required=True)
    args = parser.parse_args()
    entries = []
    namespaces = set()
    names = set()
    for namespace, filename in args.api:
        path = Path(filename)
        try:
            if namespace in namespaces:
                raise ValueError(f"duplicate namespace {namespace!r}")
            namespaces.add(namespace)
            text = path.read_text(encoding="utf-8")
            spec = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
            catalog = catalog_from_openapi(spec, namespace)
            new_names = {entry["name"] for entry in catalog}
            if duplicates := names & new_names:
                raise ValueError(f"duplicate catalog names: {', '.join(sorted(duplicates))}")
            names.update(new_names)
            entries.extend(catalog)
        except (OSError, ValueError, yaml.YAMLError, RecursionError) as error:
            parser.error(f"{path}: {error}")
    json.dump({"tools": entries}, sys.stdout, ensure_ascii=True, indent=2)
    print()


if __name__ == "__main__":
    main()
