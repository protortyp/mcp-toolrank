"""Convert a portable client config into a gateway setup file."""
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory


ENV_REFERENCE = re.compile(r"\$\{(?:env:)?([A-Za-z_][A-Za-z0-9_]*)\}")
BEARER_REFERENCE = re.compile(r"Bearer " + ENV_REFERENCE.pattern, re.IGNORECASE)


class ImportConfigError(ValueError):
    """A safe, value-free message for an unsupported client setting."""


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ImportConfigError("Client config contains a duplicate JSON key.")
        result[key] = value
    return result


def plain_string(value):
    return isinstance(value, str) and "${" not in value and "\x00" not in value


def stdio_entry(entry):
    if entry.keys() - {"command", "args", "cwd", "env"}:
        raise ImportConfigError("Unsupported stdio settings.")
    command, args = entry.get("command"), entry.get("args", [])
    if not plain_string(command) or not command.strip() or not isinstance(args, list) or not all(
        plain_string(arg) for arg in args
    ):
        raise ImportConfigError("Stdio requires a literal command and a list of literal arguments.")
    converted = {"transport": "stdio", "command": command, "args": args}
    if "cwd" in entry:
        cwd = entry["cwd"]
        if not plain_string(cwd) or not Path(cwd).is_absolute():
            raise ImportConfigError("Stdio working directories must be absolute paths.")
        converted["cwd"] = cwd
    if "env" in entry:
        env = entry["env"]
        if not isinstance(env, dict):
            raise ImportConfigError("Stdio env must map variable names to environment references.")
        references = {}
        for key, value in env.items():
            match = ENV_REFERENCE.fullmatch(value) if isinstance(value, str) else None
            if match is None:
                raise ImportConfigError("Stdio env values must be whole environment references.")
            references[key] = match[1]
        converted["env"] = references
    return converted


def http_entry(entry):
    if entry.keys() - {"url", "headers"}:
        raise ImportConfigError("Unsupported Streamable HTTP settings.")
    url = entry.get("url")
    if not plain_string(url) or "?" in url:
        raise ImportConfigError("Use a literal HTTP endpoint without a query string.")
    converted = {"transport": "streamable_http", "url": url}
    headers = entry.get("headers", {})
    if not isinstance(headers, dict):
        raise ImportConfigError("HTTP headers must be a JSON object.")
    if headers:
        if len(headers) != 1:
            raise ImportConfigError("Only an Authorization bearer environment reference is supported.")
        (key, value), = headers.items()
        match = BEARER_REFERENCE.fullmatch(value) if isinstance(value, str) else None
        if key.casefold() != "authorization" or match is None:
            raise ImportConfigError("Only an Authorization bearer environment reference is supported.")
        converted["bearer_token_env"] = match[1]
    return converted


def convert_client_config(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_fields)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ImportConfigError("Could not read the client config as JSON.") from None
    if not isinstance(data, dict) or len(data) != 1:
        raise ImportConfigError("Use a server-only JSON document with mcpServers or servers.")
    (key, servers), = data.items()
    if key not in {"mcpServers", "servers"} or not isinstance(servers, dict):
        raise ImportConfigError("Use a server-only JSON document with mcpServers or servers.")

    converted = []
    for number, (name, raw) in enumerate(servers.items(), 1):
        try:
            if not isinstance(raw, dict):
                raise ImportConfigError("Each server must be a JSON object.")
            entry = dict(raw)
            enabled = entry.pop("enabled", True)
            disabled = entry.pop("disabled", False)
            if type(enabled) is not bool or type(disabled) is not bool:
                raise ImportConfigError("Server enablement flags must be booleans.")
            if disabled or not enabled:
                continue
            if "tools" in entry and entry.pop("tools") != ["*"]:
                raise ImportConfigError("Specific client tool permissions require manual configuration.")
            transport = entry.pop("type", "stdio" if "command" in entry else None)
            if transport == "stdio":
                upstream = stdio_entry(entry)
            elif transport == "http":
                upstream = http_entry(entry)
            else:
                raise ImportConfigError("Select stdio or explicit Streamable HTTP type http.")
        except ImportConfigError as exc:
            raise ImportConfigError(f"Server entry {number}. {exc}") from None
        converted.append({"id": name, **upstream, "discover": [], "invoke": []})
    return {"ranking": "bm25", "servers": converted}


def write_new_config(path, data):
    """Publish a private file only after the JSON is complete, without overwriting."""
    with TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / "gateway.json"
        fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
        os.link(staged, path)
