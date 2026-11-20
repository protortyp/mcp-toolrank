"""Local, single-client MCP gateway with independent upstream allowlists."""

import argparse
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from ipaddress import ip_address
import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

import anyio
import httpx
import numpy as np
from jsonschema.validators import validator_for
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from referencing import Registry

from .catalog_lexical import CatalogLexicalIndex, top_indices
from .cli import checkpoint


class UpstreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    transport: Literal["stdio", "streamable_http"] = "stdio"
    command: str | None = Field(default=None, min_length=1)
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    bearer_token_env: str | None = Field(default=None, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    discover: set[str]
    invoke: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def validate_settings(self):
        if not self.invoke <= self.discover:
            raise ValueError("invoke allowlist must be a subset of discover")
        if self.transport == "stdio":
            if self.command is None or self.url is not None or self.bearer_token_env is not None:
                raise ValueError("stdio requires command and accepts only process settings")
        else:
            if self.url is None or self.command is not None or self.args or self.cwd is not None or self.env:
                raise ValueError("streamable_http requires url and accepts only HTTP settings")
            validate_http_endpoint(self.url)
        return self

    def process_parameters(self):
        if self.command is None:
            raise ValueError("This upstream does not start a process")
        return StdioServerParameters(command=self.command, args=self.args, cwd=self.cwd,
                                     env={key: os.environ[value] for key, value in self.env.items()})


def validate_http_endpoint(value):
    try:
        endpoint = httpx.URL(value)
    except httpx.InvalidURL:
        raise ValueError("Invalid Streamable HTTP endpoint") from None
    if endpoint.scheme not in {"http", "https"} or not endpoint.host or endpoint.userinfo or "#" in value:
        raise ValueError("Use an absolute MCP endpoint without user information or fragments")
    if endpoint.scheme == "http":
        try:
            loopback = ip_address(endpoint.host).is_loopback
        except ValueError:
            loopback = endpoint.host == "localhost"
        if not loopback:
            raise ValueError("Remote Streamable HTTP endpoints require HTTPS")


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    servers: list[UpstreamConfig]
    ranking: Literal["hybrid", "bm25"] = "hybrid"
    timeout_seconds: float = Field(default=10, gt=0, le=300)
    refresh_seconds: float = Field(default=30, gt=0, le=3600)
    min_cosine: float | None = Field(default=None, ge=-1, le=1)
    calibration_report: Path | None = None
    cache_dir: Path | None = None

    @model_validator(mode="after")
    def unique_servers(self):
        if len({server.id for server in self.servers}) != len(self.servers):
            raise ValueError("server IDs must be unique")
        if self.ranking == "bm25" and self.min_cosine is not None:
            raise ValueError("min_cosine requires hybrid ranking")
        if self.calibration_report is not None and (self.ranking != "hybrid" or self.min_cosine is not None):
            raise ValueError("calibration_report requires hybrid ranking and replaces min_cosine")
        return self


@dataclass(frozen=True)
class Entry:
    server: str
    tool: types.Tool
    invocable: bool
    revision: str

    @property
    def id(self):
        return f"{self.server}/{self.tool.name}"

    @property
    def text(self):
        return f"{self.tool.name}: {self.tool.description or ''}"

    def public(self):
        return {"id": self.id, "revision": self.revision, "server": self.server,
                "invocable": self.invocable, "name": self.tool.name,
                "description": self.tool.description, "inputSchema": deepcopy(self.tool.inputSchema)}


class CalibrationScope(BaseModel):
    catalog_sha256: str
    scorer_sha256: str
    invocable_only: bool


class CalibratedCutoff(BaseModel):
    schema_version: Literal[1]
    status: Literal["calibrated"]
    min_score: float = Field(allow_inf_nan=False)
    scope: CalibrationScope


def catalog_digest(entries, invocable_only):
    records = [(entry.id, entry.revision, entry.invocable) for entry in entries
               if not invocable_only or entry.invocable]
    return sha256(json.dumps(sorted(records), separators=(",", ":")).encode()).hexdigest()


def failure(code, message, **fields):
    return types.CallToolResult(isError=True, content=[types.TextContent(type="text", text=message)],
                                structuredContent={"code": code, **fields})


async def list_all_tools(session):
    tools = {}
    cursors = set()
    cursor = None
    for _ in range(128):
        page = await session.list_tools(params=types.PaginatedRequestParams(cursor=cursor))
        for tool in page.tools:
            if tool.name in tools:
                raise ValueError("duplicate tool on one server")
            tools[tool.name] = tool
        cursor = page.nextCursor
        if cursor is None:
            return tools
        if cursor in cursors:
            raise ValueError("repeated pagination cursor")
        cursors.add(cursor)
    raise ValueError("upstream page limit exceeded")


class Upstream:
    def __init__(self, config, timeout, interval):
        self.config = config
        self.timeout = timeout
        self.interval = interval
        self.ready = asyncio.Event()
        self.changed = asyncio.Event()
        self.lock = asyncio.Lock()
        self.session = None
        self.entries = {}
        self.state = "connecting"
        self.reconnect = False
        self.stopping = False

    async def on_message(self, message):
        if isinstance(message, types.ServerNotification) and isinstance(
            message.root, types.ToolListChangedNotification
        ):
            self.entries = {}
            self.changed.set()

    async def refresh(self, session):
        async with self.lock:
            self.changed.clear()
            async with asyncio.timeout(self.timeout):
                tools = await list_all_tools(session)
            if self.changed.is_set():
                return
            entries = {}
            for name in tools.keys() & self.config.discover:
                tool = tools[name]
                canonical = json.dumps([tool.description, tool.inputSchema, tool.outputSchema],
                                       sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                revision = sha256(canonical.encode()).hexdigest()
                entry = Entry(self.config.id, tool, name in self.config.invoke, revision)
                entries[entry.id] = entry
            self.session = session
            self.entries = entries
            self.state = "ready"
            self.ready.set()

    async def run(self):
        while not self.stopping:
            try:
                self.reconnect = False
                async with self.connect() as (read, write):
                    async with ClientSession(read, write, message_handler=self.on_message) as session:
                        try:
                            async with asyncio.timeout(self.timeout):
                                await session.initialize()
                            while not self.reconnect and not self.stopping:
                                await self.refresh(session)
                                if not self.stopping:
                                    await self.wait_for_change(self.interval)
                        finally:
                            self.clear_catalog()
            except Exception:
                self.clear_catalog()
            finally:
                self.ready.set()
            if not self.stopping:
                self.changed.clear()
                await self.wait_for_change(min(self.interval, 2.0))

    @asynccontextmanager
    async def connect(self):
        if self.config.transport == "stdio":
            with open(os.devnull, "w") as quiet:
                async with stdio_client(self.config.process_parameters(), errlog=quiet) as streams:
                    yield streams
            return

        headers = {}
        if self.config.bearer_token_env:
            token = os.environ[self.config.bearer_token_env]
            if not token.strip():
                raise ValueError("MCP bearer token is empty")
            headers["Authorization"] = f"Bearer {token}"

        # Bound session cleanup separately from idle server-event streams.
        async def limit_cleanup(request):
            if request.method == "DELETE":
                request.extensions["timeout"] = httpx.Timeout(self.timeout).as_dict()

        async def handle_session_expiry(response):
            request = response.request
            if response.status_code == 404 and request.method == "POST" and "mcp-session-id" in request.headers:
                self.invalidate()

        async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(self.timeout, read=300),
                                     trust_env=False, max_redirects=0,
                                     event_hooks={"request": [limit_cleanup],
                                                  "response": [handle_session_expiry]}) as client:
            async with streamable_http_client(self.config.url, http_client=client) as (read, write, _):
                yield read, write

    async def wait_for_change(self, delay):
        try:
            await asyncio.wait_for(self.changed.wait(), delay)
        except TimeoutError:
            pass

    def stop(self):
        self.stopping = True
        self.changed.set()

    def clear_catalog(self):
        self.session = None
        self.entries = {}
        self.state = "unavailable"

    def invalidate(self):
        self.clear_catalog()
        self.reconnect = True
        self.changed.set()

    async def invoke(self, tool_id, revision, arguments):
        async with self.lock:
            entry = self.entries.get(tool_id)
            if entry is None or not entry.invocable or self.session is None:
                return failure("not_available", "Tool is unavailable or outside the invocation allowlist.")
            if entry.revision != revision:
                return failure("stale_catalog", "Discover the current tool definition before invoking.")
            try:
                schema = entry.tool.inputSchema
                validator = validator_for(schema)
                validator.check_schema(schema)
                valid = validator(schema, registry=Registry()).is_valid(arguments)
            except Exception:
                return failure("invalid_schema", "The upstream input schema could not be validated locally.")
            if not valid:
                return failure("invalid_arguments", "Arguments do not match the tool input schema.")
            try:
                async with asyncio.timeout(self.timeout):
                    result = await self.session.call_tool(entry.tool.name, arguments)
            except asyncio.CancelledError:
                self.invalidate()
                raise
            except TimeoutError:
                self.invalidate()
                return failure("upstream_timeout", "The upstream did not respond. Execution may have occurred.")
            except McpError as exc:
                return failure("upstream_rpc_error", "The upstream rejected the request.", upstream_code=exc.error.code)
            except Exception:
                self.invalidate()
                return failure("upstream_unavailable", "The upstream connection failed. Execution may have occurred.")
            if result.isError:
                return failure("upstream_tool_error", "The upstream reported a tool execution error.")
            return result


async def list_upstream_tools(config):
    """Inventory configured upstreams for the local operator."""
    async def inspect(server):
        upstream = Upstream(server, config.timeout_seconds, config.refresh_seconds)
        try:
            async with upstream.connect() as (read, write):
                async with ClientSession(read, write) as session:
                    async with asyncio.timeout(upstream.timeout):
                        initialized = await session.initialize()
                        tools = await list_all_tools(session) if initialized.capabilities.tools is not None else {}
        except Exception:
            return {"status": "unavailable", "tools": []}
        return {"status": "ready", "tools": sorted(tools),
                "missing_discover": sorted(server.discover - tools.keys())}

    inventories = await asyncio.gather(*(inspect(server) for server in config.servers))
    return dict(zip((server.id for server in config.servers), inventories, strict=True))


class SearchIndex:
    def __init__(self, encoder=None, *, namespace=None, cache_dir=None):
        self.encoder = encoder
        self.namespace = namespace
        self.cache_dir = cache_dir
        self.vectors = {}
        self.key = None
        self.lexical = None
        self.documents = None
        self.lock = Lock()

    @property
    def scorer_digest(self):
        scorer = "gateway-bm25-v1" if self.encoder is None else "gateway-max-cosine-v1"
        return sha256(f"{scorer}\0{np.__version__}\0{self.namespace}".encode()).hexdigest()

    def _path(self, text):
        cache_key = sha256(f"{self.namespace}\0{text}".encode()).hexdigest()
        return self.cache_dir / f"toolrank-v1-{cache_key}.npz"

    def _update(self, entries):
        texts = [entry.text for entry in entries]
        if texts == self.key:
            return
        if self.encoder is not None:
            self._update_embeddings(texts)
        self.lexical = CatalogLexicalIndex(texts, bm25_only=self.encoder is None)
        self.key = texts

    def _update_embeddings(self, texts):
        if self.cache_dir is not None:
            from . import rank_onnx

        wanted = dict.fromkeys(texts)
        missing = []
        for text in wanted:
            if text in self.vectors:
                wanted[text] = self.vectors[text]
            elif self.cache_dir is not None:
                hit = rank_onnx.read_cached_vectors(self._path(text), 1, 384, "float32")
                if hit is not None:
                    wanted[text] = hit[0][0]
            if wanted[text] is None:
                missing.append(text)
        for start in range(0, len(missing), 32):
            work = missing[start:start + 32]
            for text, vector in zip(work, self.encoder.encode(work), strict=True):
                wanted[text] = vector
                if self.cache_dir is not None:
                    try:
                        rank_onnx.write_cached_vectors(self._path(text), (vector[None, :], None))
                    except OSError:
                        pass
        self.vectors = wanted
        self.documents = np.stack([wanted[text] for text in texts])

    def search(self, entries, query, limit, min_cosine, *, invocable_only=False):
        if not entries or not query.strip():
            return []
        with self.lock:
            self._update(entries)
            cosine = None
            if self.encoder is None:
                scores = self.lexical.bm25_scores(query)
                valid = np.flatnonzero(scores > 0)
            else:
                cosine = self._cosine(query)
                scores = self.lexical.blend(query, cosine)
                valid = np.arange(len(entries)) if min_cosine is None else np.flatnonzero(cosine >= min_cosine)
        if invocable_only:
            valid = np.fromiter((index for index in valid if entries[index].invocable), dtype=np.intp)
        order = valid[top_indices(scores[valid], limit)]
        return [entries[i].public() | {"cosine": None if cosine is None else float(cosine[i]),
                                       "ranking_score": float(scores[i])} for i in order]

    def _cosine(self, query):
        query_vector = self.encoder.encode([self.encoder.prefix + query])[0]
        # Avoid dispatching a single-query dot product to a threaded BLAS pool.
        return np.einsum("ij,j->i", self.documents, query_vector)

    def max_cosines(self, entries, queries, *, invocable_only=False):
        if self.encoder is None or not entries:
            raise ValueError("Cosine calibration needs a hybrid index and a nonempty catalog")
        eligible = np.array([not invocable_only or entry.invocable for entry in entries])
        if not eligible.any():
            raise ValueError("Cosine calibration needs eligible tools")
        with self.lock:
            self._update(entries)
            return [self._cosine(query)[eligible].max() for query in queries]


def limit_tool_bytes(matches, budget):
    """Keep the longest ranked prefix fitting in a compact UTF-8 JSON array."""
    total = 2
    for index, match in enumerate(matches):
        total += len(json.dumps(match, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        total += int(index > 0)
        if total > budget:
            return matches[:index]
    return matches


class Gateway:
    def __init__(self, config, index, *, calibration=None):
        self.config = config
        self.index = index
        self.calibration = calibration
        self.upstreams = {server.id: Upstream(server, config.timeout_seconds, config.refresh_seconds)
                          for server in config.servers}
        self.search_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: GatewayConfig, checkpoint_dir: Path | None = None):
        if config.ranking == "bm25":
            return cls(config, SearchIndex())
        from . import rank_onnx

        model = checkpoint_dir or checkpoint()
        if config.cache_dir is not None:
            config.cache_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        encoder = rank_onnx.OnnxToolRank(model, [])
        namespace = json.dumps({
            "checkpoint": rank_onnx.checkpoint_checksums(model, "embedding-int8.onnx"),
            "onnxruntime": rank_onnx.ort.__version__,
            "tokenizers": rank_onnx.tokenizers_version,
        }, sort_keys=True)
        index = SearchIndex(encoder, namespace=namespace, cache_dir=config.cache_dir)
        calibration = None
        if config.calibration_report is not None:
            calibration = CalibratedCutoff.model_validate_json(config.calibration_report.read_text(encoding="utf-8"))
            if calibration.scope.scorer_sha256 != index.scorer_digest:
                raise ValueError("Calibration report was fitted with a different scorer")
        return cls(config, index, calibration=calibration)

    def catalog_entries(self):
        return sorted((entry for upstream in self.upstreams.values() for entry in upstream.entries.values()),
                      key=lambda entry: entry.id)

    def calibration_scope(self, entries, invocable_only):
        return CalibrationScope(catalog_sha256=catalog_digest(entries, invocable_only),
                                scorer_sha256=self.index.scorer_digest, invocable_only=invocable_only)

    def _calibration_matches(self, entries, invocable_only):
        return self.calibration is None or self.calibration.scope == self.calibration_scope(entries, invocable_only)

    def _search_response(self, matches, decision):
        return {"matches": matches, "decision": decision,
                "upstreams": {key: upstream.state for key, upstream in self.upstreams.items()}}

    @asynccontextmanager
    async def running(self):
        for upstream in self.upstreams.values():
            upstream.stopping = False
            upstream.ready.clear()
        workers = [asyncio.create_task(upstream.run()) for upstream in self.upstreams.values()]
        try:
            try:
                async with asyncio.timeout(self.config.timeout_seconds):
                    await asyncio.gather(*(upstream.ready.wait() for upstream in self.upstreams.values()))
            except TimeoutError:
                pass
            yield self
        finally:
            for upstream in self.upstreams.values():
                upstream.stop()
            # Let the SDK close the owned stdio processes before the loop exits.
            with anyio.CancelScope(shield=True):
                await asyncio.gather(*workers)

    async def search(self, query, limit=5, *, invocable_only=False, max_tool_bytes=None):
        if not 1 <= limit <= 50 or not 1 <= len(query) <= 8192:
            raise ValueError("search needs a query of 1–8192 characters and a limit from 1 to 50")
        if max_tool_bytes is not None and (type(max_tool_bytes) is not int or max_tool_bytes < 2):
            raise ValueError("max_tool_bytes must be an integer of at least two")
        async with self.search_lock:
            entries = self.catalog_entries()
            if not self._calibration_matches(entries, invocable_only):
                return self._search_response([], "calibration_scope_mismatch")
            threshold = self.calibration.min_score if self.calibration is not None else self.config.min_cosine
            searchable = bool(entries) and (not invocable_only or any(entry.invocable for entry in entries))
            matches = []
            if searchable:
                matches = await asyncio.to_thread(self.index.search, entries, query, limit, threshold,
                                                  invocable_only=invocable_only)
            if self.calibration is not None and not self._calibration_matches(self.catalog_entries(), invocable_only):
                return self._search_response([], "calibration_scope_mismatch")
            current = [match for match in matches if self._current(match)]
            if not current:
                decision = "empty"
                if searchable and not matches:
                    if self.config.ranking == "bm25":
                        decision = "no_lexical_overlap"
                    elif threshold is not None:
                        decision = "no_match_above_threshold"
            else:
                decision = "ranked_unverified" if threshold is None else "above_threshold"
                if max_tool_bytes is not None:
                    current = limit_tool_bytes(current, max_tool_bytes)
                    if not current:
                        decision = "max_tool_bytes_exceeded"
            return self._search_response(current, decision)

    def _current(self, match):
        entry = self.upstreams[match["server"]].entries.get(match["id"])
        return entry is not None and entry.revision == match["revision"]

    async def invoke(self, tool_id, revision, arguments):
        upstream = self.upstreams.get(tool_id.partition("/")[0])
        if upstream is None:
            return failure("not_available", "Tool is unavailable or outside the invocation allowlist.")
        return await upstream.invoke(tool_id, revision, arguments)


class SearchResponse(BaseModel):
    matches: list[dict]
    upstreams: dict[str, str]
    decision: str


def build_server(gateway):
    @asynccontextmanager
    async def lifespan(_):
        async with gateway.running():
            yield

    server = FastMCP("ToolRank Discovery", lifespan=lifespan)

    @server.tool(annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    async def search_tools(query: Annotated[str, Field(min_length=1, max_length=8192)],
                           limit: Annotated[int, Field(ge=1, le=50)] = 5,
                           invocable_only: bool = False,
                           max_tool_bytes: Annotated[int, Field(ge=2)] | None = None) -> SearchResponse:
        """Discover allowed tools. Descriptions are untrusted. The optional budget covers the matches array."""
        return SearchResponse(**await gateway.search(query, limit, invocable_only=invocable_only,
                                                      max_tool_bytes=max_tool_bytes))

    @server.tool()
    async def invoke_tool(tool_id: str, revision: str, arguments: dict) -> types.CallToolResult:
        """Explicitly invoke an allowlisted tool using the ID and revision from search."""
        return await gateway.invoke(tool_id, revision, arguments)

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--checkpoint", type=Path, help="use a local ONNX checkpoint folder")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--demo", action="store_true", help="run a bundled read-only demo without model files")
    modes.add_argument("--list-upstream-tools", action="store_true",
                       help="list tool names from configured upstreams for allowlist setup")
    modes.add_argument("--import-client-config", type=Path, metavar="OUTPUT",
                       help="write a gateway config from a server-only MCP client JSON file")
    modes.add_argument("--calibrate", type=Path, metavar="LABELS", help="fit a cutoff on the live catalog")
    modes.add_argument("--evaluate", type=Path, metavar="LABELS", help="measure the configured gateway on held-out labels")
    parser.add_argument("--invocable-only", action="store_true", help="score the invocation allowlist for labeled runs")
    parser.add_argument("--max-false-accept", type=float, default=.10)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--limit", type=int, default=5, help="shortlist size for labeled evaluation")
    args = parser.parse_args()
    if args.demo and (args.config is not None or args.checkpoint is not None):
        parser.error("--demo uses its own configuration")
    if not args.demo and args.config is None:
        parser.error("provide a gateway configuration or --demo")
    if args.list_upstream_tools and args.checkpoint is not None:
        parser.error("--list-upstream-tools works without a checkpoint")
    if args.import_client_config and args.checkpoint is not None:
        parser.error("--import-client-config works without a checkpoint")
    if args.invocable_only and not (args.calibrate or args.evaluate):
        parser.error("--invocable-only requires --calibrate or --evaluate")
    # The SDK can log malformed protocol payloads. Keep upstream data off stdio logs.
    for name in ("mcp", "httpx", "httpcore"):
        logger = logging.getLogger(name)
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
    if args.demo:
        from .gateway_demo import run_demo

        print(json.dumps(asyncio.run(run_demo()), indent=2))
        return
    if args.import_client_config:
        from .gateway_import import ImportConfigError, convert_client_config, write_new_config

        try:
            converted = convert_client_config(args.config)
            GatewayConfig.model_validate(converted)
            write_new_config(args.import_client_config, converted)
        except ImportConfigError as error:
            parser.exit(2, f"{error}\n")
        except FileExistsError:
            parser.exit(2, "Gateway config output already exists.\n")
        except (OSError, ValueError):
            parser.exit(2, "Could not validate or write the imported gateway config.\n")
        return
    try:
        config = GatewayConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        parser.exit(2, "Could not load ToolRank gateway configuration.\n")
    if args.list_upstream_tools:
        inventory = asyncio.run(list_upstream_tools(config))
        print(json.dumps({"upstreams": inventory}, indent=2))
        if any(result["status"] == "unavailable" for result in inventory.values()):
            raise SystemExit(1)
        return
    try:
        if config.calibration_report is not None:
            config = config.model_copy(update={"calibration_report": args.config.parent / config.calibration_report})
        gateway = Gateway.from_config(config, args.checkpoint)
    except (OSError, ValueError):
        parser.exit(2, "Could not load ToolRank gateway configuration, checkpoint, or calibration report.\n")
    if args.calibrate or args.evaluate:
        from .gateway_calibration import run_labeled

        try:
            report = asyncio.run(run_labeled(gateway, args))
        except ValueError as error:
            parser.exit(2, f"Gateway labeled run failed. {error}\n")
        print(json.dumps(report, indent=2))
        return
    build_server(gateway).run()


if __name__ == "__main__":
    main()
