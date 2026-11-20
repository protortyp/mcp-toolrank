"""Exercise a packaged gateway with the pinned MCP Inspector CLI."""
import argparse
import json
from pathlib import Path
import subprocess


NODE = "24.22.1"
INSPECTOR = "2.16.0"


def inspect(gateway, method, *, name=None, arguments=None, exit_code=0):
    command = ["npm", "exec", "--yes", "--package", f"node@{NODE}", "--package",
               f"@modelcontextprotocol/inspector@{INSPECTOR}", "--", "mcp-inspector", "--cli",
               "--format", "json", "--method", method]
    if name is not None:
        command.extend(("--tool-name", name, "--tool-args-json", json.dumps(arguments)))
    result = subprocess.run(command + ["--", *gateway], capture_output=True, text=True, timeout=180)
    if result.returncode != exit_code:
        raise RuntimeError(f"Inspector {method} exited {result.returncode}, expected {exit_code}")
    return json.loads(result.stdout)["result"]


def smoke(gateway):
    exposed = sorted(tool["name"] for tool in inspect(gateway, "tools/list")["tools"])
    if exposed != ["invoke_tool", "search_tools"]:
        raise ValueError("Unexpected gateway methods")
    query = {"query": "What time is it in Tokyo?", "limit": 10}
    discovery = inspect(gateway, "tools/call", name="search_tools", arguments=query)
    matches = {match["id"]: match for match in discovery["structuredContent"]["matches"]}
    clock = matches["clock/get_current_time"]
    git = matches["repository/git_status"]
    if not clock["invocable"] or git["invocable"]:
        raise ValueError("Unexpected discovery or execution permissions")

    allowed = inspect(gateway, "tools/call", name="search_tools", arguments=query | {"invocable_only": True})
    allowed_ids = [match["id"] for match in allowed["structuredContent"]["matches"]]
    if allowed_ids != [clock["id"]]:
        raise ValueError("Execution-only search exposed an unapproved tool")

    approved_args = {"tool_id": clock["id"], "revision": clock["revision"], "arguments": {"timezone": "Asia/Tokyo"}}
    result = inspect(gateway, "tools/call", name="invoke_tool", arguments=approved_args)
    if result.get("isError") or result["structuredContent"]["timezone"] != "Asia/Tokyo":
        raise ValueError("Clock invocation failed")
    denied_args = {"tool_id": git["id"], "revision": git["revision"], "arguments": {"repo_path": "."}}
    blocked = inspect(gateway, "tools/call", name="invoke_tool", arguments=denied_args, exit_code=5)
    if not blocked.get("isError") or blocked["structuredContent"]["code"] != "not_available":
        raise ValueError("Gateway did not deny discovery-only tool")

    return {"client": f"MCP Inspector {INSPECTOR}", "node": NODE, "transport": "stdio",
            "methods": exposed, "discovered_tools": len(matches), "allowed_ids": allowed_ids,
            "approved_invocation": clock["id"], "denied_invocation": git["id"],
            "denial_code": blocked["structuredContent"]["code"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("examples/gateway_config.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    command = [str(args.gateway.absolute()), str(args.config.resolve()), "--checkpoint", str(args.checkpoint.resolve())]
    print(json.dumps(smoke(command), indent=2))
