CATALOG = [
    {"name": "list_exhibits", "description": "List exhibits that are open today.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "find_accessible_entry", "description": "Locate a step-free entrance to the museum.", "inputSchema": {"type": "object", "properties": {}}},
]
QUERY = "Which entrance can I use with a wheelchair?"


def list_exhibits() -> str:
    """List exhibits that are open today."""
    return "Photography and ceramics"


def find_accessible_entry() -> str:
    """Locate a step-free entrance to the museum."""
    return "North entrance"
