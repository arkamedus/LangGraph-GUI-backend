import sys
import json
from typing import Dict, Any, Callable, Optional

# Global metadata for logging
CURRENT_METADATA: Dict[str, Any] = {}

def flush_print(message: str, status: Optional[bool] = None):
    """
    Prints JSON-formatted logs with execution metadata.
    """
    log = {
        "graph": CURRENT_METADATA.get("graph"),
        "subgraph": CURRENT_METADATA.get("subgraph"),
        "node": CURRENT_METADATA.get("node"),
        "node_id": CURRENT_METADATA.get("node_id"),
        "node_type": CURRENT_METADATA.get("node_type"),
        "status": status,
        "message": message,#.replace("\n", "\\n")
    }
    print(json.dumps(log, ensure_ascii=False), flush=True)

def with_metadata(fn: Callable):
    def wrapped(state, *args, **kwargs):
        global CURRENT_METADATA

        # Extract node-specific metadata dynamically
        node_metadata = {
            "graph": kwargs.get("graph", "unknown_graph"),
            "subgraph": kwargs.get("subgraph", "unknown_subgraph"),
            "node": kwargs.get("name", "unknown_node"),
            "node_id": kwargs.get("node_id", "unknown_node_id"),  # Get node_id dynamically
            "node_type": kwargs.get("node_type", "unknown_type"),
        }

        CURRENT_METADATA.update(node_metadata)

        flush_print(f"START execution of {CURRENT_METADATA['node']} (ID: {CURRENT_METADATA['node_id']})", status=True)
        result = fn(state, *args, **kwargs)
        flush_print(f"END execution of {CURRENT_METADATA['node']} (ID: {CURRENT_METADATA['node_id']})", status=False)

        return result
    return wrapped



