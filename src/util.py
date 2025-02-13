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
        "message": message.replace("\n", "\\n")
    }
    print(json.dumps(log, ensure_ascii=False), flush=True)

def with_metadata(fn: Callable, sg_name: str, node_name: str, node_id: str, node_type: str):
    """
    Decorator to inject metadata before execution and log start/end events.
    """
    def wrapped(state, *args, **kwargs):
        global CURRENT_METADATA
        CURRENT_METADATA = {
            "graph": sg_name,
            "subgraph": sg_name,
            "node": node_name,
            "node_id": node_id,
            "node_type": node_type
        }
        flush_print("START execution", status=True)
        result = fn(state, *args, **kwargs)
        flush_print("END execution", status=False)
        return result
    return wrapped
