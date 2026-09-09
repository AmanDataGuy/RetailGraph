"""
Structured request logging — one JSON line per request.

Plain logging.info(f"...") calls are only useful read one at a time.
This emits a single machine-parseable JSON line per request instead, so
logs can later be piped into any aggregator (or just grepped/jq'd) to
answer questions like "what's our route mix" or "what's p95 latency"
without reading every line by hand.
"""
import json
import logging

_log = logging.getLogger("retailgraph.requests")


def log_request(**fields) -> None:
    """Emit one structured JSON log line. Pass whatever fields matter."""
    _log.info(json.dumps(fields, default=str))
