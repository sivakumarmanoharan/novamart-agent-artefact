"""The two tools the agent can call, plus their declarations for Gemini.

1. search_policies  - retrieval over NovaMart's policy documents.
2. get_order_status - a stubbed "live" order API backed by data/orders.json.

The declarations are written out by hand (rather than generated from Python
signatures) so it is obvious exactly what the model is told about each tool.
"""

from __future__ import annotations

import json
import re

from google.genai import types

from novamart import config
from novamart.retrieval import PolicyIndex

DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_policies",
        description=(
            "Search NovaMart's internal policy documents (returns, refunds, price match, "
            "warranty/NovaCare, shipping and order statuses, employee discount, NovaRewards "
            "loyalty, gift cards). Returns the most relevant passages with a 'source' id to cite. "
            "Call this for ANY policy question before answering."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query, e.g. 'return window for opened laptops'.",
                }
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="get_order_status",
        description=(
            "Look up the live status of a NovaMart online order: status, dates, carrier, "
            "tracking number and items. Order ids look like NM-12345."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order id, e.g. NM-10001."}
            },
            "required": ["order_id"],
        },
    ),
]

ORDER_ID_RE = re.compile(r"^NM-\d{5}$")


class Toolbox:
    """Holds tool state (the policy index, the orders 'database')."""

    def __init__(self, index: PolicyIndex, orders: dict | None = None):
        self.index = index
        self.orders = orders if orders is not None else json.loads(config.ORDERS_FILE.read_text())

    def search_policies(self, query: str) -> dict:
        hits = [h for h in self.index.search(query) if h.score >= config.MIN_SCORE]
        if not hits:
            # Explicit signal for the model (and the eval) that nothing in the
            # docs covers this. The system prompt tells the model to decline.
            return {
                "results": [],
                "note": "NO_RELEVANT_POLICY: no policy passage matched this query.",
            }
        return {
            "results": [
                {
                    "source": h.chunk.id,
                    "document": h.chunk.doc_title,
                    "section": h.chunk.section,
                    "text": h.chunk.text,
                    "score": round(h.score, 3),
                }
                for h in hits
            ]
        }

    def get_order_status(self, order_id: str) -> dict:
        order_id = order_id.strip().upper()
        if not ORDER_ID_RE.match(order_id):
            return {"error": f"'{order_id}' is not a valid order id. Expected format NM-12345."}
        order = self.orders.get(order_id)
        if order is None:
            return {"error": f"Order {order_id} was not found."}
        # In production this would be an authenticated call to the order
        # management system; here it is a dictionary lookup.
        return {"order_id": order_id, **order}

    def call(self, name: str, args: dict) -> dict:
        """Dispatch a model-requested function call. Unknown tools are reported, not raised."""
        if name == "search_policies":
            return self.search_policies(str(args.get("query", "")))
        if name == "get_order_status":
            return self.get_order_status(str(args.get("order_id", "")))
        return {"error": f"Unknown tool '{name}'."}
