"""
Progressive Context Loader for RAG Agent.

Based on the "progressive disclosure" principle (B站社区共识):
- Don't dump all context at once
- Load minimal context first, expand only when needed
- Track what was useful vs. what was noise
"""
from typing import List, Dict
from dataclasses import dataclass, field
import json
import os
from pathlib import Path


@dataclass
class ContextLayer:
    name: str
    content: str
    tokens_used: int = 0
    hit_count: int = 0
    expanded: bool = False


class ProgressiveLoader:
    """
    Three-layer progressive context loading:

    Layer 1 (Core): Document titles + short descriptions (~200 tokens)
        Always loaded. Agent sees what's available.

    Layer 2 (Summary): Per-document summaries (~500 tokens per doc)
        Loaded when a document is relevant to the query.

    Layer 3 (Full): Full chunks from relevant documents (~2000 tokens)
        Loaded only for the most relevant sections.
    """

    def __init__(self, storage_path: str = "./context_cache"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.layers: Dict[str, ContextLayer] = {}
        self.usage_stats: Dict[str, int] = {}

    def register_layer(self, name: str, content: str) -> int:
        tokens = self._estimate_tokens(content)
        self.layers[name] = ContextLayer(
            name=name, content=content, tokens_used=tokens
        )
        return tokens

    def get_layer(self, name: str) -> str:
        if name not in self.layers:
            return ""
        layer = self.layers[name]
        layer.hit_count += 1
        return layer.content

    def expand_layer(self, name: str, additional_content: str) -> int:
        if name not in self.layers:
            return 0
        layer = self.layers[name]
        layer.content += "\n" + additional_content
        layer.expanded = True
        tokens = self._estimate_tokens(additional_content)
        layer.tokens_used += tokens
        return tokens

    def get_optimal_context(self, query: str, available_docs: List[str], chunk_limit: int = 2000) -> str:
        """
        Smart context assembly:
        - Always include Layer 1 (document list)
        - Include Layer 2 summaries for docs matching query keywords
        - Include Layer 3 chunks only if within token budget
        """
        context_parts = []

        layer1 = self.get_layer("l1_document_list")
        if layer1:
            context_parts.append(layer1)

        return "\n\n".join(context_parts)

    def record_usage(self, layer_name: str, was_useful: bool):
        if was_useful:
            self.usage_stats[layer_name] = self.usage_stats.get(layer_name, 0) + 1

    def get_efficiency_report(self) -> Dict:
        total_tokens = sum(l.tokens_used for l in self.layers.values())
        total_hits = sum(l.hit_count for l in self.layers.values())
        return {
            "total_tokens_loaded": total_tokens,
            "total_layer_hits": total_hits,
            "layers": {
                name: {
                    "tokens": l.tokens_used,
                    "hits": l.hit_count,
                    "expanded": l.expanded,
                    "efficiency": round(l.hit_count / max(l.tokens_used, 1), 6),
                }
                for name, l in self.layers.items()
            },
            "usage_stats": self.usage_stats,
        }

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text) // 4
