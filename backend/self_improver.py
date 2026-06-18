"""
Self-improvement loop for RAG agent performance tracking.

Tracks query quality, retrieval accuracy, and answer helpfulness.
Uses feedback to adjust retrieval parameters over time.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
import time
from pathlib import Path


@dataclass
class ImprovementRecord:
    query: str
    answer: str
    sources_used: int
    retrieval_score: float
    user_feedback: Optional[int] = None  # 1-5 rating
    timestamp: float = field(default_factory=time.time)
    improved_params: Dict = field(default_factory=dict)


class SelfImprover:
    """
    SkillOpt loop for RAG:
    1. Track every query + answer + sources
    2. When user provides feedback (or auto-score), adjust parameters
    3. Over time, converge on optimal retrieval settings per document type
    """

    def __init__(self, storage_path: str = "./improvement_logs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history: List[ImprovementRecord] = []
        self.optimal_params: Dict[str, Dict] = {}
        self._load_history()

    def record_query(self, query: str, answer: str, source_count: int, retrieval_score: float):
        record = ImprovementRecord(
            query=query,
            answer=answer,
            sources_used=source_count,
            retrieval_score=retrieval_score,
        )
        self.history.append(record)
        self._save_record(record)

    def record_feedback(self, query_index: int, rating: int):
        if 0 <= query_index < len(self.history):
            self.history[query_index].user_feedback = rating
            if rating >= 4:
                self._update_optimal_params(query_index)

    def suggest_improvements(self, rag_pipeline) -> Dict:
        if len(self.history) < 5:
            return {"status": "not_enough_data", "message": "Need at least 5 queries for improvement suggestions"}

        recent = self.history[-20:]
        avg_retrieval = sum(r.retrieval_score for r in recent) / len(recent)
        avg_sources = sum(r.sources_used for r in recent) / len(recent)
        feedback_avg = sum(r.user_feedback for r in recent if r.user_feedback) / max(
            sum(1 for r in recent if r.user_feedback), 1
        )

        suggestions = {"status": "analyzed", "metrics": {"avg_retrieval_score": round(avg_retrieval, 3), "avg_sources_used": round(avg_sources, 1), "avg_user_rating": round(feedback_avg, 1), "total_queries": len(self.history)}, "improvements": []}

        if avg_retrieval < 0.6:
            suggestions["improvements"].append({
                "area": "retrieval",
                "action": "increase_top_k",
                "current": rag_pipeline.collection.count(),
                "suggestion": "Increase top_k or re-chunk with smaller chunk_size",
            })

        if avg_sources < 2:
            suggestions["improvements"].append({
                "area": "coverage",
                "action": "upload_more_documents",
                "suggestion": "Current source diversity is low. Upload more relevant papers.",
            })

        if feedback_avg > 0 and feedback_avg < 3:
            suggestions["improvements"].append({
                "area": "answer_quality",
                "action": "adjust_llm_params",
                "suggestion": "Lower temperature or switch to a stronger model for better answer precision.",
            })

        return suggestions

    def get_optimal_params(self) -> Dict:
        return self.optimal_params

    def get_history(self, limit: int = 10) -> List[Dict]:
        return [
            {
                "query": r.query[:100],
                "sources": r.sources_used,
                "score": r.retrieval_score,
                "feedback": r.user_feedback,
            }
            for r in self.history[-limit:]
        ]

    def _update_optimal_params(self, idx: int):
        record = self.history[idx]
        doc_sources = record.improved_params
        for key, value in doc_sources.items():
            if key not in self.optimal_params:
                self.optimal_params[key] = value
            else:
                self.optimal_params[key] = {
                    k: (self.optimal_params[key].get(k, 0) + v) / 2
                    for k, v in value.items()
                }

    def _save_record(self, record: ImprovementRecord):
        log_file = self.storage_path / "improvement_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps({
                "query": record.query[:200],
                "sources_used": record.sources_used,
                "retrieval_score": record.retrieval_score,
                "user_feedback": record.user_feedback,
                "timestamp": record.timestamp,
            }) + "\n")

    def _load_history(self):
        log_file = self.storage_path / "improvement_log.jsonl"
        if log_file.exists():
            with open(log_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        self.history.append(ImprovementRecord(**data))
                    except (json.JSONDecodeError, TypeError):
                        pass
