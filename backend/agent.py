import json
from typing import Dict, List
from openai import OpenAI
import os


class RAGAgent:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        self.agent_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    def plan_search_queries(self, task: str, history: List[str] = None) -> List[str]:
        history_text = "\n".join(history) if history else "None"

        prompt = (
            "You are a research query planner. Given a complex research task, break it down "
            "into specific search queries that would help retrieve relevant information from "
            "academic papers. Output a JSON array of query strings.\n\n"
            f"Task: {task}\n"
            f"Previously answered sub-questions: {history_text}\n\n"
            "Generate 1-3 new, specific search queries as a JSON array. "
            "Only output the JSON array, nothing else."
        )

        response = self.client.chat.completions.create(
            model=self.agent_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512,
        )
        try:
            queries = json.loads(response.choices[0].message.content.strip())
            return queries if isinstance(queries, list) else []
        except json.JSONDecodeError:
            return [task]

    def synthesize(self, task: str, findings: List[Dict]) -> str:
        findings_text = ""
        for f in findings:
            findings_text += f"\n--- Query: {f['query']} ---\n{f['answer']}\n"

        prompt = (
            "You are a research synthesizer. Given a research task and findings from multiple "
            "sub-queries, produce a comprehensive, well-structured answer.\n\n"
            f"Research Task: {task}\n\n"
            f"Findings:\n{findings_text}\n\n"
            "Provide a comprehensive synthesis with clear sections:\n"
            "1. Summary\n2. Key Findings\n3. Gaps/Limitations\n4. References cited\n\n"
            "If specific sources were mentioned, cite them."
        )

        response = self.client.chat.completions.create(
            model=self.agent_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def execute(self, task: str, max_iterations: int = 3) -> Dict:
        all_findings = []
        seen_questions = set()
        all_sources = set()

        for iteration in range(max_iterations):
            history = [f["answer"] for f in all_findings] if all_findings else None
            queries = self.plan_search_queries(task, history)

            for query_text in queries:
                if query_text in seen_questions:
                    continue
                seen_questions.add(query_text)

                result = self.rag.query(query_text, top_k=4)
                all_findings.append({
                    "query": query_text,
                    "answer": result["answer"],
                })
                for s in result["sources"]:
                    all_sources.add(s["source"])

        synthesis = self.synthesize(task, all_findings)

        return {
            "task": task,
            "iterations_performed": iteration + 1,
            "sub_queries_used": len(seen_questions),
            "sources_consulted": list(all_sources),
            "synthesis": synthesis,
            "findings": all_findings,
        }
