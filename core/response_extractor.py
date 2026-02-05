"""
ResponseExtractor - Extract metadata from agent responses using LLM.

Analyzes agent responses to extract:
- Summary: 1-sentence summary of the response
- Topics: Main topics covered (max 5)
- Keywords: Keywords for search (max 10)
- Response type: greeting | explanation | recommendation | opinion | action
- Scope: global (reusable FAQ), user (specific to user), group (specific to group)
"""
import json
from typing import Dict, Any, Optional


class ResponseExtractor:
    """Extract metadata from agent responses using LLM."""

    EXTRACTION_PROMPT = """Analyze this agent response and extract:

1. A 1-sentence summary (concise, under 20 words)
2. Main topics (max 5, lowercase, e.g., "base", "grants", "defi")
3. Keywords for search (max 10, lowercase, relevant terms)
4. Response type: greeting | explanation | recommendation | opinion | action | answer | other
5. Scope: global (reusable FAQ/info), user (specific to this user), group (specific to group context)

User's question: {trigger_message}
Agent's response: {response_content}

Return ONLY valid JSON, no markdown:
{{
  "summary": "...",
  "topics": ["...", "..."],
  "keywords": ["...", "..."],
  "response_type": "...",
  "scope": "..."
}}"""

    def __init__(self, llm_client):
        """
        Initialize ResponseExtractor.

        Args:
            llm_client: LLM client with call() method (e.g., OpenRouter client)
        """
        self.llm_client = llm_client

    def extract(self, trigger_message: str, response_content: str) -> Dict[str, Any]:
        """
        Extract metadata from response.

        Args:
            trigger_message: User's message that triggered the response
            response_content: Agent's full response

        Returns:
            Dict with keys: summary, topics, keywords, response_type, scope
        """
        if not self.llm_client:
            # Return defaults if no LLM client
            return self._get_defaults(response_content)

        prompt = self.EXTRACTION_PROMPT.format(
            trigger_message=trigger_message,
            response_content=response_content[:2000]  # Truncate if too long
        )

        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )

            content = response.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

            parsed = json.loads(content)

            return {
                "summary": parsed.get("summary", "")[:200],
                "topics": self._validate_list(parsed.get("topics", []), max_items=5),
                "keywords": self._validate_list(parsed.get("keywords", []), max_items=10),
                "response_type": self._validate_response_type(parsed.get("response_type", "answer")),
                "scope": self._validate_scope(parsed.get("scope", "user")),
            }

        except Exception as e:
            print(f"[ResponseExtractor] LLM extraction failed: {e}")
            return self._get_defaults(response_content)

    def _get_defaults(self, response_content: str) -> Dict[str, Any]:
        """Get default values when LLM extraction fails."""
        # Truncate for summary
        summary = response_content[:100] + "..." if len(response_content) > 100 else response_content

        return {
            "summary": summary,
            "topics": [],
            "keywords": [],
            "response_type": "answer",
            "scope": "user",
        }

    def _validate_list(self, items: list, max_items: int) -> list:
        """Validate and limit list items."""
        if not isinstance(items, list):
            return []
        return [str(i).lower().strip() for i in items[:max_items] if i]

    def _validate_response_type(self, response_type: str) -> str:
        """Validate response type."""
        valid_types = {"greeting", "explanation", "recommendation", "opinion", "action", "answer", "other"}
        return response_type if response_type in valid_types else "answer"

    def _validate_scope(self, scope: str) -> str:
        """Validate scope."""
        valid_scopes = {"global", "user", "group"}
        return scope if scope in valid_scopes else "user"
