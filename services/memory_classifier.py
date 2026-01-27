"""
LLM-based Message Analyzer for Group Memory System

Uses LLM to intelligently analyze messages and decide what memories to create.
This replaces keyword-based detection with actual understanding.
"""
import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from utils.structured_schemas import MESSAGE_ANALYSIS_SCHEMA
import config


class MessageAnalyzer:
    """
    Analyzes messages using LLM to determine what memories to create.

    Replaces brittle keyword matching with intelligent semantic analysis.
    """

    def __init__(self):
        """Initialize the LLM client"""
        self.client = OpenAI(
            api_key=getattr(config, 'OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY')),
            base_url=getattr(config, 'OPENAI_BASE_URL', None)
        )
        self.model = getattr(config, 'LLM_MODEL', 'meta-llama/llama-3.1-8b-instruct')

        # Cache for recent analyses (to avoid duplicate LLM calls)
        self._cache = {}

    def analyze_message(
        self,
        message: str,
        username: str,
        group_context: Optional[str] = None,
        recent_messages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a message and determine what memories to create.

        Args:
            message: The message to analyze
            username: The sender's username
            group_context: Optional context about the group
            recent_messages: Optional recent messages for context

        Returns:
            Dict with analysis results including:
            - should_remember: bool
            - memories: list of memory objects to create
            - interaction_type: str or None
        """
        # Check cache
        cache_key = f"{username}:{message[:50]}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build prompt
        prompt = self._build_prompt(message, username, group_context, recent_messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format=MESSAGE_ANALYSIS_SCHEMA
            )

            result = json.loads(response.choices[0].message.content)

            # Validate and clean result
            result = self._validate_result(result, message)

            # Cache result
            self._cache[cache_key] = result

            return result

        except Exception as e:
            print(f"[MessageAnalyzer] Error analyzing message: {e}")
            # Return safe default - still create a memory
            return {
                "should_remember": True,
                "memories": [
                    {
                        "type": "conversation",
                        "content": message,
                        "importance": 0.5,
                        "reasoning": "Fallback due to analysis error"
                    }
                ],
                "interaction_type": None
            }

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """You are a memory classifier for an AI agent in a Telegram group about Base blockchain and DeFi.

Your job is to analyze messages and decide what information should be remembered.

MEMORY TYPES you can create:
1. "expertise" - User mentions their skills, experience, or knowledge
2. "preference" - User expresses likes, dislikes, or interests
3. "fact" - User shares factual information about themselves or their projects
4. "announcement" - Group announcements, decisions, or plans
5. "need" - User asks for help or expresses a need

Return a JSON object with this exact structure:
{
  "should_remember": true/false,
  "memories": [
    {
      "type": "expertise|preference|fact|announcement|need",
      "content": "brief summary of what to remember",
      "importance": 0.0-1.0,
      "topics": ["topic1", "topic2"],
      "keywords": ["keyword1", "keyword2"],
      "reasoning": "why this is important"
    }
  ],
  "interaction_type": "collaboration|question|offer_help|introduction|none"
}

RULES:
- Only create memories for substantive information (>5 words, meaningful content)
- Keep content concise but informative
- Importance: 0.9+ = critical, 0.7+ = important, 0.5+ = useful, <0.5 = minor
- Extract specific topics and keywords for retrieval
- interaction_type: only set if user is directly engaging with someone
- If message is just "hi", "thanks", "ok", set should_remember to false"""

    def _build_prompt(
        self,
        message: str,
        username: str,
        group_context: Optional[str],
        recent_messages: Optional[List[str]]
    ) -> str:
        """Build the analysis prompt"""
        prompt = f"""Analyze this message from @{username}:

Message: "{message}"
"""

        if group_context:
            prompt += f"\nGroup context: {group_context}\n"

        if recent_messages and len(recent_messages) > 0:
            prompt += f"\nRecent messages for context:\n"
            for i, msg in enumerate(recent_messages[-3:], 1):
                prompt += f"  {i}. {msg[:100]}...\n"

        prompt += "\nWhat should be remembered from this message? Return JSON only."

        return prompt

    def _validate_result(self, result: Dict, original_message: str) -> Dict:
        """Validate and clean the LLM result"""
        # Ensure required fields exist
        if "should_remember" not in result:
            result["should_remember"] = True

        if "memories" not in result:
            result["memories"] = []

        if "interaction_type" not in result:
            result["interaction_type"] = None

        # Validate memories
        valid_memories = []
        for mem in result.get("memories", []):
            if not isinstance(mem, dict):
                continue

            # Ensure required fields
            if "type" not in mem:
                mem["type"] = "conversation"
            if "content" not in mem:
                mem["content"] = original_message[:200]
            if "importance" not in mem:
                mem["importance"] = 0.5
            if "topics" not in mem:
                mem["topics"] = []
            if "keywords" not in mem:
                mem["keywords"] = []

            # Validate importance range
            mem["importance"] = max(0.0, min(1.0, mem["importance"]))

            # Validate memory type
            valid_types = ["expertise", "preference", "fact", "announcement", "need", "conversation"]
            if mem["type"] not in valid_types:
                mem["type"] = "conversation"

            valid_memories.append(mem)

        result["memories"] = valid_memories

        return result

    def analyze_batch(
        self,
        messages: List[Dict[str, str]],
        group_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple messages at once.

        Args:
            messages: List of {"message": str, "username": str}
            group_context: Optional group context

        Returns:
            List of analysis results
        """
        results = []

        for msg_data in messages:
            result = self.analyze_message(
                message=msg_data["message"],
                username=msg_data["username"],
                group_context=group_context
            )
            results.append(result)

        return results

    def clear_cache(self):
        """Clear the analysis cache"""
        self._cache.clear()


# Singleton instance
_analyzer_instance = None


def get_message_analyzer() -> MessageAnalyzer:
    """Get or create the singleton MessageAnalyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = MessageAnalyzer()
    return _analyzer_instance
