"""
Structured Output Schemas for OpenRouter json_schema response_format.

These schemas guarantee valid JSON output from LLM calls, eliminating parse failures.
Only used with OpenRouter (Llama 3.1 8B). Not for MiMo or a0x-models.

Usage:
    from utils.structured_schemas import QUERY_ANALYSIS_SCHEMA
    response = llm_client.chat_completion(messages, response_format=QUERY_ANALYSIS_SCHEMA)
    data = json.loads(response)  # Guaranteed valid JSON
"""

# ============================================================
# Memory Builder Schemas
# ============================================================

DM_MEMORY_ENTRIES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "dm_memory_entries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "lossless_restatement": {"type": "string"},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "timestamp": {"type": "string"},
                            "location": {"type": "string"},
                            "persons": {"type": "array", "items": {"type": "string"}},
                            "entities": {"type": "array", "items": {"type": "string"}},
                            "topic": {"type": "string"},
                            "memory_type": {"type": "string", "enum": ["expertise", "preference", "fact", "announcement", "conversation"]},
                            "importance_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "is_shareable": {"type": "boolean"}
                        },
                        "required": ["lossless_restatement", "keywords", "timestamp", "location", "persons", "entities", "topic", "memory_type", "importance_score", "is_shareable"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["reasoning", "entries"],
            "additionalProperties": False
        }
    }
}

GROUP_MEMORIES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "group_memories",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "group_memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "memory_type": {"type": "string", "enum": ["announcement", "fact", "conversation"]},
                            "speaker": {"type": "string"},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "topics": {"type": "array", "items": {"type": "string"}},
                            "importance_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "is_shareable": {"type": "boolean"}
                        },
                        "required": ["content", "memory_type", "speaker", "keywords", "topics", "importance_score", "is_shareable"],
                        "additionalProperties": False
                    }
                },
                "user_memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "user_id": {"type": "string"},
                            "username": {"type": "string"},
                            "memory_type": {"type": "string", "enum": ["expertise", "preference", "fact"]},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "topics": {"type": "array", "items": {"type": "string"}},
                            "importance_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "is_shareable": {"type": "boolean"}
                        },
                        "required": ["content", "user_id", "username", "memory_type", "keywords", "topics", "importance_score", "is_shareable"],
                        "additionalProperties": False
                    }
                },
                "interaction_memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "speaker_id": {"type": "string"},
                            "listener_id": {"type": "string"},
                            "interaction_type": {"type": "string", "enum": ["question", "answer", "discussion", "help"]},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "topics": {"type": "array", "items": {"type": "string"}},
                            "importance_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "is_shareable": {"type": "boolean"}
                        },
                        "required": ["content", "speaker_id", "listener_id", "interaction_type", "keywords", "topics", "importance_score", "is_shareable"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["reasoning", "group_memories", "user_memories", "interaction_memories"],
            "additionalProperties": False
        }
    }
}

# ============================================================
# User Profile Extraction Schema (replaces a0x-models /summarize + /keywords)
# ============================================================

USER_PROFILE_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "user_profile_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                },
                "interests": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "score": {"type": "number"}
                        },
                        "required": ["keyword", "score"],
                        "additionalProperties": False
                    }
                },
                "expertise_level": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "advanced", "expert"]
                },
                "communication_style": {
                    "type": "string",
                    "enum": ["formal", "casual", "technical", "conversational"]
                },
                "domains": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "enum": ["defi", "nfts", "gaming", "trading", "development", "design", "marketing", "music", "other"]},
                            "score": {"type": "number"}
                        },
                        "required": ["name", "score"],
                        "additionalProperties": False
                    }
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["person", "project", "protocol", "token", "organization", "tool", "other"]},
                            "name": {"type": "string"},
                            "context": {"type": "string"}
                        },
                        "required": ["type", "name", "context"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["summary", "interests", "expertise_level", "communication_style", "domains", "entities"],
            "additionalProperties": False
        }
    }
}

# ============================================================
# Hybrid Retriever Schemas
# ============================================================

QUERY_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "persons": {"type": "array", "items": {"type": "string"}},
                "time_expression": {"type": "string"},
                "location": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["reasoning", "keywords", "persons", "time_expression", "location", "entities"],
            "additionalProperties": False
        }
    }
}

SEARCH_QUERIES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "search_queries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["queries"],
            "additionalProperties": False
        }
    }
}

ANSWER_ADEQUACY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_adequacy",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "assessment": {"type": "string", "enum": ["sufficient", "insufficient"]},
                "missing_info": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["reasoning", "assessment", "missing_info"],
            "additionalProperties": False
        }
    }
}

ADDITIONAL_QUERIES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "additional_queries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "missing_analysis": {"type": "string"},
                "additional_queries": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["missing_analysis", "additional_queries"],
            "additionalProperties": False
        }
    }
}

INFORMATION_REQUIREMENTS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "information_requirements",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "question_type": {"type": "string"},
                "key_entities": {"type": "array", "items": {"type": "string"}},
                "required_info": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "info_type": {"type": "string"},
                            "description": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                        },
                        "required": ["info_type", "description", "priority"],
                        "additionalProperties": False
                    }
                },
                "relationships": {"type": "array", "items": {"type": "string"}},
                "minimal_queries_needed": {"type": "integer"},
                "exact_match_terms": {"type": "array", "items": {"type": "string"}},
                "use_keyword_boost": {"type": "boolean"}
            },
            "required": ["reasoning", "question_type", "key_entities", "required_info", "relationships", "minimal_queries_needed", "exact_match_terms", "use_keyword_boost"],
            "additionalProperties": False
        }
    }
}

TARGETED_QUERIES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "targeted_queries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "queries": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["reasoning", "queries"],
            "additionalProperties": False
        }
    }
}

INFORMATION_COMPLETENESS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "information_completeness",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "assessment": {"type": "string", "enum": ["complete", "incomplete"]},
                "missing_info_types": {"type": "array", "items": {"type": "string"}},
                "coverage_percentage": {"type": "integer"}
            },
            "required": ["reasoning", "assessment", "missing_info_types", "coverage_percentage"],
            "additionalProperties": False
        }
    }
}

MISSING_INFO_QUERIES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "missing_info_queries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "missing_analysis": {"type": "string"},
                "targeted_queries": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["missing_analysis", "targeted_queries"],
            "additionalProperties": False
        }
    }
}

# ============================================================
# Answer Generator Schema
# ============================================================

ANSWER_GENERATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_generation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {"type": "string"}
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False
        }
    }
}

# ============================================================
# Agent Response Metadata Schema
# ============================================================

AGENT_RESPONSE_METADATA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_response_metadata",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "summary": {"type": "string"},
                "response_type": {"type": "string", "enum": ["greeting", "answer", "clarification", "recommendation", "question", "acknowledgment", "other"]},
                "topics": {"type": "array", "items": {"type": "string"}},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "importance_score": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["reasoning", "summary", "response_type", "topics", "keywords", "importance_score"],
            "additionalProperties": False
        }
    }
}

# ============================================================
# Memory Classifier Schema
# ============================================================

MESSAGE_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "message_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "should_remember": {"type": "boolean"},
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["expertise", "preference", "fact", "announcement", "need", "conversation"]},
                            "content": {"type": "string"},
                            "importance": {"type": "number"},
                            "topics": {"type": "array", "items": {"type": "string"}},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["type", "content", "importance", "topics", "keywords", "reasoning"],
                        "additionalProperties": False
                    }
                },
                "interaction_type": {"type": "string", "enum": ["collaboration", "question", "offer_help", "introduction", "none"]}
            },
            "required": ["reasoning", "should_remember", "memories", "interaction_type"],
            "additionalProperties": False
        }
    }
}

# ============================================================
# LLM-as-Judge Evaluation Schemas (for tests)
# ============================================================

PROFILE_EVALUATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "profile_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "summary_quality": {"type": "number"},
                "interests_specificity": {"type": "number"},
                "facts_accuracy": {"type": "number"},
                "no_hallucination": {"type": "number"},
                "overall": {"type": "number"}
            },
            "required": ["reasoning", "summary_quality", "interests_specificity", "facts_accuracy", "no_hallucination", "overall"],
            "additionalProperties": False
        }
    }
}

GROUP_PROFILE_EVALUATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "group_profile_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "summary_relevance": {"type": "number"},
                "topics_quality": {"type": "number"},
                "tone_accuracy": {"type": "number"},
                "completeness": {"type": "number"},
                "overall": {"type": "number"}
            },
            "required": ["reasoning", "summary_relevance", "topics_quality", "tone_accuracy", "completeness", "overall"],
            "additionalProperties": False
        }
    }
}

INTERESTS_EVALUATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "interests_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "count_score": {"type": "number"},
                "specificity": {"type": "number"},
                "evidence_quality": {"type": "number"},
                "coverage": {"type": "number"},
                "overall": {"type": "number"}
            },
            "required": ["reasoning", "count_score", "specificity", "evidence_quality", "coverage", "overall"],
            "additionalProperties": False
        }
    }
}
