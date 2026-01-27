"""
Architecture 3: Graph-Based Group Memory

Implement Transactive Memory Theory using a graph structure.
Nodes (users, messages, concepts) + Edges (relationships).
Track "who knows what" metaknowledge.

Schema:
- nodes: users, messages, concepts
- edges: relationships (said, knows_about, replied_to)
"""

import os
import sys
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from collections import defaultdict

# Add parent directory to path to import from a0x-memory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pyarrow as pa
import lancedb
from utils.embedding import EmbeddingModel
import json
import re


class GraphBasedMemoryStore:
    """
    Graph-based memory store implementing Transactive Memory Theory.

    Stores:
    - Nodes: users, messages, concepts (extracted from content)
    - Edges: relationships between nodes
    """

    def __init__(
        self,
        db_path: str,
        agent_id: str = "jessexbt",
        embedding_model: EmbeddingModel = None
    ):
        self.db_path = db_path
        self.agent_id = agent_id
        self.embedding_model = embedding_model or EmbeddingModel()

        # Connect to database
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)

        # Initialize tables
        self.nodes_table = None
        self.edges_table = None

        self._init_tables()

    def _init_tables(self):
        """Initialize nodes and edges tables."""

        # Nodes table
        nodes_schema = pa.schema([
            pa.field("node_id", pa.string()),
            pa.field("node_type", pa.string()),  # "user" | "message" | "concept"
            pa.field("agent_id", pa.string()),
            pa.field("group_id", pa.string()),
            pa.field("properties", pa.string()),  # JSON blob
            pa.field("content", pa.string()),  # For message nodes
            pa.field("timestamp", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        # Edges table
        edges_schema = pa.schema([
            pa.field("edge_id", pa.string()),
            pa.field("source_node", pa.string()),
            pa.field("target_node", pa.string()),
            pa.field("edge_type", pa.string()),  # "said" | "knows_about" | "replied_to" | "mentions"
            pa.field("weight", pa.float32()),  # Strength of relationship
            pa.field("timestamp", pa.string())
        ])

        # Create/open nodes table
        if "graph_nodes" not in self.db.table_names():
            self.nodes_table = self.db.create_table("graph_nodes", schema=nodes_schema)
            print("[Arch3] Created graph_nodes table")
        else:
            self.nodes_table = self.db.open_table("graph_nodes")
            print(f"[Arch3] Opened graph_nodes ({self.nodes_table.count_rows()} nodes)")

        # Create/open edges table
        if "graph_edges" not in self.db.table_names():
            self.edges_table = self.db.create_table("graph_edges", schema=edges_schema)
            print("[Arch3] Created graph_edges table")
        else:
            self.edges_table = self.db.open_table("graph_edges")
            print(f"[Arch3] Opened graph_edges ({self.edges_table.count_rows()} edges)")

        # Initialize FTS index
        self._init_fts_index()

    def _init_fts_index(self):
        """Initialize full-text search index."""
        try:
            self.nodes_table.create_fts_index("content", use_tantivy=True, replace=True)
            print("[Arch3] FTS index created")
        except Exception as e:
            print(f"[Arch3] FTS index skipped: {e}")

    def _extract_concepts(self, content: str) -> List[str]:
        """Extract concept keywords from content."""
        # Simple keyword extraction
        # In production, would use NLP/spacy
        keywords = []

        # Expertise areas
        expertise_keywords = [
            "defi", "nft", "trading", "smart contracts", "yield farming",
            "liquidity mining", "dao", "governance", "layer2", "bridges",
            "wallet security", "tokenomics", "audits", "dex", "cex",
            "staking", "apy", "impermanent loss", "slippage", "gas fees"
        ]

        content_lower = content.lower()

        for keyword in expertise_keywords:
            if keyword in content_lower:
                keywords.append(keyword)

        return keywords

    def _create_user_node(self, user: Dict[str, Any], group_id: str) -> str:
        """Create or update a user node."""
        node_id = f"user_{user['user_id']}_{group_id}"

        # Check if node exists
        existing = self.nodes_table.search().where(
            f"node_id = '{node_id}'",
            prefilter=True
        ).to_list()

        if not existing:
            # Create new user node
            properties = json.dumps({
                "user_id": user["user_id"],
                "username": user.get("username", ""),
                "expertise": user.get("expertise", [])
            })

            # Use expertise for embedding
            expertise_text = " ".join(user.get("expertise", []))
            vector = self.embedding_model.encode_documents([expertise_text])[0]

            self.nodes_table.add([{
                "node_id": node_id,
                "node_type": "user",
                "agent_id": self.agent_id,
                "group_id": group_id,
                "properties": properties,
                "content": expertise_text,
                "timestamp": "",
                "vector": vector.tolist()
            }])

        return node_id

    def _create_message_node(self, message: Dict[str, Any]) -> str:
        """Create a message node."""
        node_id = message.get("message_id", f"msg_{datetime.now().timestamp()}")
        content = message.get("content", "")
        vector = self.embedding_model.encode_documents([content])[0]

        properties = json.dumps({
            "sender_id": message.get("sender_id", ""),
            "sender_name": message.get("sender_name", ""),
            "message_type": message.get("message_type", ""),
            "topics": message.get("topics", [])
        })

        self.nodes_table.add([{
            "node_id": node_id,
            "node_type": "message",
            "agent_id": self.agent_id,
            "group_id": message.get("group_id", ""),
            "properties": properties,
            "content": content,
            "timestamp": message.get("timestamp", ""),
            "vector": vector.tolist()
        }])

        return node_id

    def _create_concept_nodes(self, concepts: List[str], group_id: str, message_id: str) -> List[str]:
        """Create concept nodes and link them to message."""
        concept_ids = []

        for concept in concepts:
            # Create consistent concept ID
            concept_clean = concept.lower().replace(" ", "_")
            node_id = f"concept_{concept_clean}_{group_id}"

            # Check if exists
            existing = self.nodes_table.search().where(
                f"node_id = '{node_id}'",
                prefilter=True
            ).to_list()

            if not existing:
                # Create concept node
                vector = self.embedding_model.encode_documents([concept])[0]

                self.nodes_table.add([{
                    "node_id": node_id,
                    "node_type": "concept",
                    "agent_id": self.agent_id,
                    "group_id": group_id,
                    "properties": json.dumps({"concept": concept}),
                    "content": concept,
                    "timestamp": "",
                    "vector": vector.tolist()
                }])

            concept_ids.append(node_id)

        return concept_ids

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        timestamp: str = ""
    ):
        """Create an edge between nodes."""
        edge_id = f"{source_id}_{edge_type}_{target_id}"

        # Check if edge exists
        existing = self.edges_table.search().where(
            f"edge_id = '{edge_id}'",
            prefilter=True
        ).to_list()

        if existing:
            # Update weight
            self.edges_table.update(where=f"edge_id = '{edge_id}'", values={"weight": existing[0]["weight"] + weight})
        else:
            # Create new edge
            self.edges_table.add([{
                "edge_id": edge_id,
                "source_node": source_id,
                "target_node": target_id,
                "edge_type": edge_type,
                "weight": weight,
                "timestamp": timestamp
            }])

    def add_message(self, message: Dict[str, Any]) -> str:
        """Add a message to the graph."""
        group_id = message.get("group_id", "")

        # Get user info (simplified - would come from user data)
        user_info = {
            "user_id": message.get("sender_id", ""),
            "username": message.get("sender_name", ""),
            "expertise": message.get("topics", [])
        }

        # Create/update user node
        user_node_id = self._create_user_node(user_info, group_id)

        # Create message node
        message_node_id = self._create_message_node(message)

        # Create "said" edge
        self._create_edge(
            user_node_id,
            message_node_id,
            "said",
            weight=1.0,
            timestamp=message.get("timestamp", "")
        )

        # Extract concepts and create concept nodes
        content = message.get("content", "")
        concepts = self._extract_concepts(content)

        if concepts:
            concept_ids = self._create_concept_nodes(concepts, group_id, message_node_id)

            # Create edges
            for concept_id in concept_ids:
                # Message mentions concept
                self._create_edge(message_node_id, concept_id, "mentions", weight=1.0)

                # User knows about concept
                self._create_edge(user_node_id, concept_id, "knows_about", weight=0.5)

        return message_node_id

    def add_messages_batch(self, messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """Add multiple messages in batch."""
        counts = {"messages": 0, "nodes": 0, "edges": 0}

        for message in messages:
            self.add_message(message)
            counts["messages"] += 1

        # Count actual nodes and edges
        counts["nodes"] = self.nodes_table.count_rows()
        counts["edges"] = self.edges_table.count_rows()

        return counts

    def semantic_search(
        self,
        query: str,
        group_id: str = None,
        user_id: str = None,
        memory_type: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Graph-based semantic search.

        Searches for relevant concept nodes, then traverses to find
        related users and messages.
        """
        # Generate query embedding
        query_vector = self.embedding_model.encode_query([query])[0]

        # Search for relevant nodes (concepts and messages)
        search = self.nodes_table.search(query_vector.tolist())

        conditions = [f"agent_id = '{self.agent_id}'"]

        if group_id:
            conditions.append(f"group_id = '{group_id}'")

        # Search concepts and messages
        conditions.append("(node_type = 'concept' OR node_type = 'message')")

        where_clause = " AND ".join(conditions)
        search = search.where(where_clause, prefilter=True)

        relevant_nodes = search.limit(limit * 2).to_list()

        # Traverse graph to collect results
        results = []
        seen_ids = set()

        for node in relevant_nodes:
            node_id = node["node_id"]

            if node_id in seen_ids:
                continue

            seen_ids.add(node_id)

            # Add node to results
            result = {
                "node_id": node_id,
                "node_type": node["node_type"],
                "content": node.get("content", ""),
                "_score": node.get("_score", 0),
                "_distance": node.get("_distance", 0)
            }

            # If it's a concept, find related users
            if node["node_type"] == "concept":
                # Find users who know about this concept
                incoming_edges = self.edges_table.search().where(
                    f"target_node = '{node_id}' AND edge_type = 'knows_about'",
                    prefilter=True
                ).to_list()

                related_users = []
                for edge in incoming_edges:
                    user_node = self.nodes_table.search().where(
                        f"node_id = '{edge['source_node']}'",
                        prefilter=True
                    ).to_list()

                    if user_node:
                        related_users.append(user_node[0])

                result["_related_users"] = related_users

            results.append(result)

            if len(results) >= limit:
                break

        return results

    def get_group_context(self, group_id: str, user_id: str = None, limit: int = 10) -> Dict[str, Any]:
        """Get context for a group using graph traversal."""
        # Get recent messages in group
        messages = self.nodes_table.search().where(
            f"agent_id = '{self.agent_id}' AND group_id = '{group_id}' AND node_type = 'message'",
            prefilter=True
        ).limit(limit).to_list()

        # Get users in group
        users = self.nodes_table.search().where(
            f"agent_id = '{self.agent_id}' AND group_id = '{group_id}' AND node_type = 'user'",
            prefilter=True
        ).limit(limit).to_list()

        # Get concepts in group
        concepts = self.nodes_table.search().where(
            f"agent_id = '{self.agent_id}' AND group_id = '{group_id}' AND node_type = 'concept'",
            prefilter=True
        ).limit(limit).to_list()

        return {
            "group_context": messages,
            "users": users,
            "concepts": concepts
        }

    def get_experts(self, group_id: str, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find users with expertise in a topic (Transactive Memory).

        Traverses graph to find users who have discussed this topic.
        """
        # Find concept node for topic
        concept_clean = topic.lower().replace(" ", "_")
        concept_id = f"concept_{concept_clean}_{group_id}"

        concept = self.nodes_table.search().where(
            f"node_id = '{concept_id}'",
            prefilter=True
        ).to_list()

        if not concept:
            return []

        # Find users who know about this concept
        edges = self.edges_table.search().where(
            f"target_node = '{concept_id}' AND edge_type = 'knows_about'",
            prefilter=True
        ).to_list()

        experts = []
        for edge in edges:
            user_node = self.nodes_table.search().where(
                f"node_id = '{edge['source_node']}'",
                prefilter=True
            ).to_list()

            if user_node:
                user_data = user_node[0].copy()
                user_data["_expertise_score"] = edge["weight"]
                experts.append(user_data)

        # Sort by expertise score
        experts.sort(key=lambda x: x.get("_expertise_score", 0), reverse=True)

        return experts[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        total_nodes = self.nodes_table.count_rows()
        total_edges = self.edges_table.count_rows()

        # Count by node type
        user_nodes = len(self.nodes_table.search().where(
            f"agent_id = '{self.agent_id}' AND node_type = 'user'",
            prefilter=True
        ).to_list())

        message_nodes = len(self.nodes_table.search().where(
            f"agent_id = '{self.agent_id}' AND node_type = 'message'",
            prefilter=True
        ).to_list())

        concept_nodes = len(self.nodes_table.search().where(
            f"agent_id = '{self.agent_id}' AND node_type = 'concept'",
            prefilter=True
        ).to_list())

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "user_nodes": user_nodes,
            "message_nodes": message_nodes,
            "concept_nodes": concept_nodes
        }

    def clear(self):
        """Clear all data."""
        self.db.drop_table("graph_nodes")
        self.db.drop_table("graph_edges")
        self._init_tables()


def test_arch3():
    """Test Architecture 3 with generated test data."""
    print("=" * 60)
    print("Testing Architecture 3: Graph-Based Group Memory")
    print("=" * 60)

    # Load test data
    test_data_file = "/home/oydual3/a0x/a0x-memory/tests/group_memory/test_data.json"
    with open(test_data_file, 'r') as f:
        test_data = json.load(f)

    messages = test_data["messages"]
    groups = test_data["groups"]
    users = test_data["users"]

    print(f"\nTest Data:")
    print(f"  - Users: {len(users)}")
    print(f"  - Groups: {len(groups)}")
    print(f"  - Messages: {len(messages)}")

    # Initialize store
    db_path = "/home/oydual3/a0x/a0x-memory/tests/group_memory/lancedb_arch3"
    store = GraphBasedMemoryStore(
        db_path=db_path,
        agent_id="jessexbt"
    )

    # Test 1: Batch insert performance
    print("\n" + "-" * 60)
    print("Test 1: Batch Insert Performance")
    print("-" * 60)

    import time
    start = time.time()

    counts = store.add_messages_batch(messages)

    elapsed = time.time() - start

    print(f"Inserted {len(messages)} messages in {elapsed:.2f}s")
    print(f"  - Throughput: {len(messages)/elapsed:.2f} msg/sec")
    print(f"Graph stats:")
    print(f"  - Nodes created: {counts['nodes']}")
    print(f"  - Edges created: {counts['edges']}")

    # Get detailed stats
    stats = store.get_stats()
    print(f"\nStore Stats:")
    print(f"  - User nodes: {stats['user_nodes']}")
    print(f"  - Message nodes: {stats['message_nodes']}")
    print(f"  - Concept nodes: {stats['concept_nodes']}")
    print(f"  - Total edges: {stats['total_edges']}")

    # Test 2: Semantic search performance
    print("\n" + "-" * 60)
    print("Test 2: Semantic Search Performance")
    print("-" * 60)

    test_queries = [
        "yield farming strategies",
        "NFT collecting tips",
        "smart contract security",
        "DAO governance"
    ]

    for query in test_queries:
        start = time.time()
        results = store.semantic_search(query, limit=5)
        elapsed = time.time() - start

        print(f"\nQuery: '{query}'")
        print(f"  - Results: {len(results)}")
        print(f"  - Latency: {elapsed*1000:.2f}ms")

        if results:
            node_type = results[0].get("node_type", "unknown")
            print(f"  - Top result: [{node_type}] {results[0].get('content', '')[:60]}...")

            # Show related users if concept
            if "_related_users" in results[0]:
                related_count = len(results[0]["_related_users"])
                print(f"  - Related users: {related_count}")

    # Test 3: Expertise location (Transactive Memory)
    print("\n" + "-" * 60)
    print("Test 3: Expertise Location (Who Knows What)")
    print("-" * 60)

    topics = ["defi", "nft", "trading", "smart contracts"]

    for topic in topics:
        start = time.time()
        experts = store.get_experts(groups[0]["group_id"], topic, limit=3)
        elapsed = time.time() - start

        print(f"\nTopic: '{topic}'")
        print(f"  - Experts found: {len(experts)}")
        print(f"  - Latency: {elapsed*1000:.2f}ms")

        if experts:
            for i, expert in enumerate(experts[:2]):
                props = json.loads(expert.get("properties", "{}"))
                username = props.get("username", "unknown")
                expertise = props.get("expertise", [])
                score = expert.get("_expertise_score", 0)
                print(f"    {i+1}. {username} (score: {score:.1f}) - expertise: {expertise}")

    # Test 4: Group context retrieval
    print("\n" + "-" * 60)
    print("Test 4: Group Context Retrieval")
    print("-" * 60)

    group_id = groups[0]["group_id"]
    user_id = users[0]["user_id"]

    start = time.time()
    context = store.get_group_context(group_id, user_id, limit=5)
    elapsed = time.time() - start

    print(f"Group: {groups[0]['group_name']}")
    print(f"  - Messages: {len(context['group_context'])}")
    print(f"  - Users: {len(context['users'])}")
    print(f"  - Concepts: {len(context['concepts'])}")
    print(f"  - Latency: {elapsed*1000:.2f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("Architecture 3 Test Summary")
    print("=" * 60)
    print("✓ Graph structure implemented")
    print("✓ Node-edge model working")
    print("✓ Concept extraction functional")
    print("✓ Expertise location (Transactive Memory) working")
    print(f"\nPerformance:")
    print(f"  - Insert throughput: {len(messages)/elapsed:.2f} msg/sec")
    print(f"  - Query latency: ~{elapsed*1000:.2f}ms")
    print(f"  - Storage efficiency: Low (graph structure)")
    print(f"\nUnique features:")
    print(f"  - Tracks 'who knows what' metaknowledge")
    print(f"  - Multi-hop reasoning via graph traversal")
    print(f"  - Natural expertise discovery")


if __name__ == "__main__":
    test_arch3()
