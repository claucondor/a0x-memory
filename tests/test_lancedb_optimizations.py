"""
LanceDB Native Features Benchmark

Compares our current search approach vs LanceDB native features:
1. Individual search vs Batch search
2. Manual BM25+CC fusion vs Hybrid search (vector + FTS + RRFReranker)
3. .to_list() all columns vs .select() specific columns
4. L2 (default) vs Cosine distance
5. distance_range() filtering
6. Our HTTP reranker (a0x-models) vs LanceDB built-in RRFReranker

Run: USE_LOCAL_STORAGE=true python tests/test_lancedb_optimizations.py
"""
import sys
import os
import time
import numpy as np
from datetime import datetime, timezone

os.environ["USE_LOCAL_STORAGE"] = "true"

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import config
import lancedb
from lancedb.rerankers import RRFReranker
from utils.embedding import EmbeddingModel

# ============================================================
# Setup
# ============================================================

DB_PATH = "/tmp/lancedb_optimization_test"
AGENT_ID = "bench_test"

# Realistic memory content (mimics what our system stores)
MEMORIES = [
    # Smart contract / dev
    {"content": "Alice has been building smart contracts for 5 years", "speaker": "alice_dev", "group_id": "g1", "importance": 0.8},
    {"content": "Alice specializes in gas optimization and security audits", "speaker": "alice_dev", "group_id": "g1", "importance": 0.7},
    {"content": "Alice is working on a new AMM design with concentrated liquidity", "speaker": "alice_dev", "group_id": "g1", "importance": 0.9},
    {"content": "Security audit scheduled for February with Trail of Bits", "speaker": "alice_dev", "group_id": "g1", "importance": 0.8},
    {"content": "Alice has audited over 20 contracts professionally", "speaker": "alice_dev", "group_id": "g1", "importance": 0.7},
    # Trading
    {"content": "Bob trades mostly on Base DEX, profitable for 3 months", "speaker": "bob_trader", "group_id": "g2", "importance": 0.6},
    {"content": "Volume is up 300% this month on Base", "speaker": "bob_trader", "group_id": "g2", "importance": 0.7},
    {"content": "Bob uses mean reversion strategy on smaller caps", "speaker": "bob_trader", "group_id": "g2", "importance": 0.6},
    {"content": "Risk management is key, never risk more than 2% per trade", "speaker": "bob_trader", "group_id": "g2", "importance": 0.5},
    {"content": "WETH/USDC pool liquidity looks good for arb opportunities", "speaker": "bob_trader", "group_id": "g2", "importance": 0.6},
    # NFT
    {"content": "Carol creates generative art using p5.js and Processing", "speaker": "carol_artist", "group_id": "g3", "importance": 0.7},
    {"content": "Carol's collection has 10,000 unique pieces, mint price 0.01 ETH", "speaker": "carol_artist", "group_id": "g3", "importance": 0.8},
    {"content": "Carol is learning ERC-721 basics for her NFT collection", "speaker": "carol_artist", "group_id": "g3", "importance": 0.6},
    {"content": "Ivy managed 5 successful NFT launches totaling 50k mints", "speaker": "ivy_marketer", "group_id": "g3", "importance": 0.7},
    {"content": "Build community before launch, Discord is essential", "speaker": "ivy_marketer", "group_id": "g3", "importance": 0.5},
    # Security
    {"content": "Henry specializes in smart contract security audits", "speaker": "henry_auditor", "group_id": "g1", "importance": 0.8},
    {"content": "Henry is certified by Code4rena and Sherlock for audits", "speaker": "henry_auditor", "group_id": "g1", "importance": 0.7},
    {"content": "Henry uses Slither and Mythril for automated scanning", "speaker": "henry_auditor", "group_id": "g1", "importance": 0.6},
    {"content": "Henry offered to review the security of Alice's contract", "speaker": "henry_auditor", "group_id": "g1", "importance": 0.7},
    {"content": "Security should be priority before launch", "speaker": "henry_auditor", "group_id": "g1", "importance": 0.8},
    # Founder
    {"content": "David is building a decentralized identity solution on Base", "speaker": "david_founder", "group_id": "g1", "importance": 0.8},
    {"content": "David is looking for Base and Optimism grants", "speaker": "david_founder", "group_id": "g1", "importance": 0.7},
    {"content": "David needs beta testers for the Base mainnet launch", "speaker": "david_founder", "group_id": "g1", "importance": 0.6},
    # DeFi
    {"content": "Grace is a PM at a DeFi protocol expanding to Base", "speaker": "grace_pm", "group_id": "g2", "importance": 0.7},
    {"content": "Grace focuses on yield aggregation and integration partners", "speaker": "grace_pm", "group_id": "g2", "importance": 0.6},
    {"content": "Jack is a crypto fund manager with Goldman background", "speaker": "jack_whale", "group_id": "g2", "importance": 0.7},
    {"content": "Jack has been trading on Base since day 1", "speaker": "jack_whale", "group_id": "g2", "importance": 0.5},
    {"content": "Jack prefers OTC deals for large positions on Base", "speaker": "jack_whale", "group_id": "g2", "importance": 0.6},
    # Misc
    {"content": "Frank is a complete beginner who just learned about crypto", "speaker": "frank_newbie", "group_id": "g3", "importance": 0.4},
    {"content": "Emma published a paper on MEV protection strategies", "speaker": "emma_researcher", "group_id": "g1", "importance": 0.7},
]

# Test queries with expected results
QUERIES = [
    {
        "query": "Who knows about smart contract development?",
        "expected_speakers": ["alice_dev", "henry_auditor"],
        "type": "dm"
    },
    {
        "query": "What projects are being built on Base?",
        "expected_speakers": ["david_founder", "alice_dev", "grace_pm"],
        "type": "dm"
    },
    {
        "query": "Who is interested in NFTs?",
        "expected_speakers": ["carol_artist", "ivy_marketer"],
        "type": "dm"
    },
    {
        "query": "Who can help with security audits?",
        "expected_speakers": ["henry_auditor", "alice_dev"],
        "type": "group",
        "group_id": "g1"
    },
    {
        "query": "What's the trading activity like?",
        "expected_speakers": ["bob_trader", "jack_whale"],
        "type": "group",
        "group_id": "g2"
    },
    {
        "query": "Tell me about the NFT collection being launched",
        "expected_speakers": ["carol_artist", "ivy_marketer"],
        "type": "group",
        "group_id": "g3"
    },
]


def setup_database():
    """Create test database with real embeddings."""
    print("Setting up test database...")
    embedding_model = EmbeddingModel()

    # Clean
    import shutil
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    db = lancedb.connect(DB_PATH)

    # Compute embeddings for all memories
    contents = [m["content"] for m in MEMORIES]
    t0 = time.time()
    vectors = embedding_model.encode_documents(contents)
    embed_time = time.time() - t0
    print(f"  Embedded {len(contents)} documents in {embed_time:.2f}s")

    # Build table data
    data = []
    for i, (mem, vec) in enumerate(zip(MEMORIES, vectors)):
        data.append({
            "memory_id": f"mem_{i:04d}",
            "agent_id": AGENT_ID,
            "group_id": mem["group_id"],
            "content": mem["content"],
            "speaker": mem["speaker"],
            "importance_score": mem["importance"],
            "vector": vec.tolist(),
        })

    # Create table
    tbl = db.create_table("memories", data=data, mode="overwrite")
    print(f"  Created table with {tbl.count_rows()} rows")

    # Create FTS index for hybrid search
    tbl.create_fts_index("content", replace=True)
    print("  Created FTS index on 'content'")

    return db, tbl, embedding_model


def evaluate_results(results, expected_speakers, label=""):
    """Check if expected speakers are found in results."""
    found_speakers = set()
    for r in results:
        speaker = r.get("speaker", "") if isinstance(r, dict) else getattr(r, "speaker", "")
        found_speakers.add(speaker)

    expected = set(expected_speakers)
    hits = expected & found_speakers
    recall = len(hits) / len(expected) if expected else 0
    return recall, hits, found_speakers


def benchmark_individual_vs_batch(tbl, embedding_model, queries_data):
    """Test 1: Individual searches vs batch search."""
    print("\n" + "=" * 60)
    print("TEST 1: Individual Search vs Batch Search")
    print("=" * 60)

    query_texts = [q["query"] for q in queries_data]

    # Pre-compute embeddings (same for both)
    t0 = time.time()
    query_vectors = embedding_model.encode_query(query_texts)
    embed_time = time.time() - t0
    print(f"  Embedding {len(query_texts)} queries: {embed_time * 1000:.1f}ms")

    top_k = 10
    N_RUNS = 20

    # --- Individual searches ---
    t0 = time.time()
    individual_results = {}
    for _ in range(N_RUNS):
        for i, qv in enumerate(query_vectors):
            vec = qv.tolist() if hasattr(qv, 'tolist') else qv
            results = tbl.search(vec).where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(top_k).to_list()
            if _ == 0:
                individual_results[i] = results
    t_individual = time.time() - t0

    # --- Batch search ---
    vecs_list = [qv.tolist() if hasattr(qv, 'tolist') else qv for qv in query_vectors]
    t0 = time.time()
    batch_results_all = {}
    for _ in range(N_RUNS):
        batch_df = tbl.search(vecs_list).where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(top_k).to_pandas()
        if _ == 0:
            # Split by query_index
            for qi in range(len(query_texts)):
                subset = batch_df[batch_df["query_index"] == qi]
                batch_results_all[qi] = subset.to_dict("records")
    t_batch = time.time() - t0

    print(f"\n  Individual ({len(query_texts)} queries x {N_RUNS}): {t_individual:.3f}s  ({t_individual/N_RUNS*1000:.1f}ms/iter)")
    print(f"  Batch      ({len(query_texts)} queries x {N_RUNS}): {t_batch:.3f}s  ({t_batch/N_RUNS*1000:.1f}ms/iter)")
    print(f"  Speedup: {t_individual/t_batch:.2f}x")

    # Quality comparison
    print("\n  Quality (recall):")
    for i, q in enumerate(queries_data):
        r_ind, _, _ = evaluate_results(individual_results.get(i, []), q["expected_speakers"])
        r_bat, _, _ = evaluate_results(batch_results_all.get(i, []), q["expected_speakers"])
        match = "✓" if r_ind == r_bat else "✗"
        print(f"    [{match}] '{q['query'][:45]}...' individual={r_ind:.0%} batch={r_bat:.0%}")


def benchmark_select_columns(tbl, embedding_model, queries_data):
    """Test 2: All columns vs .select() specific columns."""
    print("\n" + "=" * 60)
    print("TEST 2: All Columns vs .select()")
    print("=" * 60)

    query_texts = [q["query"] for q in queries_data]
    query_vectors = embedding_model.encode_query(query_texts)
    vecs_list = [qv.tolist() if hasattr(qv, 'tolist') else qv for qv in query_vectors]

    top_k = 10
    N_RUNS = 50
    select_cols = ["memory_id", "content", "speaker", "importance_score", "group_id"]

    # All columns
    t0 = time.time()
    for _ in range(N_RUNS):
        tbl.search(vecs_list).where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(top_k).to_list()
    t_all = time.time() - t0

    # Selected columns
    t0 = time.time()
    for _ in range(N_RUNS):
        tbl.search(vecs_list).where(f"agent_id = '{AGENT_ID}'", prefilter=True).select(select_cols).limit(top_k).to_list()
    t_select = time.time() - t0

    print(f"\n  All columns     ({N_RUNS}x): {t_all:.3f}s  ({t_all/N_RUNS*1000:.1f}ms/iter)")
    print(f"  Select {len(select_cols)} cols ({N_RUNS}x): {t_select:.3f}s  ({t_select/N_RUNS*1000:.1f}ms/iter)")
    print(f"  Speedup: {t_all/t_select:.2f}x")
    print(f"  (Vector column is 384 floats per row — skipping it saves I/O)")


def benchmark_distance_type(tbl, embedding_model, queries_data):
    """Test 3: L2 (default) vs Cosine distance."""
    print("\n" + "=" * 60)
    print("TEST 3: L2 (default) vs Cosine Distance")
    print("=" * 60)

    query_texts = [q["query"] for q in queries_data]
    query_vectors = embedding_model.encode_query(query_texts)

    top_k = 10
    N_RUNS = 20

    # L2 (default)
    vecs = [qv.tolist() if hasattr(qv, 'tolist') else qv for qv in query_vectors]
    t0 = time.time()
    l2_results = {}
    for _ in range(N_RUNS):
        for i, vec in enumerate(vecs):
            results = tbl.search(vec).where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(top_k).to_list()
            if _ == 0:
                l2_results[i] = results
    t_l2 = time.time() - t0

    # Cosine
    t0 = time.time()
    cos_results = {}
    for _ in range(N_RUNS):
        for i, vec in enumerate(vecs):
            results = tbl.search(vec).distance_type("cosine").where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(top_k).to_list()
            if _ == 0:
                cos_results[i] = results
    t_cos = time.time() - t0

    print(f"\n  L2 (default) ({N_RUNS}x): {t_l2:.3f}s  ({t_l2/N_RUNS*1000:.1f}ms/iter)")
    print(f"  Cosine       ({N_RUNS}x): {t_cos:.3f}s  ({t_cos/N_RUNS*1000:.1f}ms/iter)")
    print(f"  Speed diff: {t_l2/t_cos:.2f}x")

    # Compare result ranking quality
    print("\n  Ranking comparison (top-5 speakers):")
    for i, q in enumerate(queries_data):
        l2_speakers = [r["speaker"] for r in l2_results.get(i, [])[:5]]
        cos_speakers = [r["speaker"] for r in cos_results.get(i, [])[:5]]
        l2_dists = [f"{r['_distance']:.4f}" for r in l2_results.get(i, [])[:3]]
        cos_dists = [f"{r['_distance']:.4f}" for r in cos_results.get(i, [])[:3]]
        same = l2_speakers == cos_speakers
        r_l2, _, _ = evaluate_results(l2_results.get(i, [])[:5], q["expected_speakers"])
        r_cos, _, _ = evaluate_results(cos_results.get(i, [])[:5], q["expected_speakers"])
        print(f"    '{q['query'][:40]}...'")
        print(f"      L2:     {l2_speakers[:3]}  dists={l2_dists}  recall={r_l2:.0%}")
        print(f"      Cosine: {cos_speakers[:3]}  dists={cos_dists}  recall={r_cos:.0%}")
        print(f"      Same ranking: {'✓' if same else '✗'}")


def benchmark_distance_range(tbl, embedding_model, queries_data):
    """Test 4: distance_range() as a quality filter."""
    print("\n" + "=" * 60)
    print("TEST 4: distance_range() Filtering")
    print("=" * 60)

    query_texts = [q["query"] for q in queries_data]
    query_vectors = embedding_model.encode_query(query_texts)
    vecs = [qv.tolist() if hasattr(qv, 'tolist') else qv for qv in query_vectors]

    top_k = 25  # Get more to see filtering effect

    print("\n  Cosine distance_range thresholds (cosine distance = 1 - similarity):")
    print("  distance < 0.3 means similarity > 0.7 (high quality)")
    print("  distance < 0.5 means similarity > 0.5 (moderate quality)")
    print()

    for i, q in enumerate(queries_data):
        vec = vecs[i]
        # No filter
        results_all = tbl.search(vec).distance_type("cosine").where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(top_k).to_list()
        # distance < 0.5 (similarity > 0.5)
        try:
            results_05 = tbl.search(vec).distance_type("cosine").distance_range(upper_bound=0.5).where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(top_k).to_list()
        except Exception:
            results_05 = [r for r in results_all if r.get("_distance", 1) < 0.5]
        # distance < 0.3 (similarity > 0.7)
        try:
            results_03 = tbl.search(vec).distance_type("cosine").distance_range(upper_bound=0.3).where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(top_k).to_list()
        except Exception:
            results_03 = [r for r in results_all if r.get("_distance", 1) < 0.3]

        r_all, _, _ = evaluate_results(results_all[:10], q["expected_speakers"])
        r_05, _, _ = evaluate_results(results_05, q["expected_speakers"])
        r_03, _, _ = evaluate_results(results_03, q["expected_speakers"])

        dists_all = [r.get("_distance", 0) for r in results_all]
        print(f"  '{q['query'][:45]}...'")
        print(f"    No filter:      {len(results_all):2d} results, recall={r_all:.0%}, dist range=[{min(dists_all):.3f}, {max(dists_all):.3f}]")
        print(f"    dist < 0.5:     {len(results_05):2d} results, recall={r_05:.0%}")
        print(f"    dist < 0.3:     {len(results_03):2d} results, recall={r_03:.0%}")


def benchmark_hybrid_search(tbl, embedding_model, queries_data):
    """Test 5: Pure vector search vs Hybrid search (vector + FTS + RRF reranker)."""
    print("\n" + "=" * 60)
    print("TEST 5: Vector Search vs Hybrid Search (FTS + RRF)")
    print("=" * 60)

    reranker = RRFReranker()
    query_texts = [q["query"] for q in queries_data]
    query_vectors = embedding_model.encode_query(query_texts)
    vecs = [qv.tolist() if hasattr(qv, 'tolist') else qv for qv in query_vectors]

    top_k = 10
    N_RUNS = 20

    # Pure vector
    t0 = time.time()
    vec_results = {}
    for _ in range(N_RUNS):
        for i, vec in enumerate(vecs):
            results = tbl.search(vec).distance_type("cosine").where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(top_k).to_list()
            if _ == 0:
                vec_results[i] = results
    t_vec = time.time() - t0

    # Hybrid (vector + FTS + RRF)
    t0 = time.time()
    hybrid_results = {}
    for _ in range(N_RUNS):
        for i, (vec, qt) in enumerate(zip(vecs, query_texts)):
            try:
                results = (
                    tbl.search(query_type="hybrid")
                    .vector(vec)
                    .text(qt)
                    .where(f"agent_id = '{AGENT_ID}'", prefilter=True)
                    .rerank(reranker=reranker)
                    .limit(top_k)
                    .to_list()
                )
            except Exception as e:
                # Fallback if hybrid fails
                results = tbl.search(vec).distance_type("cosine").where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(top_k).to_list()
                if _ == 0:
                    print(f"    [hybrid fallback for query {i}: {e}]")
            if _ == 0:
                hybrid_results[i] = results
    t_hybrid = time.time() - t0

    print(f"\n  Vector only ({N_RUNS}x): {t_vec:.3f}s  ({t_vec/N_RUNS*1000:.1f}ms/iter)")
    print(f"  Hybrid+RRF  ({N_RUNS}x): {t_hybrid:.3f}s  ({t_hybrid/N_RUNS*1000:.1f}ms/iter)")
    print(f"  Speed diff: {t_vec/t_hybrid:.2f}x (>1 means vector is faster)")

    # Quality comparison
    print("\n  Quality comparison:")
    for i, q in enumerate(queries_data):
        r_vec, hits_v, found_v = evaluate_results(vec_results.get(i, [])[:5], q["expected_speakers"])
        r_hyb, hits_h, found_h = evaluate_results(hybrid_results.get(i, [])[:5], q["expected_speakers"])
        better = "hybrid" if r_hyb > r_vec else ("same" if r_hyb == r_vec else "vector")
        vec_top3 = [r["speaker"] for r in vec_results.get(i, [])[:3]]
        hyb_top3 = [r["speaker"] for r in hybrid_results.get(i, [])[:3]]
        print(f"    '{q['query'][:40]}...'")
        print(f"      Vector: recall={r_vec:.0%} top3={vec_top3}")
        print(f"      Hybrid: recall={r_hyb:.0%} top3={hyb_top3}")
        print(f"      Winner: {better}")


def benchmark_reranker_http_vs_none(tbl, embedding_model, queries_data):
    """Test 6: Compare our HTTP reranker (a0x-models) vs no reranker vs RRF."""
    print("\n" + "=" * 60)
    print("TEST 6: HTTP Reranker (a0x-models) vs RRF vs No Reranker")
    print("=" * 60)

    import requests
    a0x_models_url = os.getenv("A0X_MODELS_URL", "https://a0x-models-679925931457.us-central1.run.app")

    query_texts = [q["query"] for q in queries_data]
    query_vectors = embedding_model.encode_query(query_texts)
    vecs = [qv.tolist() if hasattr(qv, 'tolist') else qv for qv in query_vectors]

    top_k_search = 25
    top_k_final = 10
    reranker = RRFReranker()

    for i, q in enumerate(queries_data):
        vec = vecs[i]

        # Get raw results
        raw_results = tbl.search(vec).distance_type("cosine").where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(top_k_search).to_list()

        # A) No reranker (just vector order)
        no_rerank_results = raw_results[:top_k_final]

        # B) HTTP reranker (a0x-models cross-encoder)
        documents = [r["content"] for r in raw_results]
        t0 = time.time()
        try:
            resp = requests.post(
                f"{a0x_models_url}/rerank",
                json={"query": q["query"], "documents": documents, "top_k": top_k_final},
                timeout=30
            )
            resp.raise_for_status()
            rerank_data = resp.json().get("results", [])
            http_results = [raw_results[r["index"]] for r in rerank_data if r["index"] < len(raw_results)]
            t_http = time.time() - t0
        except Exception as e:
            http_results = no_rerank_results
            t_http = time.time() - t0
            print(f"    [HTTP rerank failed: {e}]")

        # C) RRF reranker via hybrid search
        t0 = time.time()
        try:
            rrf_results = (
                tbl.search(query_type="hybrid")
                .vector(vec)
                .text(q["query"])
                .where(f"agent_id = '{AGENT_ID}'", prefilter=True)
                .rerank(reranker=reranker)
                .limit(top_k_final)
                .to_list()
            )
        except Exception:
            rrf_results = no_rerank_results
        t_rrf = time.time() - t0

        r_none, _, _ = evaluate_results(no_rerank_results[:5], q["expected_speakers"])
        r_http, _, _ = evaluate_results(http_results[:5], q["expected_speakers"])
        r_rrf, _, _ = evaluate_results(rrf_results[:5], q["expected_speakers"])

        none_top3 = [r["speaker"] for r in no_rerank_results[:3]]
        http_top3 = [r["speaker"] for r in http_results[:3]]
        rrf_top3 = [r["speaker"] for r in rrf_results[:3]]

        print(f"\n  '{q['query'][:45]}...'")
        print(f"    No rerank:     recall={r_none:.0%} top3={none_top3}")
        print(f"    HTTP a0x:      recall={r_http:.0%} top3={http_top3}  ({t_http*1000:.0f}ms)")
        print(f"    RRF (local):   recall={r_rrf:.0%} top3={rrf_top3}  ({t_rrf*1000:.0f}ms)")


def benchmark_full_pipeline_comparison(tbl, embedding_model, queries_data):
    """Test 7: Full pipeline — Current approach vs Optimized approach."""
    print("\n" + "=" * 60)
    print("TEST 7: FULL PIPELINE — Current vs Optimized")
    print("=" * 60)
    print("  Current:   encode individually → individual searches → content dedup (re-embed) → HTTP rerank")
    print("  Optimized: batch encode → batch search + .select() + cosine → RRF hybrid → done")

    import requests
    a0x_models_url = os.getenv("A0X_MODELS_URL", "https://a0x-models-679925931457.us-central1.run.app")
    reranker = RRFReranker()

    query_texts = [q["query"] for q in queries_data]
    top_k = 10
    select_cols = ["memory_id", "content", "speaker", "importance_score", "group_id"]

    # ========== CURRENT APPROACH ==========
    print("\n  --- Current Approach ---")
    t_current_total = time.time()

    # Step 1: Encode queries individually
    t0 = time.time()
    individual_vectors = []
    for qt in query_texts:
        v = embedding_model.encode_single(qt, is_query=True)
        individual_vectors.append(v)
    t_encode_current = time.time() - t0

    # Step 2: Individual searches (simulate 3 query variants per original)
    t0 = time.time()
    current_raw = {}
    for i, vec in enumerate(individual_vectors):
        v = vec.tolist() if hasattr(vec, 'tolist') else vec
        results = tbl.search(v).where(f"agent_id = '{AGENT_ID}'", prefilter=True).limit(25).to_list()
        current_raw[i] = results
    t_search_current = time.time() - t0

    # Step 3: Content dedup (re-embed all results)
    t0 = time.time()
    current_deduped = {}
    for i, results in current_raw.items():
        if len(results) > 1:
            contents = [r["content"] for r in results]
            embeddings = embedding_model.encode_documents(contents)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / np.where(norms > 0, norms, 1)
            sim_matrix = np.dot(normalized, normalized.T)
            to_remove = set()
            for a in range(len(results)):
                if a in to_remove:
                    continue
                for b in range(a + 1, len(results)):
                    if b in to_remove:
                        continue
                    if sim_matrix[a, b] > 0.85:
                        to_remove.add(b)
            current_deduped[i] = [r for idx, r in enumerate(results) if idx not in to_remove]
        else:
            current_deduped[i] = results
    t_dedup_current = time.time() - t0

    # Step 4: HTTP rerank
    t0 = time.time()
    current_final = {}
    for i, results in current_deduped.items():
        documents = [r["content"] for r in results]
        try:
            resp = requests.post(
                f"{a0x_models_url}/rerank",
                json={"query": query_texts[i], "documents": documents, "top_k": top_k},
                timeout=30
            )
            resp.raise_for_status()
            rerank_data = resp.json().get("results", [])
            current_final[i] = [results[r["index"]] for r in rerank_data if r["index"] < len(results)]
        except Exception:
            current_final[i] = results[:top_k]
    t_rerank_current = time.time() - t0

    t_current_total = time.time() - t_current_total

    # ========== OPTIMIZED APPROACH ==========
    print("  --- Optimized Approach ---")
    t_optimized_total = time.time()

    # Step 1: Batch encode
    t0 = time.time()
    batch_vectors = embedding_model.encode_query(query_texts)
    t_encode_opt = time.time() - t0

    # Step 2: Batch search with .select() + cosine + hybrid + RRF
    t0 = time.time()
    opt_final = {}
    for i, (vec, qt) in enumerate(zip(batch_vectors, query_texts)):
        v = vec.tolist() if hasattr(vec, 'tolist') else vec
        try:
            results = (
                tbl.search(query_type="hybrid")
                .vector(v)
                .text(qt)
                .where(f"agent_id = '{AGENT_ID}'", prefilter=True)
                .rerank(reranker=reranker)
                .select(select_cols)
                .limit(top_k)
                .to_list()
            )
        except Exception:
            results = tbl.search(v).distance_type("cosine").where(f"agent_id = '{AGENT_ID}'", prefilter=True).select(select_cols).limit(top_k).to_list()
        opt_final[i] = results
    t_search_opt = time.time() - t0

    t_optimized_total = time.time() - t_optimized_total

    # ========== RESULTS ==========
    print(f"\n  {'Step':<25} {'Current':>10} {'Optimized':>10} {'Speedup':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Encode queries':<25} {t_encode_current*1000:>8.0f}ms {t_encode_opt*1000:>8.0f}ms {t_encode_current/max(t_encode_opt,0.001):>8.1f}x")
    print(f"  {'Search':<25} {t_search_current*1000:>8.0f}ms {t_search_opt*1000:>8.0f}ms {'—':>10}")
    print(f"  {'Content dedup (re-embed)':<25} {t_dedup_current*1000:>8.0f}ms {'0':>9}ms {'∞':>10}")
    print(f"  {'HTTP rerank':<25} {t_rerank_current*1000:>8.0f}ms {'0':>9}ms {'∞':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<25} {t_current_total*1000:>8.0f}ms {t_optimized_total*1000:>8.0f}ms {t_current_total/max(t_optimized_total,0.001):>8.1f}x")

    # Quality comparison
    print(f"\n  {'Query':<45} {'Current':>10} {'Optimized':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10}")
    for i, q in enumerate(queries_data):
        r_cur, _, _ = evaluate_results(current_final.get(i, [])[:5], q["expected_speakers"])
        r_opt, _, _ = evaluate_results(opt_final.get(i, [])[:5], q["expected_speakers"])
        print(f"  {q['query'][:45]:<45} {r_cur:>9.0%} {r_opt:>9.0%}")

    avg_cur = np.mean([evaluate_results(current_final.get(i, [])[:5], q["expected_speakers"])[0] for i, q in enumerate(queries_data)])
    avg_opt = np.mean([evaluate_results(opt_final.get(i, [])[:5], q["expected_speakers"])[0] for i, q in enumerate(queries_data)])
    print(f"  {'AVERAGE RECALL':<45} {avg_cur:>9.0%} {avg_opt:>9.0%}")


if __name__ == "__main__":
    print("=" * 60)
    print("LanceDB Native Features Benchmark")
    print("=" * 60)

    db, tbl, embedding_model = setup_database()

    benchmark_individual_vs_batch(tbl, embedding_model, QUERIES)
    benchmark_select_columns(tbl, embedding_model, QUERIES)
    benchmark_distance_type(tbl, embedding_model, QUERIES)
    benchmark_distance_range(tbl, embedding_model, QUERIES)
    benchmark_hybrid_search(tbl, embedding_model, QUERIES)
    benchmark_reranker_http_vs_none(tbl, embedding_model, QUERIES)
    benchmark_full_pipeline_comparison(tbl, embedding_model, QUERIES)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
