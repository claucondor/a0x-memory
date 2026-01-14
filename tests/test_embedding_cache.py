"""
Tests for embedding cache functionality.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import shutil


def test_cache_miss_then_hit():
    """Test that cache miss computes and stores, cache hit returns stored."""
    from utils.embedding import EmbeddingModel

    test_db = "./tests/test_cache_db"
    if os.path.exists(test_db):
        shutil.rmtree(test_db)

    model = EmbeddingModel(cache_db_path=test_db)

    text = "This is a test sentence for embedding cache"

    # First call - cache miss
    start = time.time()
    emb1 = model.encode_single(text, is_query=True)
    time1 = time.time() - start

    # Second call - cache hit
    start = time.time()
    emb2 = model.encode_single(text, is_query=True)
    time2 = time.time() - start

    # Verify same embedding
    assert np.allclose(emb1, emb2, rtol=1e-5), "Cached embedding should match original"

    # Cache hit should be faster (at least 2x)
    print(f"  First call (miss): {time1*1000:.2f}ms")
    print(f"  Second call (hit): {time2*1000:.2f}ms")
    print(f"  Speedup: {time1/time2:.1f}x")

    shutil.rmtree(test_db)
    return True


def test_query_vs_document_separate_cache():
    """Test that queries and documents have separate cache entries."""
    from utils.embedding import EmbeddingModel

    test_db = "./tests/test_cache_db2"
    if os.path.exists(test_db):
        shutil.rmtree(test_db)

    model = EmbeddingModel(cache_db_path=test_db)

    text = "Same text different purpose"

    emb_query = model.encode_single(text, is_query=True)
    emb_doc = model.encode_single(text, is_query=False)

    stats = model.get_cache_stats()
    assert stats["total_entries"] == 2, f"Should have 2 cache entries, got {stats['total_entries']}"

    print(f"  Cache entries: {stats['total_entries']}")

    shutil.rmtree(test_db)
    return True


def test_batch_encoding_with_cache():
    """Test batch encoding with partial cache hits."""
    from utils.embedding import EmbeddingModel

    test_db = "./tests/test_cache_db3"
    if os.path.exists(test_db):
        shutil.rmtree(test_db)

    model = EmbeddingModel(cache_db_path=test_db)

    texts = [
        "First sentence",
        "Second sentence",
        "Third sentence"
    ]

    # Cache first two
    model.encode([texts[0], texts[1]], is_query=False)

    # Now encode all three - should have 2 hits, 1 miss
    start = time.time()
    embeddings = model.encode(texts, is_query=False)
    elapsed = time.time() - start

    assert embeddings.shape[0] == 3, f"Should return 3 embeddings, got {embeddings.shape[0]}"

    stats = model.get_cache_stats()
    assert stats["total_entries"] == 3, f"Should have 3 cache entries, got {stats['total_entries']}"

    print(f"  Batch encode time: {elapsed*1000:.2f}ms")
    print(f"  Cache entries: {stats['total_entries']}")

    shutil.rmtree(test_db)
    return True


def test_cache_stats():
    """Test cache statistics reporting."""
    from utils.embedding import EmbeddingModel

    test_db = "./tests/test_cache_db4"
    if os.path.exists(test_db):
        shutil.rmtree(test_db)

    model = EmbeddingModel(cache_db_path=test_db)

    stats = model.get_cache_stats()
    assert stats["enabled"] is True, "Cache should be enabled"
    assert stats["total_entries"] == 0, "Should start with 0 entries"

    model.encode_single("test", is_query=True)

    stats = model.get_cache_stats()
    assert stats["total_entries"] == 1, "Should have 1 entry after encoding"

    print(f"  Stats: {stats}")

    shutil.rmtree(test_db)
    return True


def test_gcs_cache(bucket_path, service_account_path=None):
    """Test embedding cache with GCS backend."""
    from utils.embedding import EmbeddingModel

    print(f"\nTesting GCS cache at {bucket_path}")

    storage_options = None
    if service_account_path:
        storage_options = {"service_account": service_account_path}

    model = EmbeddingModel(
        cache_db_path=bucket_path,
        storage_options=storage_options
    )

    text = "Test sentence for GCS cache"

    # First call
    start = time.time()
    emb1 = model.encode_single(text, is_query=True)
    time1 = time.time() - start

    # Second call
    start = time.time()
    emb2 = model.encode_single(text, is_query=True)
    time2 = time.time() - start

    assert np.allclose(emb1, emb2, rtol=1e-5), "GCS cached embedding should match"

    print(f"  First call: {time1*1000:.2f}ms")
    print(f"  Second call: {time2*1000:.2f}ms")
    print(f"  Stats: {model.get_cache_stats()}")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs", help="GCS bucket path for cache (gs://bucket/path)")
    parser.add_argument("--sa", help="Service account JSON path")
    args = parser.parse_args()

    if args.gcs:
        print("\n" + "=" * 60)
        print("Embedding Cache GCS Test")
        print("=" * 60)
        try:
            test_gcs_cache(args.gcs, args.sa)
            print("\nGCS test PASSED")
            return True
        except Exception as e:
            print(f"\nGCS test FAILED: {e}")
            return False

    print("=" * 60)
    print("Embedding Cache Tests (Local)")
    print("=" * 60)

    tests = [
        ("Cache miss then hit", test_cache_miss_then_hit),
        ("Query vs document separate cache", test_query_vs_document_separate_cache),
        ("Batch encoding with cache", test_batch_encoding_with_cache),
        ("Cache statistics", test_cache_stats),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[TEST] {name}...")
        try:
            if test_func():
                print(f"  PASS")
                passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
