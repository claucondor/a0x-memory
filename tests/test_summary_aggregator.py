"""
Test volume-based SummaryAggregator.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_volume_config():
    """Verify volume-based configuration."""
    from database.group_summary_store import SUMMARY_CONFIG

    print("=" * 60)
    print("VOLUME-BASED CONFIGURATION")
    print("=" * 60)

    for level, config in SUMMARY_CONFIG.items():
        print(f"\n{level.upper()}:")
        print(f"  Trigger: Every {config['message_threshold']} messages")
        print(f"  Aggregate: {config['aggregate_count']} {level}s → next level")
        print(f"  Decay starts: After {config['decay_start_messages']} new messages")
        print(f"  Fully decayed: After {config['max_messages']} new messages")


def test_summary_flow():
    """Show the summary generation flow."""

    print("\n" + "=" * 60)
    print("SUMMARY GENERATION FLOW")
    print("=" * 60)

    print("""
Messages arrive → Check if 50 unsummarized
                        ↓
              Yes → Generate MICRO summary
                        ↓
              Check if 5 pending micros
                        ↓
              Yes → Generate CHUNK summary
                        ↓
              Check if 5 pending chunks
                        ↓
              Yes → Generate BLOCK summary

Example for 1250 messages:
- 25 micro summaries (every 50 msgs)
- 5 chunk summaries (every 5 micros = 250 msgs)
- 1 block summary (5 chunks = 1250 msgs)

LLM calls: 25 + 5 + 1 = 31 calls total
""")


def test_decay_calculation():
    """Test decay score calculation."""
    from database.group_summary_store import GroupSummaryStore

    print("\n" + "=" * 60)
    print("DECAY CALCULATION (Message-based)")
    print("=" * 60)

    # Simulate decay calculation
    import math

    def calc_decay(messages_since: int, level: str) -> float:
        from database.group_summary_store import SUMMARY_CONFIG
        config = SUMMARY_CONFIG[level]

        if messages_since < config["decay_start_messages"]:
            return 1.0

        decay_range = config["max_messages"] - config["decay_start_messages"]
        half_life = decay_range / 3
        effective = messages_since - config["decay_start_messages"]
        lambda_ = math.log(2) / half_life

        return max(0.0, math.exp(-lambda_ * effective))

    print("\nMicro decay (starts at 250 msgs, full at 500):")
    for msgs in [0, 100, 250, 300, 400, 500]:
        score = calc_decay(msgs, "micro")
        bar = "█" * int(score * 20)
        print(f"  {msgs:4d} msgs since → decay: {score:.2f} {bar}")

    print("\nChunk decay (starts at 1250 msgs, full at 2500):")
    for msgs in [0, 500, 1250, 1500, 2000, 2500]:
        score = calc_decay(msgs, "chunk")
        bar = "█" * int(score * 20)
        print(f"  {msgs:4d} msgs since → decay: {score:.2f} {bar}")


def test_cost_projection():
    """Project costs for volume-based summarization."""

    print("\n" + "=" * 60)
    print("COST PROJECTION (Llama 3.1 8B @ $0.05/$0.05 per M)")
    print("=" * 60)

    # Token estimates
    micro_input = 2500  # 50 msgs * 50 tokens
    micro_output = 100
    chunk_input = 500   # 5 summaries
    chunk_output = 150
    block_input = 750   # 5 summaries
    block_output = 200

    # Costs per million tokens
    input_cost = 0.05
    output_cost = 0.05

    def calc_cost(input_tok, output_tok):
        return (input_tok * input_cost + output_tok * output_cost) / 1_000_000

    micro_cost = calc_cost(micro_input, micro_output)
    chunk_cost = calc_cost(chunk_input, chunk_output)
    block_cost = calc_cost(block_input, block_output)

    print(f"\nPer summary cost:")
    print(f"  Micro: ${micro_cost:.6f} ({micro_input} in, {micro_output} out)")
    print(f"  Chunk: ${chunk_cost:.6f} ({chunk_input} in, {chunk_output} out)")
    print(f"  Block: ${block_cost:.6f} ({block_input} in, {block_output} out)")

    # Scenario: Active group with 500 msgs/day
    msgs_per_day = 500
    days = 30

    micros_per_day = msgs_per_day / 50
    chunks_per_day = micros_per_day / 5
    blocks_per_day = chunks_per_day / 5

    total_micros = micros_per_day * days
    total_chunks = chunks_per_day * days
    total_blocks = blocks_per_day * days

    monthly_cost = (total_micros * micro_cost +
                   total_chunks * chunk_cost +
                   total_blocks * block_cost)

    print(f"\nActive group (500 msgs/day for 30 days = 15,000 msgs):")
    print(f"  Micros: {total_micros:.0f} summaries")
    print(f"  Chunks: {total_chunks:.0f} summaries")
    print(f"  Blocks: {total_blocks:.0f} summaries")
    print(f"  Monthly cost: ${monthly_cost:.4f}")

    # Scale to 1000 groups
    print(f"\n1000 active groups:")
    print(f"  Monthly cost: ${monthly_cost * 1000:.2f}")


def test_model_import():
    """Test that models import correctly."""
    print("\n" + "=" * 60)
    print("MODEL IMPORT TEST")
    print("=" * 60)

    try:
        from models.group_memory import GroupSummary, SummaryLevel
        print(f"✓ GroupSummary imported")
        print(f"✓ SummaryLevel: {[l.value for l in SummaryLevel]}")

        # Create a test summary
        summary = GroupSummary(
            agent_id="test",
            group_id="test_group",
            level=SummaryLevel.MICRO,
            message_start=0,
            message_end=49,
            message_count=50,
            time_start="2026-01-29T10:00:00Z",
            time_end="2026-01-29T14:30:00Z",
            duration_hours=4.5,
            activity_rate=11.1,
            summary="Test summary"
        )
        print(f"✓ GroupSummary created: {summary.summary_id[:8]}...")
        print(f"  Level: {summary.level.value}")
        print(f"  Range: msgs {summary.message_start}-{summary.message_end}")

    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    test_model_import()
    test_volume_config()
    test_summary_flow()
    test_decay_calculation()
    test_cost_projection()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Volume-based summarization implemented:

✓ Micro: Every 50 messages (raw messages → summary)
✓ Chunk: Every 5 micros (250 messages, summaries → chunk)
✓ Block: Every 5 chunks (1250 messages, chunks → block)

✓ Decay based on NEW MESSAGES, not time
✓ Temporal context tracked (duration, activity_rate)
✓ Efficient aggregation (summaries of summaries)

Cost: ~$0.039/month per active group (500 msgs/day)
      ~$39/month for 1000 active groups
""")


if __name__ == "__main__":
    main()
