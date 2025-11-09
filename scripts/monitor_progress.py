#!/usr/bin/env python
"""
Monitor progress of long-running pair extraction.

Usage:
    python monitor_progress.py --checkpoint-dir data/pairs/checkpoints_50k_filtered
"""

import argparse
import time
import os
from pathlib import Path
import subprocess


def count_lines_fast(filepath):
    """Fast line counting using wc."""
    try:
        result = subprocess.run(['wc', '-l', str(filepath)],
                              capture_output=True, text=True, check=True)
        return int(result.stdout.split()[0])
    except:
        return 0


def get_file_size_mb(filepath):
    """Get file size in MB."""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except:
        return 0


def monitor_checkpoint(checkpoint_dir, refresh_interval=5):
    """
    Monitor checkpoint file and show progress.

    Args:
        checkpoint_dir: Directory containing pairs_checkpoint.csv
        refresh_interval: Seconds between updates
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_file = checkpoint_path / "pairs_checkpoint.csv"
    fragments_cache = checkpoint_path / "fragments_cache.pkl"

    print("=" * 70)
    print(" MONITORING PAIR EXTRACTION PROGRESS")
    print("=" * 70)
    print(f" Checkpoint dir: {checkpoint_dir}")
    print(f" Checkpoint file: {checkpoint_file}")
    print(f" Refresh interval: {refresh_interval}s")
    print("=" * 70)
    print()

    # Check if fragments are cached
    if fragments_cache.exists():
        frag_size = get_file_size_mb(fragments_cache)
        print(f"✓ Fragments cached: {frag_size:.1f} MB")
    else:
        print("⏳ Waiting for fragmentation to complete...")

    print()
    print("Monitoring checkpoint file (Ctrl+C to stop)...")
    print()

    last_count = 0
    last_time = time.time()

    try:
        while True:
            if checkpoint_file.exists():
                current_count = count_lines_fast(checkpoint_file) - 1  # Subtract header
                current_size = get_file_size_mb(checkpoint_file)
                current_time = time.time()

                # Calculate rate
                time_delta = current_time - last_time
                count_delta = current_count - last_count

                if time_delta > 0:
                    pairs_per_sec = count_delta / time_delta
                else:
                    pairs_per_sec = 0

                # Update display
                print(f"\r  Pairs: {current_count:,} | Size: {current_size:.1f} MB | "
                      f"Rate: {pairs_per_sec:.1f} pairs/s", end='', flush=True)

                last_count = current_count
                last_time = current_time
            else:
                print(f"\r  ⏳ Waiting for checkpoint file to appear...", end='', flush=True)

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print(" MONITORING STOPPED")
        print("=" * 70)

        if checkpoint_file.exists():
            final_count = count_lines_fast(checkpoint_file) - 1
            final_size = get_file_size_mb(checkpoint_file)
            print(f" Final count: {final_count:,} pairs")
            print(f" Final size: {final_size:.1f} MB")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Monitor pair extraction progress")
    parser.add_argument('--checkpoint-dir', required=True,
                       help='Checkpoint directory to monitor')
    parser.add_argument('--refresh-interval', type=int, default=5,
                       help='Seconds between updates (default: 5)')

    args = parser.parse_args()

    monitor_checkpoint(args.checkpoint_dir, args.refresh_interval)


if __name__ == '__main__':
    main()
