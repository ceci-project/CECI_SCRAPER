#!/usr/bin/env python3
"""
Fix truncated summaries caused by gemini-2.5-flash thinking mode consuming the token budget.

Finds decisions with summaries < 50 chars, re-runs AI on content already in DB,
and updates the records. Run this AFTER deploying the thinking_budget=0 fix.

Usage:
  python3 bin/fix_short_summaries.py --preview          # Show what would be fixed
  python3 bin/fix_short_summaries.py --run              # Actually fix them
  python3 bin/fix_short_summaries.py --run --limit 10   # Fix first 10 only
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gov_scraper.config import GOVERNMENT_NUMBER
from gov_scraper.db.connector import get_supabase_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_short_summary_decisions(min_content_length: int = 50, max_summary_length: int = 50, limit: int = 500):
    """Fetch decisions with short summaries that have usable content."""
    client = get_supabase_client()

    # Get decisions where summary is very short but content exists
    # Look at decisions from last 30 days to be safe
    cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()

    response = client.table('israeli_government_decisions').select(
        'id, decision_key, decision_date, decision_title, decision_content, summary, '
        'operativity, tags_policy_area, tags_government_body'
    ).gte('created_at', cutoff).order('created_at', desc=True).limit(limit * 3).execute()

    rows = response.data
    logger.info(f"Fetched {len(rows)} decisions from last 30 days")

    # Filter: short summary but has usable content
    bad = [
        r for r in rows
        if len(r.get('summary') or '') < max_summary_length
        and len(r.get('decision_content') or '') >= min_content_length
    ]

    logger.info(f"Found {len(bad)} decisions with short summaries (< {max_summary_length} chars)")
    return bad[:limit]


def reprocess_decision(row: dict) -> dict:
    """Re-run AI on a decision and return updated fields."""
    from gov_scraper.processors.ai import process_decision_with_ai

    logger.info(f"Re-processing {row['decision_key']} (content: {len(row.get('decision_content','') or '')} chars)")

    # process_decision_with_ai expects the full decision dict and mutates + returns it
    decision_data = {
        'decision_number': row['decision_key'].split('_', 1)[-1],
        'decision_title': row.get('decision_title', ''),
        'decision_content': row.get('decision_content', ''),
        'decision_date': row.get('decision_date', ''),
    }

    result = process_decision_with_ai(decision_data)

    return {
        'summary': result.get('summary', ''),
        'operativity': result.get('operativity', row.get('operativity', '')),
        'tags_policy_area': result.get('tags_policy_area', row.get('tags_policy_area', '')),
        'tags_government_body': result.get('tags_government_body', row.get('tags_government_body', '')),
        'tags_location': result.get('tags_location', ''),
        'all_tags': result.get('all_tags', ''),
    }


def update_decision(client, decision_id: int, updates: dict):
    """Update a decision record in Supabase."""
    client.table('israeli_government_decisions').update(updates).eq('id', decision_id).execute()


def main():
    parser = argparse.ArgumentParser(description='Fix decisions with truncated summaries')
    parser.add_argument('--preview', action='store_true', help='Show what would be fixed without making changes')
    parser.add_argument('--run', action='store_true', help='Actually run the fix')
    parser.add_argument('--limit', type=int, default=200, help='Max decisions to process (default: 200)')
    args = parser.parse_args()

    if not args.preview and not args.run:
        parser.print_help()
        sys.exit(1)

    decisions = fetch_short_summary_decisions(limit=args.limit)

    if not decisions:
        logger.info("No decisions with short summaries found — nothing to fix!")
        return

    print(f"\nFound {len(decisions)} decisions to fix:\n")
    for r in decisions[:10]:
        print(f"  {r['decision_key']} | {r['decision_date']} | summary: '{r.get('summary','')}' | content: {len(r.get('decision_content','') or '')} chars")
    if len(decisions) > 10:
        print(f"  ... and {len(decisions) - 10} more")

    if args.preview:
        print("\n[PREVIEW MODE] No changes made. Run with --run to apply fixes.")
        return

    print(f"\nProcessing {len(decisions)} decisions...\n")

    from gov_scraper.db.connector import get_supabase_client
    client = get_supabase_client()

    fixed = 0
    failed = 0

    for i, row in enumerate(decisions, 1):
        try:
            updates = reprocess_decision(row)
            new_summary = updates.get('summary', '')

            if len(new_summary) > len(row.get('summary') or ''):
                update_decision(client, row['id'], updates)
                logger.info(f"[{i}/{len(decisions)}] Fixed {row['decision_key']}: '{row.get('summary','')}' -> '{new_summary[:80]}'")
                fixed += 1
            else:
                logger.warning(f"[{i}/{len(decisions)}] Skipped {row['decision_key']}: new summary not better ('{new_summary}')")

        except Exception as e:
            logger.error(f"[{i}/{len(decisions)}] Failed {row['decision_key']}: {e}")
            failed += 1

    print(f"\nDone: {fixed} fixed, {failed} failed out of {len(decisions)} total.")


if __name__ == '__main__':
    main()
