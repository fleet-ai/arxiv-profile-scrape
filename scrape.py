#!/usr/bin/env python3
"""
US researcher scraper for RL, post-training, world models, and environment simulation.

Uses OpenAlex API to find US-affiliated researchers publishing in these areas,
with structured institution/country/city data. Outputs a single CSV for hiring.
"""

import argparse
import csv
import logging
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "researchers.csv"

OPENALEX_QUERIES = [
    "reinforcement learning RLHF language model",
    "reinforcement learning human feedback LLM",
    "GRPO group relative policy optimization",
    "reward model language model alignment",
    "post-training alignment large language model",
    "world model deep reinforcement learning",
    "sim-to-real transfer robot reinforcement learning",
    "model-based reinforcement learning deep learning",
    "DPO direct preference optimization language model",
    "policy optimization reinforcement learning neural network",
]

OPENALEX_API = "https://api.openalex.org"
OPENALEX_EMAIL = "scraper@fleet.ai"

# Paper title must contain at least one of these to be included (case-insensitive).
# Filters out off-topic results from OpenAlex's fuzzy text search.
RELEVANCE_KEYWORDS = [
    "reinforcement learning", "rl ", " rl,", "reward model", "reward shaping",
    "rlhf", "grpo", "dpo", "ppo", "policy optimization", "policy gradient",
    "preference optimization", "preference learning", "preference alignment",
    "post-training", "alignment", "fine-tuning", "finetuning",
    "world model", "model-based rl", "model based rl",
    "sim-to-real", "sim2real", "environment simulation",
    "large language model", "llm", "language model",
    "deep learning", "neural network", "transformer",
    "machine learning", "multi-agent", "multi agent",
    "robot learning", "robot control", "locomotion",
    "imitation learning", "inverse reinforcement",
    "offline reinforcement", "online reinforcement",
    "actor-critic", "q-learning", "value function",
    "curriculum learning", "exploration",
]

FIELDNAMES = [
    "name",
    "paper_link",
    "paper_title",
    "published_date",
    "institution",
    "institution_type",
    "city",
    "linkedin_search_url",
    "google_scholar_url",
]


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

def openalex_search(query: str, start_date: str, end_date: str) -> list[dict]:
    """Search OpenAlex for US-affiliated works matching query. Paginates through all results."""
    all_results = []
    cursor = "*"
    filter_str = f"from_publication_date:{start_date},to_publication_date:{end_date},institutions.country_code:US"

    while cursor:
        try:
            resp = requests.get(
                f"{OPENALEX_API}/works",
                params={
                    "search": query,
                    "filter": filter_str,
                    "select": "doi,title,authorships,publication_date",
                    "per_page": 200,
                    "cursor": cursor,
                    "mailto": OPENALEX_EMAIL,
                },
                timeout=30,
            )
            if resp.status_code == 429:
                log.warning("OpenAlex rate limited, sleeping 5s")
                time.sleep(5)
                continue
            if resp.status_code != 200:
                log.warning(f"OpenAlex returned {resp.status_code}")
                break
            data = resp.json()
            results = data.get("results", [])
            all_results.extend(results)
            cursor = data.get("meta", {}).get("next_cursor")
            if not results:
                break
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"OpenAlex search failed: {e}")
            break

    return all_results


def is_relevant(title: str) -> bool:
    """Check if a paper title contains at least one ML/RL keyword."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in RELEVANCE_KEYWORDS)


def extract_us_authors(works: list[dict], seen: set[str]) -> list[dict]:
    """Extract US-affiliated author rows from relevant OpenAlex works."""
    rows = []
    for work in works:
        title = (work.get("title") or "").replace("\n", " ").strip()
        if not is_relevant(title):
            continue
        doi = work.get("doi") or ""
        pub_date = work.get("publication_date") or ""

        for auth in work.get("authorships", []):
            if "US" not in auth.get("countries", []):
                continue

            name = auth.get("author", {}).get("display_name") or auth.get("raw_author_name", "")
            if not name:
                continue

            key = (name, title)
            if key in seen:
                continue
            seen.add(key)

            institution = ""
            institution_type = ""
            city = ""
            for inst in auth.get("institutions", []):
                if inst.get("country_code") == "US":
                    institution = inst.get("display_name", "")
                    institution_type = inst.get("type", "")
                    break

            raw_strings = auth.get("raw_affiliation_strings", [])
            if raw_strings:
                parts = [p.strip() for p in raw_strings[0].split(",")]
                if len(parts) >= 3:
                    city = parts[-2]

            rows.append({
                "name": name,
                "paper_link": doi,
                "paper_title": title,
                "published_date": pub_date,
                "institution": institution,
                "institution_type": institution_type,
                "city": city,
                "linkedin_search_url": build_linkedin_search_url(name),
                "google_scholar_url": build_google_scholar_url(name),
            })
    return rows


# ---------------------------------------------------------------------------
# Search URL builders
# ---------------------------------------------------------------------------

def build_linkedin_search_url(name: str) -> str:
    q = urllib.parse.quote_plus(f"{name} reinforcement learning OR machine learning")
    return f"https://www.google.com/search?q=site%3Alinkedin.com%2Fin+{q}"


def build_google_scholar_url(name: str) -> str:
    q = urllib.parse.quote_plus(f'author:"{name}"')
    return f"https://scholar.google.com/scholar?q={q}"


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def load_existing_data() -> tuple[list[dict], set[str]]:
    rows = []
    seen = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                seen.add((row["name"], row["paper_title"]))
    return rows, seen


def save_data(rows: list[dict]):
    DATA_DIR.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        rows.sort(key=lambda r: (r.get("published_date", ""), r.get("name", "")), reverse=True)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape(start_date: datetime, end_date: datetime):
    existing_rows, seen = load_existing_data()
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    new_rows = []

    for query in OPENALEX_QUERIES:
        log.info(f"Searching US researchers: {query}")
        works = openalex_search(query, start_str, end_str)
        rows = extract_us_authors(works, seen)
        new_rows.extend(rows)
        log.info(f"  {len(works)} works -> {len(rows)} new US entries")

    all_rows = existing_rows + new_rows
    save_data(all_rows)

    unique_names = len(set(r["name"] for r in all_rows))
    log.info(f"Done. New entries: {len(new_rows)}. Total: {len(all_rows)}. Unique researchers: {unique_names}")


def main():
    parser = argparse.ArgumentParser(description="Scrape OpenAlex for US-based ML researchers")
    parser.add_argument("--days", type=int, default=2, help="Number of days to look back (default: 2 for nightly)")
    parser.add_argument("--backfill-months", type=int, default=0, help="Backfill N months of data")
    args = parser.parse_args()

    end_date = datetime.now(timezone.utc).replace(tzinfo=None)

    if args.backfill_months > 0:
        start_date = end_date - timedelta(days=args.backfill_months * 30)
        log.info(f"Backfilling {args.backfill_months} months: {start_date.date()} to {end_date.date()}")
    else:
        start_date = end_date - timedelta(days=args.days)
        log.info(f"Scraping last {args.days} day(s): {start_date.date()} to {end_date.date()}")

    scrape(start_date, end_date)


if __name__ == "__main__":
    main()
