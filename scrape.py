#!/usr/bin/env python3
"""
US industry ML researcher discovery pipeline.

Two strategies, both via OpenAlex:
1. Conference authors — pull all authors from top ML/RL venues (NeurIPS, ICML, ICLR, etc.)
2. Lab authors — pull all authors affiliated with known AI research labs

Output is researcher-centric (one row per person), ranked by h-index.
"""

import argparse
import csv
import logging
import math
import re
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "researchers.csv"

OPENALEX_API = "https://api.openalex.org"
OPENALEX_EMAIL = "scraper@fleet.ai"

# --- Top ML/RL conferences (OpenAlex source IDs) ---
CONFERENCES = {
    "S4306420609": "NeurIPS",
    "S4306419644": "ICML",
    "S4306419637": "ICLR",
    "S4210191458": "AAAI",
    "S4306419146": "AISTATS",
    "S4306506823": "CoRL",
    "S4306420803": "RSS",
    "S4306420508": "ACL",
    "S4306418267": "EMNLP",
}

# --- Known AI research labs in the US (OpenAlex institution IDs) ---
AI_LABS = {
    "I1291425158": "Google",
    "I4210114444": "Meta",
    "I4210161460": "OpenAI",
    "I4210127875": "NVIDIA",
    "I4210153776": "Apple",
    "I1311688040": "Amazon",
    "I4210156221": "Allen AI (AI2)",
    "I4391768151": "Toyota Research Institute",
    "I4210114115": "IBM Research (Watson)",
    "I4210085935": "IBM Research (Almaden)",
    "I4387154989": "Hugging Face",
}

# OpenAlex topic subfields — restrict lab papers to AI/ML/CV
AI_CV_SUBFIELDS = "subfields/1702|subfields/1707"

# Venue keywords to exclude (repositories, medical, non-CS journals)
EXCLUDED_VENUE_KEYWORDS = {
    "zenodo", "cancer", "medicine", "medical", "genetics", "genomics",
    "quantum information", "solid-state circuits", "social science",
    "diabetes", "clinical", "biomedical", "health informatics",
    "pharmaceutical", "epidemiol",
}

# Institution types to EXCLUDE
EXCLUDED_INSTITUTION_TYPES = {"education", "healthcare"}

# Regex to detect non-human "author" names
_BAD_AUTHOR_RE = re.compile(r"\(.*\)|^\d|^[A-Z]{2,}\d|GPT|Gemini|Claude|LLaMA|Llama|Mistral|Copilot", re.IGNORECASE)

FIELDNAMES = [
    "priority_score",
    "name",
    "openalex_id",
    "h_index",
    "i10_index",
    "cited_by_count",
    "works_count",
    "2yr_mean_citedness",
    "institution",
    "institution_type",
    "city",
    "paper_count_in_window",
    "top_paper_title",
    "top_paper_link",
    "venues",
    "linkedin_search_url",
    "google_scholar_url",
]


# ---------------------------------------------------------------------------
# OpenAlex helpers
# ---------------------------------------------------------------------------

def openalex_get(endpoint: str, params: dict) -> dict | None:
    """Single OpenAlex API call with retry on rate limit."""
    params["mailto"] = OPENALEX_EMAIL
    for _ in range(3):
        try:
            resp = requests.get(f"{OPENALEX_API}/{endpoint}", params=params, timeout=30)
            if resp.status_code == 429:
                log.warning("Rate limited, sleeping 5s")
                time.sleep(5)
                continue
            if resp.status_code != 200:
                log.warning(f"OpenAlex {resp.status_code}: {resp.text[:200]}")
                return None
            return resp.json()
        except Exception as e:
            log.warning(f"OpenAlex request failed: {e}")
            return None
    return None


def openalex_paginate(filter_str: str, select: str = "doi,title,authorships,publication_date,primary_location") -> list[dict]:
    """Paginate through OpenAlex works matching a filter."""
    all_results = []
    cursor = "*"

    while cursor:
        data = openalex_get("works", {
            "filter": filter_str,
            "select": select,
            "per_page": 200,
            "cursor": cursor,
        })
        if not data:
            break
        results = data.get("results", [])
        all_results.extend(results)
        cursor = data.get("meta", {}).get("next_cursor")
        if not results:
            break
        time.sleep(0.15)

    return all_results


def fetch_conference_papers(source_id: str, start_date: str, end_date: str) -> list[dict]:
    filt = f"primary_location.source.id:{source_id},type:article,from_publication_date:{start_date},to_publication_date:{end_date}"
    return openalex_paginate(filt)


def fetch_lab_papers(institution_id: str, start_date: str, end_date: str) -> list[dict]:
    filt = (
        f"authorships.institutions.id:{institution_id},"
        f"topics.subfield.id:{AI_CV_SUBFIELDS},"
        f"type:article,"
        f"from_publication_date:{start_date},to_publication_date:{end_date}"
    )
    return openalex_paginate(filt)


def fetch_author_profiles(author_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch author profiles from OpenAlex. Returns {author_id: profile}."""
    profiles = {}
    batch_size = 50  # OpenAlex pipe filter limit
    id_list = list(author_ids)

    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        filter_str = "id:" + "|".join(batch)
        data = openalex_get("authors", {
            "filter": filter_str,
            "select": "id,display_name,summary_stats,cited_by_count,works_count,last_known_institutions",
            "per_page": 50,
        })
        if data:
            for author in data.get("results", []):
                aid = author["id"].split("/")[-1]
                profiles[aid] = author
        time.sleep(0.15)
        if (i // batch_size) % 20 == 0 and i > 0:
            log.info(f"  Fetched {i}/{len(id_list)} author profiles")

    return profiles


# ---------------------------------------------------------------------------
# Author extraction (Phase 1: collect from works)
# ---------------------------------------------------------------------------

def collect_authors_from_works(works: list[dict], authors: dict, venue: str = ""):
    """Extract US non-academic authors from works into the authors dict.

    authors dict: {openalex_id: {name, institution, institution_type, city, papers: [...]}}
    """
    for work in works:
        title = (work.get("title") or "").replace("\n", " ").strip()
        if not title:
            continue
        doi = work.get("doi") or ""
        pub_date = work.get("publication_date") or ""

        work_venue = venue
        if not work_venue:
            loc = work.get("primary_location") or {}
            src = loc.get("source") or {}
            work_venue = src.get("display_name", "")

        if any(kw in work_venue.lower() for kw in EXCLUDED_VENUE_KEYWORDS):
            continue

        for auth in work.get("authorships", []):
            if "US" not in auth.get("countries", []):
                continue

            author_obj = auth.get("author") or {}
            author_id = (author_obj.get("id") or "").split("/")[-1]
            name = author_obj.get("display_name") or auth.get("raw_author_name", "")
            if not name or not author_id or _BAD_AUTHOR_RE.search(name):
                continue

            # Find US institution
            institution = ""
            institution_type = ""
            city = ""
            for inst in auth.get("institutions", []):
                if inst.get("country_code") == "US":
                    institution = inst.get("display_name", "")
                    institution_type = inst.get("type", "")
                    break

            if institution_type in EXCLUDED_INSTITUTION_TYPES:
                continue

            raw_strings = auth.get("raw_affiliation_strings", [])
            if raw_strings:
                parts = [p.strip() for p in raw_strings[0].split(",")]
                if len(parts) >= 3:
                    city = parts[-2]

            paper = {"title": title, "doi": doi, "date": pub_date, "venue": work_venue}

            if author_id not in authors:
                authors[author_id] = {
                    "name": name,
                    "institution": institution,
                    "institution_type": institution_type,
                    "city": city,
                    "papers": [paper],
                }
            else:
                # Update institution to most recent if this paper is newer
                existing = authors[author_id]
                existing["papers"].append(paper)
                if pub_date > (existing["papers"][0].get("date") or ""):
                    existing["institution"] = institution or existing["institution"]
                    existing["institution_type"] = institution_type or existing["institution_type"]
                    existing["city"] = city or existing["city"]


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

def save_data(rows: list[dict]):
    DATA_DIR.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape(start_date: datetime, end_date: datetime):
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # {openalex_author_id: {name, institution, ..., papers: [...]}}
    authors: dict[str, dict] = {}

    # Phase 1: Conference papers
    for source_id, conf_name in CONFERENCES.items():
        log.info(f"[Conference] {conf_name} ({source_id})")
        works = fetch_conference_papers(source_id, start_str, end_str)
        before = len(authors)
        collect_authors_from_works(works, authors, venue=conf_name)
        log.info(f"  {len(works)} papers -> {len(authors) - before} new researchers")

    # Phase 2: Known AI lab papers
    for inst_id, lab_name in AI_LABS.items():
        log.info(f"[Lab] {lab_name} ({inst_id})")
        works = fetch_lab_papers(inst_id, start_str, end_str)
        before = len(authors)
        collect_authors_from_works(works, authors)
        log.info(f"  {len(works)} papers -> {len(authors) - before} new researchers")

    log.info(f"Total unique researchers found: {len(authors)}")

    # Phase 3: Fetch author profiles for h-index, citation count
    log.info(f"Fetching author profiles for {len(authors)} researchers...")
    profiles = fetch_author_profiles(list(authors.keys()))
    log.info(f"  Got {len(profiles)} profiles")

    # Build final rows, sorted by h-index
    rows = []
    for author_id, info in authors.items():
        profile = profiles.get(author_id, {})
        stats = profile.get("summary_stats", {})

        # Sort papers by date descending
        papers = sorted(info["papers"], key=lambda p: p.get("date", ""), reverse=True)
        top_paper = papers[0] if papers else {}
        venues = sorted(set(p["venue"] for p in papers if p["venue"]))

        h = stats.get("h_index") or 0
        citedness_2yr = stats.get("2yr_mean_citedness") or 0
        n_papers = len(papers)

        # Priority score: career impact + recent activity + citation momentum
        # h-index: career impact (0-200 range)
        # paper_count * 10: recent productivity bonus
        # log(1 + 2yr_citedness) * 15: citation momentum (log-scaled to avoid viral papers dominating)
        score = round(h + n_papers * 10 + math.log1p(citedness_2yr) * 15, 1)

        rows.append({
            "priority_score": score,
            "name": info["name"],
            "openalex_id": author_id,
            "h_index": h,
            "i10_index": stats.get("i10_index", ""),
            "cited_by_count": profile.get("cited_by_count", ""),
            "works_count": profile.get("works_count", ""),
            "2yr_mean_citedness": round(citedness_2yr, 2) if citedness_2yr else "",
            "institution": info["institution"],
            "institution_type": info["institution_type"],
            "city": info["city"],
            "paper_count_in_window": n_papers,
            "top_paper_title": top_paper.get("title", ""),
            "top_paper_link": top_paper.get("doi", ""),
            "venues": "; ".join(venues),
            "linkedin_search_url": build_linkedin_search_url(info["name"]),
            "google_scholar_url": build_google_scholar_url(info["name"]),
        })

    rows.sort(key=lambda r: r["priority_score"], reverse=True)

    save_data(rows)
    log.info(f"Done. {len(rows)} researchers saved to {CSV_PATH}")
    if rows:
        log.info(f"Top 10 by priority score:")
        for r in rows[:10]:
            log.info(f"  score={r['priority_score']:>6}  h={r['h_index']:>3}  papers={r['paper_count_in_window']}  {r['name']:30s}  {r['institution']}")


def main():
    parser = argparse.ArgumentParser(description="Discover US-based industry ML researchers, ranked by impact")
    parser.add_argument("--days", type=int, default=2, help="Days to look back (default: 2)")
    parser.add_argument("--backfill-months", type=int, default=0, help="Backfill N months")
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
