# arxiv-profile-scrape

Nightly scraper for finding **US-based industry/lab researchers** publishing in:
- Reinforcement Learning (RL, RLHF, GRPO, DPO, PPO, policy optimization)
- Post-training alignment & optimization
- World models & model-based RL
- Environment simulation & sim-to-real

Built as a hiring pipeline using the [OpenAlex API](https://docs.openalex.org/).

## Filters

- **Country**: US-affiliated institutions only
- **Institution type**: Industry, national labs, research institutes, nonprofits (excludes universities)
- **Relevance**: Regex-based title matching for RL/post-training keywords (rejects off-topic papers from OpenAlex's fuzzy search)

## Output

`data/researchers.csv`

Columns: `name`, `paper_link`, `paper_title`, `published_date`, `institution`, `institution_type`, `city`, `linkedin_search_url`, `google_scholar_url`

## Usage

```bash
pip install -r requirements.txt

# Nightly run (last 2 days)
python scrape.py

# Backfill last 3 months
python scrape.py --backfill-months 3
```

## GitHub Actions

Runs daily at 6 AM UTC. Can also be triggered manually with an optional `backfill_months` input.
