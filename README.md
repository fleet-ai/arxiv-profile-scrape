# arxiv-profile-scrape

Nightly scraper for finding **US-based researchers** publishing in:
- Reinforcement Learning (RL, RLHF, GRPO, DPO, policy optimization)
- Post-training alignment & optimization
- World models & model-based RL
- Environment simulation & sim-to-real

Built as a hiring pipeline using the [OpenAlex API](https://docs.openalex.org/) for structured institution/location data.

## Output

`data/researchers.csv` — US researchers only

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
