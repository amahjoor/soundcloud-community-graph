# SoundCloud community graph

A Python pipeline that reads public SoundCloud profiles in a browser and plots the community. `main.py` crawls likes, reposts, comments, and follows from seed accounts, then reduces that interaction matrix to a 2D landscape and clusters it. `ego_network.py` does the same for one account and draws that account's neighborhood.

Scraped files stay in `data/`, which is gitignored. Nothing here calls the SoundCloud API. The browser default is Safari.

## Quick start

```bash
git clone https://github.com/amahjoor/soundcloud-community-graph.git
cd soundcloud-community-graph
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python main.py pipeline --browser chrome --seed_accounts flume --max_accounts 20
```

macOS, after `source .venv/bin/activate` and the same `pip install`:

```bash
python main.py pipeline --seed_accounts flume --max_accounts 20
```

Safari needs Develop > Allow Remote Automation. On Windows and Linux pass `--browser chrome`. `--chrome_path` sets the Chrome binary.

If you omit `--seed_accounts`, the scraper starts from diplo, flume, kaytranada, fourtet, and charlesthefirst. `--max_accounts` defaults to 100 and `--max_items` defaults to 500 per interaction type.

## Plot a saved crawl

```bash
python main.py visualize --dim_reduction umap --clustering hdbscan
```

This reads `data/seeds.csv`, `data/remaining_seeds.csv`, and `data/sparse_matrix.npz`. It writes into `data/results/`:

- `soundcloud_landscape_<timestamp>.png`
- `soundcloud_landscape_interactive_<timestamp>.html` unless you pass `--no_interactive`
- `processed_accounts_<timestamp>.csv`

`--dim_reduction` is `umap` (default), `tsne`, or `svd`. `--clustering` is `hdbscan` (default), `dbscan`, or `kmeans`. Accounts need at least `--min_interactions` (default 10). `--max_dominance` (default 0.4) drops an account when too large a share of its interactions point at one other account.

Resume a crawl with `python main.py scrape --resume --browser chrome`.

## One account

```bash
python ego_network.py flume --browser chrome
```

That writes `data/ego_networks/flume/interactions.csv`, `account_info.csv`, `flume_network.png`, and `flume_network_interactive.html`. `--skip_scrape` reuses a folder that is already there. `--min_weight` defaults to 1. Likes are weighted 1 and other interaction types are weighted 3.

## Requirements

From `requirements.txt`:

- selenium and webdriver-manager
- numpy, pandas, scipy, scikit-learn
- hdbscan and umap-learn
- matplotlib, seaborn, plotly, networkx

Python 3.9 or newer. Chrome or Safari has to be installed because the scraper drives a real browser. SoundCloud can rate-limit or interrupt a long crawl. Progress is saved, and `--resume` continues it. Do not commit the `data/` directory.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
