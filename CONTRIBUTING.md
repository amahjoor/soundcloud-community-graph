# Contributing

Bugs and pull requests are welcome.

## Run a small crawl before a scraper change

```bash
python main.py scrape --browser chrome --seed_accounts flume --max_accounts 2 --max_items 20
python ego_network.py flume --browser chrome --max_items 20
```

Use `--browser safari` on macOS if that is the browser you changed.

## What not to commit

- `data/`, including seeds, matrices, and ego-network CSV files
- a virtualenv, `__pycache__`, or `.DS_Store`
- cookies, session tokens, or a logged-in browser profile

## Pull requests

1. Say which command you ran and whether you used Chrome or Safari.
2. Keep new dependencies in `requirements.txt`.
3. This project is [MIT](LICENSE). Your contribution is licensed the same way.
