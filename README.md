# S&P 500 value screener

Lists S&P 500 stocks that meet all three rules:

1. **Profitable every year:** diluted EPS is above 0 in every fiscal year used.
2. **P/E < 15:** P/E = current share price / average annual diluted EPS.
3. **Current ratio > 2:** current assets / current liabilities, from the most recent quarterly balance sheet. If that's missing, the annual one is used.

**Earnings window:** the script averages up to the last 5 fiscal years. yfinance usually has only 4 annual income statements, so a stock is kept if it has at least 4 years, and skipped if it has fewer. The `Yrs` and `EPS Years` columns show which years were used for each stock.

## Data sources

| Data | Source |
|---|---|
| S&P 500 members (ticker, name, sector) | Wikipedia |
| Annual EPS (`Diluted EPS`, or `Basic EPS` if that's missing) | Yahoo Finance (`yfinance`, `Ticker.income_stmt`) |
| Latest price | Yahoo Finance (`yfinance`), one batch download |
| Balance sheet for the current ratio | Yahoo Finance (`yfinance`), fetched only for stocks that pass the P/E rule |

No API key is needed.

## Usage

```bash
pip install -r requirements.txt
python screener.py

# Options
python screener.py --max-pe 12 --min-current-ratio 1.5
python screener.py --years 5 --min-years 5    # require a full 5 years
python screener.py --tickers AAPL MSFT NUE     # quick test on a few stocks
```

A full run makes about 500 yfinance requests for income statements, using 4 parallel workers. Progress goes to stderr. The results table and a breakdown of why each stock was excluded go to stdout.

## Notes

- Banks and insurers don't report current assets or current liabilities, so they're always excluded.
- A stock is skipped if its latest fiscal year in Yahoo's data ended more than about 18 months ago.
- Yahoo's historical EPS is normally already adjusted for stock splits, so the script doesn't adjust it again.
- yfinance is an unofficial API and can be rate-limited. A stock whose data can't be fetched appears in the breakdown as "no EPS data on Yahoo" or "no price".

## Tests

```bash
pip install pytest
python -m pytest -q
```

The tests run offline and mock all network calls.
