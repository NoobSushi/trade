# S&P 500 value screener

Lists S&P 500 stocks that meet all three rules:

1. **Profitable every year:** diluted EPS is above 0 in each of the last 5 fiscal years.
2. **P/E < 15:** P/E = current share price / average diluted EPS over those 5 years.
3. **Current ratio > 2:** current assets / current liabilities, from the most recent quarterly balance sheet. If that's missing, the annual one is used.

## Data sources

| Data | Source |
|---|---|
| S&P 500 members (ticker, name, sector, CIK) | Wikipedia |
| 5 years of annual diluted EPS | SEC EDGAR XBRL `companyfacts` API (10-K filings) |
| Latest price, stock-split history | Yahoo Finance (`yfinance`), batch download |
| Balance sheet for the current ratio | Yahoo Finance (`yfinance`), fetched only for stocks that pass the P/E rule |

The EPS history comes from SEC because yfinance usually returns only 4 annual statements. Older EPS values are adjusted for any stock splits that happened after they were filed.

## Usage

```bash
pip install -r requirements.txt

# SEC requires a User-Agent with your name and email
export SEC_USER_AGENT="Your Name you@example.com"
python screener.py

# Options
python screener.py --max-pe 12 --min-current-ratio 1.5 --years 5
python screener.py --tickers AAPL MSFT NUE      # quick test on a few stocks
```

A full run downloads one SEC file per company (rate-limited to 8 requests/s) and takes a few minutes. Progress goes to stderr. The results table and a breakdown of why each stock was excluded go to stdout.

## Notes

- Banks and insurers don't report current assets or current liabilities, so they're always excluded.
- A stock is skipped if its latest 10-K fiscal year ended more than about 18 months ago, or if it has fewer than 5 annual EPS values.

## Tests

```bash
pip install pytest
python -m pytest -q
```

The tests run offline and mock all network calls.
