# US large-cap value screener (S&P 500, Nasdaq-100, Dow)

Screens the S&P 500, Nasdaq-100 and Dow Jones Industrial Average together. A stock that's in more than one index is screened once, and the `Index` column lists every index it belongs to. Use `--index` to screen only some of them.

It lists the stocks that meet all three rules:

1. **Profitable every year:** diluted EPS is above 0 in every fiscal year used.
2. **P/E < 15:** P/E = current share price / average annual diluted EPS.
3. **Current ratio > 2:** current assets / current liabilities, from the most recent quarterly balance sheet. If that's missing, the annual one is used.

**Earnings window:** the script averages up to the last 5 fiscal years. yfinance usually has only 4 annual income statements, so a stock is kept if it has at least 4 years, and skipped if it has fewer. The `Yrs` and `EPS Years` columns show which years were used for each stock.

## Operating-investment indicators (selected stocks only)

| # | Indicator | Formula |
|---|---|---|
| 1 | Available Shares | common shares + stock options + RSU/PSU + DSU + exchangeable shares |
| 2 | Paid for Entire Company | Available Shares x current price |
| 3 | **Paid for Operating Property** | Paid for Entire Company - current assets |
| 4 | Earnings Before Amortization | latest annual net income + latest annual D&A |
| 5 | Balance Earned on Operating Investment | Earnings Before Amortization - 5% of current assets |
| 6 | **% Earned Before Amortization** | Balance Earned / Paid for Operating Property x 100 |
| 7 | Estimated Life | net PP&E / latest annual D&A |
| 8 | Investor's Amortization | Paid for Operating Property / Estimated Life |
| 9 | Earned on Operating Investment After Amortization | Balance Earned - Investor's Amortization |
| 10 | **% Earned After Amortization** | Earned After Amortization / Paid for Operating Property x 100 |

The report shows the three bold values for each selected stock. Add `--details` to print every step.

Inputs:
- **Common shares, current assets, net PP&E:** the same latest balance sheet used for the current ratio. If that sheet lacks a row, the newest balance sheet that has it is used.
- **Net income:** `Net Income Common Stockholders`, or `Net Income` if that's missing, from the latest annual income statement.
- **D&A:** depreciation and amortization combined, from the latest annual cash-flow statement.
- **Stock options:** total options outstanding, from SEC XBRL (`...OptionsOutstandingNumber`).
- **RSU/PSU:** nonvested non-option awards, from SEC XBRL (`...EquityInstrumentsOtherThanOptionsNonvestedNumber`). This is one combined figure, because SEC's company-level data doesn't break RSUs and PSUs out separately.
- **DSUs and exchangeable shares:** there's no standard XBRL tag for these, so they're counted as 0. Unvested DSUs are usually already included in the RSU/PSU figure.

When an input is missing, the values that depend on it show `n/a`, and the `Notes` column says why. If current assets exceed what the whole company costs, Paid for Operating Property is negative. It's shown with a minus sign and used as-is in steps 6–10, so the percentages can be negative too. Only an exact zero gives `n/a`, because that would be a division by zero.

## Data sources

| Data | Source |
|---|---|
| Index members (ticker, name, sector) | Wikipedia (S&P 500, Nasdaq-100 and Dow pages). If one page fails to load, the run warns and continues with the others |
| SEC company ID (CIK) for stocks whose Wikipedia page doesn't list it | SEC `company_tickers.json` |
| Annual EPS (`Diluted EPS`, or `Basic EPS` if that's missing) | Yahoo Finance (`yfinance`, `Ticker.income_stmt`) |
| Latest price | Yahoo Finance (`yfinance`), one batch download |
| Balance sheet, cash flow | Yahoo Finance (`yfinance`), fetched only for stocks that pass the P/E rule |
| Options and RSU/PSU counts | SEC EDGAR XBRL `companyfacts`, fetched only for selected stocks |

No API key is needed, but SEC requires a User-Agent with your name and email.

## Usage

```bash
pip install -r requirements.txt
export SEC_USER_AGENT="Your Name you@example.com"
python screener.py
python screener.py --details                   # show every step of the indicators
python screener.py --index sp500 dow           # only some indexes (sp500, nasdaq100, dow)

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
