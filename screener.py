#!/usr/bin/env python3
"""S&P 500 value screener.

Selects S&P 500 stocks with:
  * positive diluted EPS in every fiscal year used (the last 5, or at least 4 if
    Yahoo has fewer),
  * P/E < max_pe, where P/E = current price / average diluted EPS of those years,
  * current ratio (current assets / current liabilities) > min_current_ratio.

Data sources:
  * S&P 500 constituents (ticker, name, sector): Wikipedia.
  * Annual EPS, current price, latest balance sheet: Yahoo Finance via yfinance.
"""

from __future__ import annotations

import argparse
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import requests
import yfinance as yf

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Income-statement rows tried in order for annual EPS.
EPS_ROWS = ("Diluted EPS", "Basic EPS")
# The most recent fiscal year must have ended within this many days, so stale data drops out.
MAX_STALENESS_DAYS = 550


# --------------------------------------------------------------------------- #
# S&P 500 constituents
# --------------------------------------------------------------------------- #
def get_sp500_constituents() -> pd.DataFrame:
    """Return DataFrame with columns: ticker, yf_ticker, name, sector."""
    resp = requests.get(WIKI_SP500_URL, headers={"User-Agent": "Mozilla/5.0 (sp500-screener)"}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]
    df = pd.DataFrame(
        {
            "ticker": table["Symbol"].astype(str).str.strip(),
            "name": table["Security"],
            "sector": table["GICS Sector"],
        }
    )
    # Yahoo uses '-' for share classes (BRK.B -> BRK-B).
    df["yf_ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Yahoo Finance: prices, annual EPS, balance sheet
# --------------------------------------------------------------------------- #
def get_latest_prices(yf_tickers: list[str]) -> pd.Series:
    """Batch-download recent daily closes and return the latest close per ticker."""
    data = yf.download(yf_tickers, period="10d", interval="1d", auto_adjust=False,
                       progress=False, threads=True, group_by="column")
    close = data["Close"]
    if isinstance(close, pd.Series):  # single ticker
        close = close.to_frame(yf_tickers[0])
    return close.ffill().iloc[-1].dropna()


def get_annual_eps(ticker: yf.Ticker) -> pd.Series:
    """Return annual EPS indexed by fiscal-year end date, oldest first (empty if unavailable)."""
    stmt = ticker.income_stmt
    if stmt is None or stmt.empty:
        return pd.Series(dtype=float)
    for row in EPS_ROWS:
        if row in stmt.index:
            eps = pd.to_numeric(stmt.loc[row], errors="coerce").dropna()
            if not eps.empty:
                eps.index = [pd.Timestamp(d).date() for d in eps.index]
                return eps.sort_index()
    return pd.Series(dtype=float)


def get_current_ratio(ticker: yf.Ticker) -> tuple[float | None, str | None]:
    """Return (current ratio, balance-sheet date) from the latest quarterly, else annual, balance sheet."""
    for sheet in (ticker.quarterly_balance_sheet, ticker.balance_sheet):
        if sheet is None or sheet.empty:
            continue
        if "Current Assets" not in sheet.index or "Current Liabilities" not in sheet.index:
            continue
        for col in sheet.columns:  # newest first
            ca, cl = sheet.at["Current Assets", col], sheet.at["Current Liabilities", col]
            if pd.notna(ca) and pd.notna(cl) and cl > 0:
                return float(ca) / float(cl), pd.Timestamp(col).date().isoformat()
    return None, None


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
@dataclass
class Criteria:
    max_pe: float = 15.0
    min_current_ratio: float = 2.0
    years: int = 5      # average up to this many fiscal years
    min_years: int = 4  # skip stocks with fewer years available


def evaluate_earnings(eps_annual: pd.Series, price: float, criteria: Criteria,
                      today: date | None = None) -> dict:
    """Apply the earnings rules to one stock. Returns a dict with ``status`` and metrics."""
    today = today or date.today()
    if len(eps_annual) < criteria.min_years:
        return {"status": f"fewer than {criteria.min_years} years of EPS"}
    eps = eps_annual.iloc[-criteria.years:]
    if (today - eps.index[-1]).days > MAX_STALENESS_DAYS:
        return {"status": "EPS data is stale"}
    if (eps <= 0).any():
        return {"status": "negative/zero EPS in a year"}
    avg_eps = float(eps.mean())
    pe = price / avg_eps
    result = {
        "avg_eps": avg_eps,
        "pe": pe,
        "n_years": len(eps),
        "eps_years": f"FY{eps.index[0].year}-FY{eps.index[-1].year}",
    }
    result["status"] = "ok" if pe < criteria.max_pe else f"P/E >= {criteria.max_pe:g}"
    return result


def run_screen(criteria: Criteria, tickers: list[str] | None = None,
               workers: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    log("Loading S&P 500 constituents from Wikipedia...")
    universe = get_sp500_constituents()
    if tickers:
        wanted = {t.upper() for t in tickers}
        universe = universe[universe["ticker"].isin(wanted) | universe["yf_ticker"].isin(wanted)]
    log(f"  {len(universe)} companies")

    log("Downloading latest prices from Yahoo Finance...")
    prices = get_latest_prices(universe["yf_ticker"].tolist())

    log(f"Downloading annual EPS from Yahoo Finance ({len(universe)} stocks)...")
    eps_by_ticker: dict[str, pd.Series] = {}
    symbols = universe["yf_ticker"].tolist()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_safe, lambda s=s: get_annual_eps(yf.Ticker(s))): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            eps = fut.result()
            eps_by_ticker[futures[fut]] = eps if eps is not None else pd.Series(dtype=float)
            if i % 50 == 0 or i == len(symbols):
                log(f"  {i}/{len(symbols)}")

    log("Applying earnings and P/E rules...")
    statuses: dict[str, str] = {}
    candidates = []
    for row in universe.itertuples(index=False):
        price = prices.get(row.yf_ticker)
        if price is None or pd.isna(price):
            statuses[row.ticker] = "no price"
            continue
        eps_annual = eps_by_ticker.get(row.yf_ticker)
        if eps_annual is None or eps_annual.empty:
            statuses[row.ticker] = "no EPS data on Yahoo"
            continue
        res = evaluate_earnings(eps_annual, float(price), criteria)
        statuses[row.ticker] = res["status"]
        if res["status"] == "ok":
            candidates.append((row, float(price), res))

    log(f"Checking current ratio for {len(candidates)} stocks that pass P/E...")
    results = []
    for row, price, res in candidates:
        cr, bs_date = _safe(lambda: get_current_ratio(yf.Ticker(row.yf_ticker)), default=(None, None))
        if cr is None:
            statuses[row.ticker] = "no current ratio (e.g. bank/insurer)"
            continue
        if cr <= criteria.min_current_ratio:
            statuses[row.ticker] = f"current ratio <= {criteria.min_current_ratio:g}"
            continue
        statuses[row.ticker] = "selected"
        results.append({
            "Ticker": row.ticker,
            "Company": row.name,
            "Sector": row.sector,
            "Price": price,
            "Avg EPS": res["avg_eps"],
            "P/E": res["pe"],
            "Current Ratio": cr,
            "Yrs": res["n_years"],
            "EPS Years": res["eps_years"],
            "BS Date": bs_date,
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("P/E").reset_index(drop=True)
    return df, pd.Series(statuses, name="status")


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:  # yfinance raises a variety of errors for missing data
        return default


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def print_report(df: pd.DataFrame, statuses: pd.Series, criteria: Criteria) -> None:
    print()
    print(f"S&P 500 screen  |  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Rules: EPS > 0 in each of the last {criteria.years} fiscal years "
          f"(at least {criteria.min_years} required), "
          f"P/E (price / avg EPS) < {criteria.max_pe:g}, "
          f"current ratio > {criteria.min_current_ratio:g}")
    print()
    if df.empty:
        print("No stocks matched.")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 200,
                               "display.max_colwidth", 30, "display.float_format", "{:,.2f}".format):
            print(df.to_string(index=False))
    print()
    print(f"Selected {len(df)} of {len(statuses)} stocks. Breakdown:")
    for status, count in statuses.value_counts().items():
        print(f"  {count:4d}  {status}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-pe", type=float, default=15.0, help="maximum P/E (default 15)")
    p.add_argument("--min-current-ratio", type=float, default=2.0, help="minimum current ratio (default 2)")
    p.add_argument("--years", type=int, default=5, help="max fiscal years of EPS to average (default 5)")
    p.add_argument("--min-years", type=int, default=4, help="min fiscal years of EPS required (default 4)")
    p.add_argument("--tickers", nargs="+", help="only screen these tickers (for quick tests)")
    args = p.parse_args(argv)

    if not 1 <= args.min_years <= args.years:
        p.error("--min-years must be between 1 and --years")

    criteria = Criteria(max_pe=args.max_pe, min_current_ratio=args.min_current_ratio,
                        years=args.years, min_years=args.min_years)
    df, statuses = run_screen(criteria, args.tickers)
    print_report(df, statuses, criteria)
    return 0


if __name__ == "__main__":
    sys.exit(main())
