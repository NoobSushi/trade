#!/usr/bin/env python3
"""S&P 500 value screener.

Selects S&P 500 stocks with:
  * positive diluted EPS in every one of the last N fiscal years (default 5),
  * P/E < max_pe, where P/E = current price / average diluted EPS of those N years,
  * current ratio (current assets / current liabilities) > min_current_ratio.

Data sources:
  * S&P 500 constituents (ticker, name, sector, CIK): Wikipedia.
  * Annual diluted EPS history: SEC EDGAR XBRL "companyfacts" API (10-K filings).
  * Current price, stock-split history, latest balance sheet: Yahoo Finance via yfinance.

SEC requires a User-Agent that identifies you, e.g. "Jane Doe jane@example.com".
Pass it with --user-agent or the SEC_USER_AGENT environment variable.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import requests
import yfinance as yf

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# XBRL concepts tried in order for annual EPS.
EPS_CONCEPTS = ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic")
ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A"}
# A fiscal year is 52/53 weeks; allow some slack for odd calendars.
MIN_FY_DAYS, MAX_FY_DAYS = 340, 390
# The most recent fiscal year must have ended within this many days, so stale filers drop out.
MAX_STALENESS_DAYS = 550


# --------------------------------------------------------------------------- #
# S&P 500 constituents
# --------------------------------------------------------------------------- #
def get_sp500_constituents() -> pd.DataFrame:
    """Return DataFrame with columns: ticker, yf_ticker, name, sector, cik."""
    resp = requests.get(WIKI_SP500_URL, headers={"User-Agent": "Mozilla/5.0 (sp500-screener)"}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]
    df = pd.DataFrame(
        {
            "ticker": table["Symbol"].astype(str).str.strip(),
            "name": table["Security"],
            "sector": table["GICS Sector"],
            "cik": table["CIK"].astype(int),
        }
    )
    # Yahoo uses '-' for share classes (BRK.B -> BRK-B).
    df["yf_ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SEC EDGAR: annual EPS history
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Thread-safe limiter; SEC allows at most 10 requests per second."""

    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                time.sleep(self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval


def fetch_company_facts(cik: int, session: requests.Session, limiter: RateLimiter, retries: int = 3) -> dict | None:
    url = SEC_FACTS_URL.format(cik=cik)
    for attempt in range(retries):
        limiter.wait()
        try:
            resp = session.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(2**attempt)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        time.sleep(2**attempt)  # 429 / 5xx: back off and retry
    return None


def extract_annual_eps(facts: dict) -> pd.DataFrame:
    """Extract one diluted-EPS value per fiscal year from 10-K filings.

    Returns DataFrame indexed by fiscal-year end date with columns ``eps`` and ``filed``,
    sorted ascending. When a year was reported in several filings (each 10-K repeats
    prior years), the most recently filed value is kept, which picks up restatements.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in EPS_CONCEPTS:
        units = us_gaap.get(concept, {}).get("units", {})
        rows = units.get("USD/shares")
        if not rows:
            continue
        records = []
        for r in rows:
            if r.get("form") not in ANNUAL_FORMS or "start" not in r:
                continue
            start, end = date.fromisoformat(r["start"]), date.fromisoformat(r["end"])
            if not MIN_FY_DAYS <= (end - start).days <= MAX_FY_DAYS:
                continue  # skip quarterly / partial-period values inside 10-Ks
            records.append({"end": end, "eps": float(r["val"]), "filed": date.fromisoformat(r["filed"])})
        if not records:
            continue
        df = pd.DataFrame(records).sort_values(["end", "filed"])
        df = df.drop_duplicates("end", keep="last").set_index("end").sort_index()
        return _merge_near_duplicate_years(df)
    return pd.DataFrame(columns=["eps", "filed"])


def _merge_near_duplicate_years(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse fiscal-year ends a few days apart (52/53-week calendars tagged inconsistently)."""
    keep: list[date] = []
    for end in df.index:
        if keep and (end - keep[-1]).days < 300:
            # Same fiscal year reported with slightly different end date: keep the latest filing.
            if df.loc[end, "filed"] >= df.loc[keep[-1], "filed"]:
                keep[-1] = end
        else:
            keep.append(end)
    return df.loc[keep]


def split_adjust_eps(eps: pd.DataFrame, splits: pd.Series) -> pd.Series:
    """Restate EPS to today's share basis.

    A value filed on date F is on the share basis as of F, so it is divided by the
    cumulative ratio of all splits that took effect after F.
    """
    if splits is None or splits.empty:
        return eps["eps"].copy()
    split_dates = [pd.Timestamp(d).date() for d in splits.index]
    ratios = list(splits.astype(float))
    adjusted = {}
    for end, row in eps.iterrows():
        factor = 1.0
        for d, ratio in zip(split_dates, ratios):
            if d > row["filed"] and ratio > 0:
                factor *= ratio
        adjusted[end] = row["eps"] / factor
    return pd.Series(adjusted, name="eps")


# --------------------------------------------------------------------------- #
# Yahoo Finance: prices, splits, balance sheet
# --------------------------------------------------------------------------- #
def get_prices_and_splits(yf_tickers: list[str], years: int) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Batch-download daily history once: latest close per ticker, plus split events."""
    period = f"{years + 2}y"  # splits since the oldest 10-K used can matter
    data = yf.download(yf_tickers, period=period, interval="1d", auto_adjust=False, actions=True,
                       progress=False, threads=True, group_by="column")
    close, split_col = data["Close"], data["Stock Splits"]
    if isinstance(close, pd.Series):  # single ticker
        close, split_col = close.to_frame(yf_tickers[0]), split_col.to_frame(yf_tickers[0])
    prices = close.ffill().iloc[-1].dropna()
    splits = {t: split_col[t][split_col[t].fillna(0) > 0] for t in split_col.columns}
    return prices, splits


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
    years: int = 5


def evaluate_earnings(eps_annual: pd.DataFrame, splits: pd.Series, price: float,
                      criteria: Criteria, today: date | None = None) -> dict:
    """Apply the earnings rules to one stock. Returns a dict with ``status`` and metrics."""
    today = today or date.today()
    if len(eps_annual) < criteria.years:
        return {"status": f"fewer than {criteria.years} years of EPS"}
    last = eps_annual.iloc[-criteria.years:]
    if (today - last.index[-1]).days > MAX_STALENESS_DAYS:
        return {"status": "EPS data is stale"}
    eps = split_adjust_eps(last, splits)
    if (eps <= 0).any():
        return {"status": "negative/zero EPS in a year"}
    avg_eps = float(eps.mean())
    pe = price / avg_eps
    result = {
        "avg_eps": avg_eps,
        "pe": pe,
        "eps_years": f"FY{last.index[0].year}-FY{last.index[-1].year}",
    }
    result["status"] = "ok" if pe < criteria.max_pe else f"P/E >= {criteria.max_pe:g}"
    return result


def run_screen(criteria: Criteria, user_agent: str, tickers: list[str] | None = None,
               workers: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    log("Loading S&P 500 constituents from Wikipedia...")
    universe = get_sp500_constituents()
    if tickers:
        wanted = {t.upper() for t in tickers}
        universe = universe[universe["ticker"].isin(wanted) | universe["yf_ticker"].isin(wanted)]
    log(f"  {len(universe)} companies")

    log("Downloading prices and split history from Yahoo Finance...")
    prices, splits_by_ticker = get_prices_and_splits(universe["yf_ticker"].tolist(), criteria.years)

    log(f"Downloading annual EPS history from SEC EDGAR ({len(universe)} filers)...")
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    limiter = RateLimiter(per_second=8)
    eps_by_cik: dict[int, pd.DataFrame] = {}
    ciks = universe["cik"].unique().tolist()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_company_facts, cik, session, limiter): cik for cik in ciks}
        for i, fut in enumerate(as_completed(futures), 1):
            facts = fut.result()
            eps_by_cik[futures[fut]] = extract_annual_eps(facts) if facts else pd.DataFrame(columns=["eps", "filed"])
            if i % 50 == 0 or i == len(ciks):
                log(f"  {i}/{len(ciks)}")

    log("Applying earnings and P/E rules...")
    statuses: dict[str, str] = {}
    candidates = []
    for row in universe.itertuples(index=False):
        price = prices.get(row.yf_ticker)
        if price is None or pd.isna(price):
            statuses[row.ticker] = "no price"
            continue
        eps_annual = eps_by_cik.get(row.cik)
        if eps_annual is None or eps_annual.empty:
            statuses[row.ticker] = "no EPS data in SEC filings"
            continue
        res = evaluate_earnings(eps_annual, splits_by_ticker.get(row.yf_ticker), float(price), criteria)
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
            f"Avg EPS ({criteria.years}y)": res["avg_eps"],
            "P/E": res["pe"],
            "Current Ratio": cr,
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
    print(f"Rules: EPS > 0 in each of last {criteria.years} fiscal years, "
          f"P/E (price / {criteria.years}y avg EPS) < {criteria.max_pe:g}, "
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
    p.add_argument("--years", type=int, default=5, help="years of EPS to average (default 5)")
    p.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"),
                   help='SEC User-Agent, e.g. "Jane Doe jane@example.com" (or set SEC_USER_AGENT)')
    p.add_argument("--tickers", nargs="+", help="only screen these tickers (for quick tests)")
    args = p.parse_args(argv)

    if not args.user_agent or "@" not in args.user_agent:
        p.error('SEC requires a User-Agent with your name and email: --user-agent "Jane Doe jane@example.com"')

    criteria = Criteria(max_pe=args.max_pe, min_current_ratio=args.min_current_ratio, years=args.years)
    df, statuses = run_screen(criteria, args.user_agent, args.tickers)
    print_report(df, statuses, criteria)
    return 0


if __name__ == "__main__":
    sys.exit(main())
