#!/usr/bin/env python3
"""S&P 500 value screener.

Selects S&P 500 stocks with:
  * positive diluted EPS in every fiscal year used (the last 5, or at least 4 if
    Yahoo has fewer),
  * P/E < max_pe, where P/E = current price / average diluted EPS of those years,
  * current ratio (current assets / current liabilities) > min_current_ratio.

For each selected stock it also computes operating-investment indicators:
  Available Shares          = common shares + options + RSU/PSU + DSU + exchangeable shares
  Paid for Entire Company   = Available Shares x price
  Paid for Operating Prop.  = Paid for Entire Company - current assets
  Earnings Before Amort.    = latest annual net income + annual D&A
  Balance Earned on Op. Inv = Earnings Before Amort. - 5% of current assets
  % Earned Before Amort.    = Balance Earned / Paid for Operating Property x 100
  Estimated Life            = net PP&E / annual D&A
  Investor's Amortization   = Paid for Operating Property / Estimated Life
  Earned After Amort.       = Balance Earned - Investor's Amortization
  % Earned After Amort.     = Earned After Amort. / Paid for Operating Property x 100

Data sources:
  * S&P 500 constituents (ticker, name, sector, CIK): Wikipedia.
  * Annual EPS, net income, D&A, current price, latest balance sheet: Yahoo Finance via yfinance.
  * Stock options outstanding and nonvested RSU/PSU counts: SEC EDGAR XBRL "companyfacts"
    (fetched only for selected stocks).

SEC requires a User-Agent that identifies you, e.g. "Jane Doe jane@example.com".
Pass it with --user-agent or the SEC_USER_AGENT environment variable.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import requests
import yfinance as yf

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
# Statement rows tried in order for each figure.
EPS_ROWS = ("Diluted EPS", "Basic EPS")
NET_INCOME_ROWS = ("Net Income Common Stockholders", "Net Income")
DA_ROWS = ("Depreciation And Amortization", "Depreciation Amortization Depletion")
DA_INCOME_ROWS = ("Reconciled Depreciation",)
SHARES_ROWS = ("Ordinary Shares Number", "Share Issued")
NET_PPE_ROWS = ("Net PPE",)
# SEC XBRL concepts (us-gaap, unit "shares") for potential new shares.
OPTIONS_CONCEPT = "ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsOutstandingNumber"
STOCK_AWARDS_CONCEPT = ("ShareBasedCompensationArrangementByShareBasedPaymentAward"
                        "EquityInstrumentsOtherThanOptionsNonvestedNumber")
SEC_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A"}
# Share of current assets deducted from earnings in "Balance Earned on Operating Investment".
CURRENT_ASSET_CHARGE = 0.05
# The most recent fiscal year must have ended within this many days, so stale data drops out.
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
# Yahoo Finance: prices, statements
# --------------------------------------------------------------------------- #
def get_latest_prices(yf_tickers: list[str]) -> pd.Series:
    """Batch-download recent daily closes and return the latest close per ticker."""
    data = yf.download(yf_tickers, period="10d", interval="1d", auto_adjust=False,
                       progress=False, threads=True, group_by="column")
    close = data["Close"]
    if isinstance(close, pd.Series):  # single ticker
        close = close.to_frame(yf_tickers[0])
    return close.ffill().iloc[-1].dropna()


def get_annual_eps(income_stmt: pd.DataFrame | None) -> pd.Series:
    """Return annual EPS indexed by fiscal-year end date, oldest first (empty if unavailable)."""
    if income_stmt is None or income_stmt.empty:
        return pd.Series(dtype=float)
    for row in EPS_ROWS:
        if row in income_stmt.index:
            eps = pd.to_numeric(income_stmt.loc[row], errors="coerce").dropna()
            if not eps.empty:
                eps.index = [pd.Timestamp(d).date() for d in eps.index]
                return eps.sort_index()
    return pd.Series(dtype=float)


def latest_value(sheets, rows, col=None) -> float | None:
    """Newest non-missing value of the first available row, searching ``sheets`` in order.

    With ``col``, that column of the first sheet is tried before falling back to the newest
    value anywhere, so figures line up with the same balance-sheet date when possible.
    """
    sheets = [sh for sh in sheets if sh is not None and not sh.empty]
    if col is not None and sheets and col in sheets[0].columns:
        for row in rows:
            if row in sheets[0].index and pd.notna(sheets[0].at[row, col]):
                return float(sheets[0].at[row, col])
    for sheet in sheets:
        for row in rows:
            if row in sheet.index:
                vals = pd.to_numeric(sheet.loc[row], errors="coerce").dropna()
                if not vals.empty:
                    return float(vals.sort_index().iloc[-1])
    return None


def get_balance_sheet(ticker: yf.Ticker) -> dict | None:
    """Latest balance-sheet figures: quarterly preferred, annual as fallback.

    Returns None when current assets / liabilities are unavailable (e.g. banks, insurers).
    """
    quarterly, annual = ticker.quarterly_balance_sheet, ticker.balance_sheet
    for sheet in (quarterly, annual):
        if sheet is None or sheet.empty:
            continue
        if "Current Assets" not in sheet.index or "Current Liabilities" not in sheet.index:
            continue
        for col in sheet.columns:  # newest first
            ca, cl = sheet.at["Current Assets", col], sheet.at["Current Liabilities", col]
            if pd.notna(ca) and pd.notna(cl) and cl > 0:
                others = [sheet] + [sh for sh in (quarterly, annual) if sh is not sheet]
                return {
                    "date": pd.Timestamp(col).date().isoformat(),
                    "current_assets": float(ca),
                    "current_liabilities": float(cl),
                    "shares": latest_value(others, SHARES_ROWS, col),
                    "net_ppe": latest_value(others, NET_PPE_ROWS, col),
                }
    return None


def get_annual_flows(income_stmt: pd.DataFrame | None, cashflow: pd.DataFrame | None) -> dict:
    """Latest annual net income and depreciation & amortization (D&A)."""
    da = latest_value([cashflow], DA_ROWS)
    if da is None:
        da = latest_value([income_stmt], DA_INCOME_ROWS)
    return {"net_income": latest_value([income_stmt], NET_INCOME_ROWS),
            "da": abs(da) if da is not None else None}


# --------------------------------------------------------------------------- #
# SEC EDGAR: options and stock awards outstanding
# --------------------------------------------------------------------------- #
def fetch_company_facts(cik: int, session: requests.Session, retries: int = 3) -> dict | None:
    url = SEC_FACTS_URL.format(cik=cik)
    for attempt in range(retries):
        time.sleep(0.15)  # stay well under SEC's 10 requests/second
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


def latest_share_count(facts: dict | None, concept: str, today: date | None = None) -> tuple[float | None, str | None]:
    """Most recent point-in-time share count for a us-gaap concept from 10-K/10-Q filings."""
    today = today or date.today()
    rows = (facts or {}).get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {}).get("shares", [])
    rows = [r for r in rows if r.get("form") in SEC_FORMS and "start" not in r]
    if not rows:
        return None, None
    best = max(rows, key=lambda r: (r["end"], r["filed"]))
    if (today - date.fromisoformat(best["end"])).days > MAX_STALENESS_DAYS:
        return None, None
    return float(best["val"]), best["end"]


def get_dilution(facts: dict | None) -> dict:
    """Option and stock-award counts. DSUs and exchangeable shares have no standard us-gaap
    concept, so they are not reported separately (DSUs that are still unvested are usually
    included in the stock-award count)."""
    options, options_date = latest_share_count(facts, OPTIONS_CONCEPT)
    awards, awards_date = latest_share_count(facts, STOCK_AWARDS_CONCEPT)
    return {"options": options, "stock_awards": awards, "dsu": None, "exchangeable": None,
            "as_of": max(filter(None, [options_date, awards_date]), default=None)}


# --------------------------------------------------------------------------- #
# Operating-investment indicators
# --------------------------------------------------------------------------- #
def compute_operating_metrics(price: float, shares: float | None, current_assets: float | None,
                              net_income: float | None, da: float | None, net_ppe: float | None,
                              options: float | None = None, stock_awards: float | None = None,
                              dsu: float | None = None, exchangeable: float | None = None) -> dict:
    """Steps 1-10. Missing inputs give NaN for the values that depend on them.

    Percentages are NaN when Paid for Operating Property is <= 0 (current assets exceed the
    price of the whole company), since dividing by it would flip the sign of the result.
    """
    nan = float("nan")
    f = lambda x: nan if x is None else float(x)  # noqa: E731
    dilution = sum(x or 0.0 for x in (options, stock_awards, dsu, exchangeable))
    available = f(shares) + dilution
    paid_entire = available * price
    paid_op = paid_entire - f(current_assets)
    eba = f(net_income) + f(da)
    balance = eba - CURRENT_ASSET_CHARGE * f(current_assets)
    life = f(net_ppe) / f(da) if da and net_ppe and da > 0 and net_ppe > 0 else nan
    valid_base = paid_op > 0
    inv_amort = paid_op / life if valid_base and life > 0 else nan
    after = balance - inv_amort
    return {
        "available_shares": available,
        "paid_entire": paid_entire,
        "paid_op": paid_op,
        "eba": eba,
        "balance": balance,
        "pct_before": balance / paid_op * 100 if valid_base else nan,
        "est_life": life,
        "inv_amort": inv_amort,
        "after": after,
        "pct_after": after / paid_op * 100 if valid_base else nan,
    }


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


def run_screen(criteria: Criteria, user_agent: str, tickers: list[str] | None = None,
               workers: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    log("Loading S&P 500 constituents from Wikipedia...")
    universe = get_sp500_constituents()
    if tickers:
        wanted = {t.upper() for t in tickers}
        universe = universe[universe["ticker"].isin(wanted) | universe["yf_ticker"].isin(wanted)]
    log(f"  {len(universe)} companies")

    log("Downloading latest prices from Yahoo Finance...")
    prices = get_latest_prices(universe["yf_ticker"].tolist())

    log(f"Downloading annual income statements from Yahoo Finance ({len(universe)} stocks)...")
    income_by_ticker: dict[str, pd.DataFrame | None] = {}
    symbols = universe["yf_ticker"].tolist()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_safe, lambda s=s: yf.Ticker(s).income_stmt): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            income_by_ticker[futures[fut]] = fut.result()
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
        eps_annual = get_annual_eps(income_by_ticker.get(row.yf_ticker))
        if eps_annual.empty:
            statuses[row.ticker] = "no EPS data on Yahoo"
            continue
        res = evaluate_earnings(eps_annual, float(price), criteria)
        statuses[row.ticker] = res["status"]
        if res["status"] == "ok":
            candidates.append((row, float(price), res))

    log(f"Checking current ratio for {len(candidates)} stocks that pass P/E...")
    selected = []
    for row, price, res in candidates:
        yt = yf.Ticker(row.yf_ticker)
        bs = _safe(lambda: get_balance_sheet(yt))
        if bs is None:
            statuses[row.ticker] = "no current ratio (e.g. bank/insurer)"
            continue
        cr = bs["current_assets"] / bs["current_liabilities"]
        if cr <= criteria.min_current_ratio:
            statuses[row.ticker] = f"current ratio <= {criteria.min_current_ratio:g}"
            continue
        statuses[row.ticker] = "selected"
        flows = get_annual_flows(income_by_ticker.get(row.yf_ticker), _safe(lambda: yt.cashflow))
        selected.append((row, price, res, cr, bs, flows))

    log(f"Downloading option / RSU counts from SEC EDGAR ({len(selected)} stocks)...")
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    results = []
    for row, price, res, cr, bs, flows in selected:
        dil = get_dilution(fetch_company_facts(row.cik, session))
        m = compute_operating_metrics(price, bs["shares"], bs["current_assets"], flows["net_income"],
                                      flows["da"], bs["net_ppe"], dil["options"], dil["stock_awards"],
                                      dil["dsu"], dil["exchangeable"])
        notes = []
        if dil["options"] is None and dil["stock_awards"] is None:
            notes.append("no option/RSU data")
        if bs["shares"] is None:
            notes.append("no share count")
        if flows["da"] is None:
            notes.append("no D&A")
        if m["paid_op"] <= 0:
            notes.append("current assets > company price")
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
            "BS Date": bs["date"],
            "Notes": ", ".join(notes),
            # inputs and intermediate values, shown with --details
            "common_shares": bs["shares"], "options": dil["options"], "stock_awards": dil["stock_awards"],
            "dilution_as_of": dil["as_of"], "current_assets": bs["current_assets"],
            "net_income": flows["net_income"], "da": flows["da"], "net_ppe": bs["net_ppe"],
            **m,
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


SCREEN_COLS = ["Ticker", "Company", "Sector", "Price", "Avg EPS", "P/E", "Current Ratio", "Yrs", "EPS Years"]

DETAIL_ROWS = [
    ("common_shares", "Common shares", "shares"),
    ("options", "+ Stock options (SEC)", "shares"),
    ("stock_awards", "+ RSU/PSU nonvested (SEC)", "shares"),
    ("available_shares", "1. Available Shares", "shares"),
    ("Price", "   x Price", "price"),
    ("paid_entire", "2. Paid for Entire Company", "money"),
    ("current_assets", "   - Current assets", "money"),
    ("paid_op", "3. Paid for Operating Property", "money"),
    ("net_income", "   Net income (latest FY)", "money"),
    ("da", "   + D&A (latest FY)", "money"),
    ("eba", "4. Earnings Before Amortization", "money"),
    ("balance", "5. Balance Earned on Operating Investment", "money"),
    ("pct_before", "6. % Earned Before Amortization", "pct"),
    ("net_ppe", "   Net PP&E", "money"),
    ("est_life", "7. Estimated Life (years)", "num"),
    ("inv_amort", "8. Investor's Amortization", "money"),
    ("after", "9. Earned on Operating Investment After Amortization", "money"),
    ("pct_after", "10. % Earned After Amortization", "pct"),
]


def _fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if kind == "shares":
        return f"{value:,.0f}"
    if kind == "money":
        return f"{'-' if value < 0 else ''}${abs(value) / 1e6:,.1f}M"
    if kind == "pct":
        return f"{value:.2f}%"
    return f"{value:,.2f}"


def operating_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Ticker": df["Ticker"],
        "Paid for Operating Property": [_fmt(v, "money") for v in df["paid_op"]],
        "% Earned Before Amortization": [_fmt(v, "pct") for v in df["pct_before"]],
        "% Earned After Amortization": [_fmt(v, "pct") for v in df["pct_after"]],
        "Notes": df["Notes"],
    })


def print_report(df: pd.DataFrame, statuses: pd.Series, criteria: Criteria, details: bool = False) -> None:
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
            print(df[SCREEN_COLS].to_string(index=False))
            print()
            print("Operating-investment indicators")
            print(operating_table(df).to_string(index=False))
        if details:
            for _, r in df.iterrows():
                print()
                print(f"{r['Ticker']} - {r['Company']}  (balance sheet {r['BS Date']}, "
                      f"option/RSU counts {r['dilution_as_of'] or 'n/a'})")
                for key, label, kind in DETAIL_ROWS:
                    print(f"  {label:<55} {_fmt(r[key], kind):>22}")
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
    p.add_argument("--details", action="store_true", help="show every step of the indicator calculations")
    p.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"),
                   help='SEC User-Agent, e.g. "Jane Doe jane@example.com" (or set SEC_USER_AGENT)')
    args = p.parse_args(argv)

    if not args.user_agent or "@" not in args.user_agent:
        p.error('SEC requires a User-Agent with your name and email: --user-agent "Jane Doe jane@example.com"')
    if not 1 <= args.min_years <= args.years:
        p.error("--min-years must be between 1 and --years")

    criteria = Criteria(max_pe=args.max_pe, min_current_ratio=args.min_current_ratio,
                        years=args.years, min_years=args.min_years)
    df, statuses = run_screen(criteria, args.user_agent, args.tickers)
    print_report(df, statuses, criteria, details=args.details)
    return 0


if __name__ == "__main__":
    sys.exit(main())
