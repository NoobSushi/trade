import math
from datetime import date

import pandas as pd
import pytest

import screener
from screener import Criteria, compute_operating_metrics, evaluate_earnings, get_annual_eps

TODAY = date(2026, 10, 2)


def income_stmt(values: dict, row="Diluted EPS", start=2021):
    """Fake yfinance income_stmt: rows are line items, columns are FY ends, newest first."""
    cols = [pd.Timestamp(f"{start + i}-12-31") for i in range(len(next(iter(values.values()))))]
    df = pd.DataFrame({c: [v[i] for v in values.values()] for i, c in enumerate(cols)}, index=list(values))
    return df[cols[::-1]]


def eps_series(vals, start=2021):
    return pd.Series(vals, index=[date(y, 12, 31) for y in range(start, start + len(vals))], dtype=float)


def test_get_annual_eps_sorts_oldest_first_and_drops_nan():
    stmt = income_stmt({"Diluted EPS": [float("nan"), 2.0, 3.0, 4.0, 5.0], "Net Income": [1, 2, 3, 4, 5]})
    eps = get_annual_eps(stmt)
    assert list(eps.index) == [date(2022, 12, 31), date(2023, 12, 31), date(2024, 12, 31), date(2025, 12, 31)]
    assert eps.tolist() == [2.0, 3.0, 4.0, 5.0]


def test_get_annual_eps_falls_back_to_basic_and_handles_empty():
    assert get_annual_eps(income_stmt({"Basic EPS": [1.0, 2.0]})).tolist() == [1.0, 2.0]
    assert get_annual_eps(pd.DataFrame()).empty
    assert get_annual_eps(None).empty
    assert get_annual_eps(income_stmt({"Net Income": [1.0]})).empty


def test_evaluate_averages_up_to_5_years():
    res = evaluate_earnings(eps_series([1, 2, 3, 4, 5, 6], start=2020), 40.0, Criteria(), today=TODAY)
    assert res["status"] == "ok" and res["avg_eps"] == 4 and res["pe"] == 10 and res["n_years"] == 5


def test_evaluate_accepts_4_years_rejects_3():
    res = evaluate_earnings(eps_series([2, 2, 2, 2], start=2022), 20.0, Criteria(), today=TODAY)
    assert res["status"] == "ok" and res["n_years"] == 4 and res["eps_years"] == "FY2022-FY2025"
    res = evaluate_earnings(eps_series([2, 2, 2], start=2023), 20.0, Criteria(), today=TODAY)
    assert res["status"].startswith("fewer than 4")


def test_evaluate_rejects_negative_year_high_pe_and_stale():
    c = Criteria()
    assert evaluate_earnings(eps_series([5, 5, -0.1, 5, 5]), 10.0, c, today=TODAY)["status"].startswith("negative")
    assert evaluate_earnings(eps_series([1] * 5), 15.0, c, today=TODAY)["status"].startswith("P/E")
    assert evaluate_earnings(eps_series([1] * 5, start=2018), 5.0, c, today=TODAY)["status"] == "EPS data is stale"


def test_balance_sheet_prefers_latest_quarter_and_falls_back_for_missing_rows():
    class T:
        quarterly_balance_sheet = pd.DataFrame(
            {pd.Timestamp("2026-06-30"): [300.0, 100.0, None, 50.0],
             pd.Timestamp("2026-03-31"): [100.0, 100.0, 9.0, 40.0]},
            index=["Current Assets", "Current Liabilities", "Ordinary Shares Number", "Net PPE"])
        balance_sheet = pd.DataFrame()
    bs = screener.get_balance_sheet(T())
    assert bs == {"date": "2026-06-30", "current_assets": 300.0, "current_liabilities": 100.0,
                  "shares": 9.0, "net_ppe": 50.0}


def test_balance_sheet_none_without_current_items():
    class T:
        quarterly_balance_sheet = pd.DataFrame({pd.Timestamp("2026-06-30"): [1.0]}, index=["Total Assets"])
        balance_sheet = pd.DataFrame()
    assert screener.get_balance_sheet(T()) is None


def test_annual_flows_uses_latest_year_and_positive_da():
    inc = income_stmt({"Net Income Common Stockholders": [80.0, 100.0], "Reconciled Depreciation": [5.0, 6.0]})
    cf = income_stmt({"Depreciation And Amortization": [-18.0, -20.0]})
    assert screener.get_annual_flows(inc, cf) == {"net_income": 100.0, "da": 20.0}
    assert screener.get_annual_flows(inc, None) == {"net_income": 100.0, "da": 6.0}


def test_latest_share_count_picks_newest_instant_from_filings():
    rows = [
        {"end": "2024-12-31", "val": 900, "form": "10-K", "filed": "2025-02-10"},
        {"end": "2025-12-31", "val": 1000, "form": "10-K", "filed": "2026-02-10"},
        {"end": "2026-06-30", "val": 5, "form": "8-K", "filed": "2026-07-01"},          # not a 10-K/10-Q
        {"start": "2025-01-01", "end": "2026-06-30", "val": 7, "form": "10-Q", "filed": "2026-08-01"},  # period
    ]
    facts = {"facts": {"us-gaap": {screener.OPTIONS_CONCEPT: {"units": {"shares": rows}}}}}
    assert screener.latest_share_count(facts, screener.OPTIONS_CONCEPT, today=TODAY) == (1000.0, "2025-12-31")
    assert screener.latest_share_count(facts, screener.OPTIONS_CONCEPT, today=date(2030, 1, 1)) == (None, None)
    assert screener.latest_share_count(None, screener.OPTIONS_CONCEPT) == (None, None)


def test_operating_metrics_follow_the_10_steps():
    m = compute_operating_metrics(price=10.0, shares=900, current_assets=2_000, net_income=1_000, da=500,
                                  net_ppe=5_000, options=60, stock_awards=40)
    assert m["available_shares"] == 1_000                 # 900 + 60 + 40
    assert m["paid_entire"] == 10_000                     # x price
    assert m["paid_op"] == 8_000                          # - current assets
    assert m["eba"] == 1_500                              # net income + D&A
    assert m["balance"] == 1_400                          # - 5% of current assets
    assert m["pct_before"] == pytest.approx(17.5)         # 1400 / 8000
    assert m["est_life"] == 10                            # 5000 / 500
    assert m["inv_amort"] == 800                          # 8000 / 10
    assert m["after"] == 600                              # 1400 - 800
    assert m["pct_after"] == pytest.approx(7.5)           # 600 / 8000


def test_operating_metrics_keep_sign_when_paid_for_operating_property_is_negative():
    m = compute_operating_metrics(price=10.0, shares=100, current_assets=2_000, net_income=100, da=50, net_ppe=500)
    assert m["paid_op"] == -1_000                         # 1000 - 2000
    assert m["balance"] == 50                             # 150 - 100
    assert m["pct_before"] == pytest.approx(-5.0)         # 50 / -1000
    assert m["inv_amort"] == pytest.approx(-100)          # -1000 / 10
    assert m["after"] == pytest.approx(150)               # 50 - (-100)
    assert m["pct_after"] == pytest.approx(-15.0)         # 150 / -1000


def test_operating_metrics_handle_missing_and_zero_base():
    m = compute_operating_metrics(price=10.0, shares=200, current_assets=2_000, net_income=100, da=50, net_ppe=500)
    assert m["paid_op"] == 0 and math.isnan(m["pct_before"]) and math.isnan(m["pct_after"])
    m = compute_operating_metrics(price=10.0, shares=1_000, current_assets=2_000, net_income=100, da=None,
                                  net_ppe=500)
    assert m["paid_op"] == 8_000 and math.isnan(m["pct_before"]) and math.isnan(m["est_life"])


def test_run_screen_end_to_end(monkeypatch):
    universe = pd.DataFrame({
        "ticker": ["GOOD", "LOWCR", "LOSS", "PRICY", "SHORT", "BRK.B"],
        "name": ["Good", "LowCR", "Loss", "Pricy", "Short", "Berkshire"],
        "sector": ["X"] * 6, "cik": range(1, 7),
    })
    universe["yf_ticker"] = universe["ticker"].str.replace(".", "-", regex=False)
    universe["member_of"] = "S&P 500"
    monkeypatch.setattr(screener, "get_universe", lambda keys, session: universe)
    prices = pd.Series({"GOOD": 50.0, "LOWCR": 50.0, "LOSS": 50.0, "PRICY": 500.0, "SHORT": 10.0, "BRK-B": 60.0})
    monkeypatch.setattr(screener, "get_latest_prices", lambda t: prices)

    first = date.today().year - 4  # latest fiscal year = last calendar year
    eps = {"GOOD": [5] * 4, "LOWCR": [5] * 4, "LOSS": [5, -1, 5, 5], "PRICY": [5] * 4,
           "SHORT": [5] * 3, "BRK-B": [6] * 4}
    class FakeTicker:
        def __init__(self, sym):
            self.sym, self.income_stmt, self.cashflow = sym, sym, None
    monkeypatch.setattr(screener.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(screener, "get_annual_eps",
                        lambda stmt: eps_series(eps[stmt], start=first + 4 - len(eps[stmt])))

    def bs(ca, cl):
        return {"date": "2026-06-30", "current_assets": ca, "current_liabilities": cl,
                "shares": 1_000.0, "net_ppe": 5_000.0}
    sheets = {"GOOD": bs(2_500.0, 1_000.0), "LOWCR": bs(1_500.0, 1_000.0), "BRK-B": None}
    monkeypatch.setattr(screener, "get_balance_sheet", lambda t: sheets[t.sym])
    monkeypatch.setattr(screener, "get_annual_flows", lambda inc, cf: {"net_income": 4_000.0, "da": 500.0})
    sec_calls = []
    monkeypatch.setattr(screener, "fetch_company_facts", lambda cik, session: sec_calls.append(cik) or None)

    df, statuses = screener.run_screen(Criteria(), "test test@example.com")
    assert df["Ticker"].tolist() == ["GOOD"]
    assert df["P/E"].iloc[0] == pytest.approx(10.0)
    assert sec_calls == [1]  # SEC is only queried for selected stocks
    good = df.iloc[0]
    assert good["paid_op"] == pytest.approx(1_000 * 50 - 2_500)
    assert good["pct_before"] == pytest.approx((4_500 - 125) / 47_500 * 100)
    assert good["Notes"] == "no option/RSU data"
    assert good["Index"] == "S&P 500"
    assert statuses["LOWCR"].startswith("current ratio")
    assert statuses["LOSS"].startswith("negative")
    assert statuses["PRICY"].startswith("P/E")
    assert statuses["SHORT"].startswith("fewer than 4")
    assert statuses["BRK.B"].startswith("no current ratio")
    screener.print_report(df, statuses, Criteria(), details=True)


def html_table(header, rows, table_id="constituents"):
    th = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<html><body><table id="{table_id}"><tr>{th}</tr>{trs}</table></body></html>'


SP500_HTML = html_table(["Symbol", "Security", "GICS Sector", "CIK"],
                        [["AAPL", "Apple Inc.", "Information Technology", "320193"],
                         ["BRK.B", "Berkshire Hathaway", "Financials", "1067983"]])
NDX_HTML = html_table(["Company", "Ticker", "GICS Sector", "GICS Sub-Industry"],
                      [["Apple Inc.", "AAPL", "Information Technology", "Hardware"],
                       ["Lululemon", "LULU", "Consumer Discretionary", "Apparel"]])
DOW_HTML = html_table(["Company", "Exchange", "Symbol", "Industry"],
                      [["Apple Inc.", "NASDAQ", "AAPL", "Information technology"],
                       ["Travelers", "NYSE", "TRV", "Insurance"]], table_id="other")


def test_parse_constituents_handles_each_wikipedia_layout():
    sp = screener.parse_constituents(SP500_HTML, "S&P 500")
    assert sp["yf_ticker"].tolist() == ["AAPL", "BRK-B"] and sp["cik"].tolist() == [320193, 1067983]
    ndx = screener.parse_constituents(NDX_HTML, "Nasdaq-100")
    assert ndx["ticker"].tolist() == ["AAPL", "LULU"] and ndx["cik"].isna().all()
    dow = screener.parse_constituents(DOW_HTML, "Dow")  # no id="constituents": falls back to any table
    assert dow["ticker"].tolist() == ["AAPL", "TRV"] and dow["sector"].tolist()[1] == "Insurance"


def test_get_universe_merges_duplicates_and_fills_ciks(monkeypatch):
    pages = {screener.INDEXES["sp500"][1]: SP500_HTML, screener.INDEXES["nasdaq100"][1]: NDX_HTML,
             screener.INDEXES["dow"][1]: DOW_HTML}

    class Resp:
        def __init__(self, text):
            self.text = text
        def raise_for_status(self):
            pass
    monkeypatch.setattr(screener.requests, "get", lambda url, **kw: Resp(pages[url]))
    monkeypatch.setattr(screener, "fetch_sec_ciks", lambda session: {"LULU": 1397187, "TRV": 86312})

    df = screener.get_universe(list(screener.INDEXES), session=None).set_index("ticker")
    assert sorted(df.index) == ["AAPL", "BRK.B", "LULU", "TRV"]
    assert df.loc["AAPL", "member_of"] == "S&P 500, Nasdaq-100, Dow"
    assert df.loc["AAPL", "sector"] == "Information Technology"  # first index's value wins
    assert df.loc["LULU", "cik"] == 1397187 and df.loc["TRV", "cik"] == 86312


def test_get_universe_skips_an_index_that_fails(monkeypatch):
    def get(url, **kw):
        if "Nasdaq" in url:
            raise screener.requests.ConnectionError("blocked")
        class R:
            text = DOW_HTML
            def raise_for_status(self):
                pass
        return R()
    monkeypatch.setattr(screener.requests, "get", get)
    monkeypatch.setattr(screener, "fetch_sec_ciks", lambda session: {})
    df = screener.get_universe(["nasdaq100", "dow"], session=None)
    assert df["ticker"].tolist() == ["AAPL", "TRV"] and df["cik"].isna().all()
