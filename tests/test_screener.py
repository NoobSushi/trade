from datetime import date

import pandas as pd
import pytest

import screener
from screener import Criteria, evaluate_earnings, get_annual_eps

TODAY = date(2026, 10, 2)


def income_stmt(values: dict, row="Diluted EPS", start=2021):
    """Fake yfinance income_stmt: rows are line items, columns are FY ends, newest first."""
    cols = [pd.Timestamp(f"{start + i}-12-31") for i in range(len(next(iter(values.values()))))]
    df = pd.DataFrame({c: [v[i] for v in values.values()] for i, c in enumerate(cols)}, index=list(values))
    return df[cols[::-1]]


class FakeTicker:
    def __init__(self, stmt):
        self.income_stmt = stmt


def eps_series(vals, start=2021):
    return pd.Series(vals, index=[date(y, 12, 31) for y in range(start, start + len(vals))], dtype=float)


def test_get_annual_eps_sorts_oldest_first_and_drops_nan():
    stmt = income_stmt({"Diluted EPS": [float("nan"), 2.0, 3.0, 4.0, 5.0], "Net Income": [1, 2, 3, 4, 5]})
    eps = get_annual_eps(FakeTicker(stmt))
    assert list(eps.index) == [date(2022, 12, 31), date(2023, 12, 31), date(2024, 12, 31), date(2025, 12, 31)]
    assert eps.tolist() == [2.0, 3.0, 4.0, 5.0]


def test_get_annual_eps_falls_back_to_basic_and_handles_empty():
    assert get_annual_eps(FakeTicker(income_stmt({"Basic EPS": [1.0, 2.0]}))).tolist() == [1.0, 2.0]
    assert get_annual_eps(FakeTicker(pd.DataFrame())).empty
    assert get_annual_eps(FakeTicker(income_stmt({"Net Income": [1.0]}))).empty


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


def test_current_ratio_prefers_latest_quarter():
    class T:
        quarterly_balance_sheet = pd.DataFrame(
            {pd.Timestamp("2026-06-30"): [300.0, 100.0], pd.Timestamp("2026-03-31"): [100.0, 100.0]},
            index=["Current Assets", "Current Liabilities"])
        balance_sheet = pd.DataFrame()
    assert screener.get_current_ratio(T()) == (3.0, "2026-06-30")


def test_run_screen_end_to_end(monkeypatch):
    universe = pd.DataFrame({
        "ticker": ["GOOD", "LOWCR", "LOSS", "PRICY", "SHORT", "BRK.B"],
        "name": ["Good", "LowCR", "Loss", "Pricy", "Short", "Berkshire"],
        "sector": ["X"] * 6,
    })
    universe["yf_ticker"] = universe["ticker"].str.replace(".", "-", regex=False)
    monkeypatch.setattr(screener, "get_sp500_constituents", lambda: universe)
    prices = pd.Series({"GOOD": 50.0, "LOWCR": 50.0, "LOSS": 50.0, "PRICY": 500.0, "SHORT": 10.0, "BRK-B": 60.0})
    monkeypatch.setattr(screener, "get_latest_prices", lambda t: prices)

    first = date.today().year - 4  # latest fiscal year = last calendar year
    eps = {"GOOD": [5] * 4, "LOWCR": [5] * 4, "LOSS": [5, -1, 5, 5], "PRICY": [5] * 4,
           "SHORT": [5] * 3, "BRK-B": [6] * 4}
    monkeypatch.setattr(screener.yf, "Ticker", lambda t: t)
    monkeypatch.setattr(screener, "get_annual_eps", lambda t: eps_series(eps[t], start=first + 4 - len(eps[t])))
    cr = {"GOOD": (2.5, "2026-06-30"), "LOWCR": (1.5, "2026-06-30"), "BRK-B": (None, None)}
    monkeypatch.setattr(screener, "get_current_ratio", lambda t: cr[t])

    df, statuses = screener.run_screen(Criteria())
    assert df["Ticker"].tolist() == ["GOOD"]
    assert df["P/E"].iloc[0] == pytest.approx(10.0)
    assert statuses["LOWCR"].startswith("current ratio")
    assert statuses["LOSS"].startswith("negative")
    assert statuses["PRICY"].startswith("P/E")
    assert statuses["SHORT"].startswith("fewer than 4")
    assert statuses["BRK.B"].startswith("no current ratio")
    screener.print_report(df, statuses, Criteria())
