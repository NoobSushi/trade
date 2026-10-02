from datetime import date

import pandas as pd
import pytest

import screener
from screener import Criteria, evaluate_earnings, extract_annual_eps, split_adjust_eps


def fy(year, val, filed=None, form="10-K", days=364):
    end = date(year, 12, 31)
    start = end - pd.Timedelta(days=days)
    return {"start": start.isoformat(), "end": end.isoformat(), "val": val, "form": form,
            "filed": (filed or date(year + 1, 2, 15)).isoformat()}


def facts(rows, concept="EarningsPerShareDiluted"):
    return {"facts": {"us-gaap": {concept: {"units": {"USD/shares": rows}}}}}


def test_extract_keeps_annual_10k_values_and_latest_restatement():
    rows = [
        fy(2021, 2.0),
        fy(2021, 2.1, filed=date(2023, 2, 15)),          # restated in a later 10-K
        fy(2022, 3.0),
        fy(2022, 0.8, days=90),                           # quarterly value inside 10-K
        {**fy(2023, 9.9), "form": "10-Q"},               # not an annual form
        fy(2023, 4.0),
    ]
    df = extract_annual_eps(facts(rows))
    assert list(df.index) == [date(2021, 12, 31), date(2022, 12, 31), date(2023, 12, 31)]
    assert list(df["eps"]) == [2.1, 3.0, 4.0]


def test_extract_falls_back_to_basic_eps():
    df = extract_annual_eps(facts([fy(2023, 1.5)], concept="EarningsPerShareBasic"))
    assert df["eps"].tolist() == [1.5]


def test_extract_merges_52_53_week_year_ends():
    a = fy(2022, 1.0)
    b = {**fy(2022, 1.1, filed=date(2024, 2, 1)), "end": "2023-01-01", "start": "2022-01-02"}
    df = extract_annual_eps(facts([a, b]))
    assert len(df) == 1 and df["eps"].iloc[0] == 1.1


def test_split_adjust_only_applies_splits_after_filing():
    eps = pd.DataFrame({"eps": [10.0, 12.0], "filed": [date(2020, 2, 1), date(2025, 2, 1)]},
                       index=[date(2019, 12, 31), date(2024, 12, 31)])
    splits = pd.Series([4.0], index=pd.to_datetime(["2022-06-10"]).tz_localize("America/New_York"))
    adj = split_adjust_eps(eps, splits)
    assert adj.tolist() == [2.5, 12.0]


def five_years(vals, start=2021):
    return pd.DataFrame({"eps": vals, "filed": [date(y + 1, 2, 15) for y in range(start, start + len(vals))]},
                        index=[date(y, 12, 31) for y in range(start, start + len(vals))])


TODAY = date(2026, 10, 2)


def test_evaluate_uses_average_eps_of_last_n_years():
    eps = five_years([1, 2, 3, 4, 5, 6], start=2020)  # last 5 -> avg 4
    res = evaluate_earnings(eps, None, price=40.0, criteria=Criteria(), today=TODAY)
    assert res["status"] == "ok" and res["avg_eps"] == 4 and res["pe"] == 10


def test_evaluate_rejects_any_negative_year():
    eps = five_years([5, 5, -0.1, 5, 5])
    assert evaluate_earnings(eps, None, 10.0, Criteria(), today=TODAY)["status"].startswith("negative")


def test_evaluate_rejects_high_pe_and_short_history_and_stale():
    c = Criteria()
    assert evaluate_earnings(five_years([1] * 5), None, 15.0, c, today=TODAY)["status"].startswith("P/E")
    assert evaluate_earnings(five_years([1] * 4), None, 5.0, c, today=TODAY)["status"].startswith("fewer")
    assert evaluate_earnings(five_years([1] * 5, start=2018), None, 5.0, c, today=TODAY)["status"] == "EPS data is stale"


def test_current_ratio_prefers_latest_quarter():
    class T:
        quarterly_balance_sheet = pd.DataFrame(
            {pd.Timestamp("2026-06-30"): [300.0, 100.0], pd.Timestamp("2026-03-31"): [100.0, 100.0]},
            index=["Current Assets", "Current Liabilities"])
        balance_sheet = pd.DataFrame()
    assert screener.get_current_ratio(T()) == (3.0, "2026-06-30")


def test_run_screen_end_to_end(monkeypatch):
    universe = pd.DataFrame({
        "ticker": ["GOOD", "LOWCR", "LOSS", "PRICY", "BRK.B"],
        "name": ["Good", "LowCR", "Loss", "Pricy", "Berkshire"],
        "sector": ["X"] * 5, "cik": [1, 2, 3, 4, 5],
    })
    universe["yf_ticker"] = universe["ticker"].str.replace(".", "-", regex=False)
    monkeypatch.setattr(screener, "get_sp500_constituents", lambda: universe)
    prices = pd.Series({"GOOD": 50.0, "LOWCR": 50.0, "LOSS": 50.0, "PRICY": 500.0, "BRK-B": 60.0})
    monkeypatch.setattr(screener, "get_prices_and_splits", lambda t, y: (prices, {}))
    first = date.today().year - 5  # latest fiscal year = last calendar year
    eps = {1: [5] * 5, 2: [5] * 5, 3: [5, 5, -1, 5, 5], 4: [5] * 5, 5: [6] * 5}
    monkeypatch.setattr(screener, "fetch_company_facts",
                        lambda cik, s, l: facts([fy(first + i, v) for i, v in enumerate(eps[cik])]))
    cr = {"GOOD": (2.5, "2026-06-30"), "LOWCR": (1.5, "2026-06-30"), "BRK-B": (None, None)}
    monkeypatch.setattr(screener.yf, "Ticker", lambda t: t)
    monkeypatch.setattr(screener, "get_current_ratio", lambda t: cr[t])

    df, statuses = screener.run_screen(Criteria(), "test test@example.com")
    assert df["Ticker"].tolist() == ["GOOD"]
    assert df["P/E"].iloc[0] == pytest.approx(10.0)
    assert statuses["LOWCR"].startswith("current ratio")
    assert statuses["LOSS"].startswith("negative")
    assert statuses["PRICY"].startswith("P/E")
    assert statuses["BRK.B"].startswith("no current ratio")
    screener.print_report(df, statuses, Criteria())
