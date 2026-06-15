from __future__ import annotations

import pandas as pd


def make_walk_forward_windows(
    dates: pd.DatetimeIndex,
    train_years: int = 3,
    test_months: int = 6,
) -> list[dict[str, pd.Timestamp]]:
    clean_dates = pd.DatetimeIndex(dates).sort_values().unique()
    if len(clean_dates) == 0:
        return []

    windows = []
    cursor = clean_dates[0]
    last_date = clean_dates[-1]
    while cursor < last_date:
        train_end = cursor + pd.DateOffset(years=train_years)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_start > last_date:
            break
        windows.append(
            {
                "train_start": pd.Timestamp(cursor),
                "train_end": pd.Timestamp(train_end),
                "test_start": pd.Timestamp(test_start),
                "test_end": pd.Timestamp(min(test_end, last_date)),
            }
        )
        cursor = cursor + pd.DateOffset(months=test_months)
    return windows
