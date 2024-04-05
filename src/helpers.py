import datetime as dt
from typing import Callable

import polars as pl
from polars import DataFrame, LazyFrame


def check_dir(tx_price, midpoint):
    if tx_price > midpoint:
        return "b"
    elif tx_price < midpoint:
        return "s"
    elif tx_price == midpoint:
        return "m"


def convert_time(block_time: str):
    if block_time is None:
        return None
    time = dt.datetime.fromtimestamp(int(block_time[0:5]))
    time = time + dt.timedelta(hours=5)
    return str(time.time()) + "." + block_time[5:]


def get_trading_days(date_start, date_end):
    date_start = dt.datetime.strptime(date_start, "%Y%m%d")
    date_end = dt.datetime.strptime(date_end, "%Y%m%d")

    result = []
    while date_start <= date_end:
        if date_start.weekday() in (0, 1, 2, 3, 4):
            result.append(date_start.strftime("%Y%m%d"))

        date_start += dt.timedelta(days=1)

    return result


def cast_quotes(df: pl.DataFrame):
    return df.with_columns(
        [
            pl.from_epoch(pl.col("c2"), time_unit="us"),
            pl.col("c7").cast(pl.UInt16, strict=False),
            pl.col("c8").cast(pl.Float32, strict=False).round(2),
            pl.col("c11").cast(pl.UInt16, strict=False),
            pl.col("c12").cast(pl.Float32, strict=False).round(2),
        ]
    )


def cast_trades(df: pl.DataFrame):
    return df.with_columns(
        [
            pl.from_epoch(pl.col("c2"), time_unit="us"),
            pl.col("c8").cast(pl.UInt16, strict=False),
            pl.col("c9").cast(pl.Float32, strict=False).round(2),
        ]
    )


def prep_quotes(df: LazyFrame) -> LazyFrame:
    swap_exprs = [
        pl.when(pl.col("c1") == "FT").then(new).otherwise(old).alias(old)
        for old, new in [
            ("c7", "c8"),
            ("c8", "c9"),
            ("c9", "c7"),
            ("c11", pl.lit(0)),
            ("c12", pl.lit(0)),
            ("c14", "c12"),
            ("c15", "c13"),
        ]
    ]
    df = df.with_columns(swap_exprs)
    df = df.select(
        "c1",
        "c2",
        "c3",
        "c6",
        "c7",
        "c8",
        "c11",
        "c12",
        "c14",
        "c15",
    )
    df = df.rename(
        {
            "c1": "type",
            "c2": "time",
            "c3": "seq",
            "c6": "symbol",
            "c7": "bid_size",
            "c8": "bid_price",
            "c11": "ask_size",
            "c12": "ask_price",
            "c14": "exchange",
            "c15": "condition",
        }
    )

    # Cast types
    df = df.with_columns(
        [
            # pl.from_epoch(pl.col("time"), time_unit="us"),
            pl.col("time").cast(pl.UInt64),
            pl.col("seq").cast(pl.UInt32),
            pl.col("bid_size").cast(pl.Float32).round(0).cast(pl.UInt16),
            pl.col("bid_price").cast(pl.Float32).round(2),
            pl.col("ask_size").cast(pl.Float32).round(0).cast(pl.UInt16),
            pl.col("ask_price").cast(pl.Float32).round(2),
        ]
    )
    return df
