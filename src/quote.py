import datetime

import polars as pl
from polars import LazyFrame, DataFrame


def _add_floor_rows(quotes: LazyFrame) -> LazyFrame:
    temp = quotes.select(pl.col("time").dt.truncate("1m")).collect()
    start = temp.item(0, "time")
    end = temp.item(-1, "time")
    rng = pl.date_range(
        start,
        (end + pl.duration(minutes=1)),
        "1m",
        eager=True,
    )[1:]
    time = pl.DataFrame({"time": rng}).lazy()
    quotes = (
        quotes.join(time, on="time", how="outer")
        .with_columns(pl.col("time").fill_null(pl.col("time_right")))
        .drop("time_right")
        .sort("time")
        .select(pl.all().forward_fill())
    )
    return quotes


def generate_qr(quotes: LazyFrame) -> LazyFrame:
    quotes = quotes.filter(pl.col("type").ne("FT")).select(
        "time", "bid_size", "condition"
    )
    quotes = (
        _add_floor_rows(quotes)
        .with_columns(
            pl.col("time").diff().shift(-1).alias("dif"),
        )
        .group_by(pl.col("time").dt.truncate("1m"))
        .agg(
            [
                pl.col("dif").filter(pl.col("condition").eq(col)).sum().alias(col)
                for col in ["A", "B", "C", "O"]
            ]
            + [
                pl.col("dif")
                .filter(pl.col("condition").eq(col) & pl.col("bid_size").gt(0))
                .sum()
                .alias(f"{col}_ts")
                for col in ["A", "B", "C", "O"]
            ]
        )
        .with_columns(
            [
                pl.col(col) / pl.duration(minutes=1)
                for col in ["A", "B", "C", "O", "A_ts", "B_ts", "C_ts", "O_ts"]
            ]
        )
    )

    return quotes


def find_turning_quotes(df: LazyFrame) -> LazyFrame:
    indices = (
        df.with_row_index()
        .select(
            [
                "index",
                "condition",
                pl.col("condition").shift(1).alias("prev_condition"),
            ]
        )
        .filter(
            (pl.col("prev_condition").eq("A") & pl.col("condition").ne("A"))
            | (pl.col("prev_condition").ne("C") & pl.col("condition").eq("C"))
        )
        .collect()[:, "index"]
    )
    after = df.collect()[indices.to_list(), :]
    after.columns = [f"after_{col}" for col in after.columns]
    before = df.collect()[(indices - 1).to_list(), :]
    before.columns = [f"before_{col}" for col in before.columns]
    df = pl.concat([before, after], how="horizontal").lazy()
    df = df.filter(
        (
            pl.col("after_condition").eq("B")
            & (pl.col("after_bid_price") > pl.col("before_bid_price"))
        )
        | (
            pl.col("after_condition").eq("O")
            & (pl.col("after_ask_price") < pl.col("before_ask_price"))
        )
        | (
            pl.col("after_condition").eq("C")
            & pl.col("before_condition").eq("B")
            & (pl.col("after_ask_price") < pl.col("before_ask_price"))
        )
        | (
            pl.col("after_condition").eq("C")
            & pl.col("before_condition").eq("O")
            & (pl.col("after_bid_price") > pl.col("before_bid_price"))
        )
    )
    return df
