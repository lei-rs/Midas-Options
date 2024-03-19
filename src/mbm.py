import polars as pl
from polars import LazyFrame

from .helpers import prep_quotes


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


def _generate_mbm(quotes: LazyFrame) -> LazyFrame:
    quotes = quotes.filter(pl.col("type").ne("FT")).select(
        "time", "symbol", "bid_price", "ask_price", "bid_size", "condition"
    )
    quotes = (
        _add_floor_rows(quotes)
        .with_columns(
            pl.col("time").diff().shift(-1).alias("dif"),
        )
        .group_by(pl.col("time").dt.truncate("1m"))
        .agg(
            [pl.last("symbol", "bid_price", "ask_price")]
            + [
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


def generate_mbm(in_path: str) -> pl.DataFrame:
    df = pl.scan_parquet(in_path)
    df = prep_quotes(df)
    return _generate_mbm(df).collect()
