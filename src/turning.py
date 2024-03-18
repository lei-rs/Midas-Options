import polars as pl
from polars import LazyFrame


def generate_turning(df: LazyFrame) -> LazyFrame:
    df = (
        df.filter(pl.col("type").eq("F@"))
        .drop(["type", "exchange"])
        .select(
            [pl.all().name.prefix("before_"), pl.all().shift(-1).name.prefix("after_")]
        )
        .drop_nulls()
        .filter(
            (
                pl.col("after_condition").is_in({"B", "C"})
                & (pl.col("after_bid_price") > pl.col("before_bid_price"))
            )
            | (
                pl.col("after_condition").is_in({"C", "O"})
                & (pl.col("after_ask_price") < pl.col("before_ask_price"))
            )
        )
    )
    return df
