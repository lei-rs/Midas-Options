import polars as pl
from polars import LazyFrame
from cli import polars_generate


@polars_generate
def generate_turning(df: LazyFrame) -> LazyFrame:
    indices = (
        df.filter(pl.col("type").eq("F@"))
        .with_row_index()
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
