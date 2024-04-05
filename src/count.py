import polars as pl
from polars import DataFrame


def count_cc(path: str) -> DataFrame:
    df = pl.read_parquet(path)
    symbol = df.item(0, "c6")
    df = (
        df.select("c1", "c15")
        .filter(pl.col("c1") == "F@")
        .groupby("c15")
        .agg(pl.count())
        .filter(pl.col("c15").is_in({"A", "B", "C", "O", "F"}))
        .transpose(column_names="c15")
    )
    cols = df.columns
    new_cols = []
    for c in ["A", "B", "C", "O", "F"]:
        if c not in cols:
            new_cols.append(pl.lit(0).alias(c))
    new_cols.append(pl.lit(symbol).alias("symbol"))
    df = df.with_columns(new_cols)
    return df.select(["symbol", "A", "B", "C", "O", "F"])
