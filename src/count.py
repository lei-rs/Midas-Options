import polars as pl
from polars import DataFrame


def count_cc(path: str) -> DataFrame:
    df = pl.read_parquet(path)
    symbol = df.item(0, "c6")
    temp = pl.DataFrame(
        {
            "c15": ["A", "B", "C", "O"],
            "count": [0, 0, 0, 0],
        }
    ).lazy()
    df = (
        df.lazy()
        .filter(pl.col("c1") == "F@")
        .select("c15")
        .groupby("c15")
        .agg(pl.count())
        .filter(pl.col("c15").is_in(["A", "B", "C", "O"]))
    )
    df = (
        df.join(temp, on="c15", how="outer")
        .fill_null(0)
        .sort("c15")
        .select(pl.col("count") + pl.col("count_right"))
        .collect()
    )
    df = (
        df.transpose(column_names=["A", "B", "C", "O"])
        .with_columns(
            [
                pl.Series("symbol", [symbol]),
            ]
        )
        .select(["date", "symbol", "A", "B", "C", "O"])
    )
    return df
