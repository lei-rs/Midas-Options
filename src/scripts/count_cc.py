import polars as pl


def count_cc(path: str) -> pl.DataFrame:
    df = pl.scan_parquet(path).filter(pl.col('c1') != 'F@')
    df = df.select('c14').groupby('c14').agg(pl.count())
    return df.collect()


if __name__ == '__main__':
    count_cc()
