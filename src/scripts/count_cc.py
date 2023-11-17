import glob
import polars as pl
from polars import DataFrame

from src import get_trading_days


def count_cc(path: str, date: str) -> DataFrame:
    df = pl.read_parquet(path)
    symbol = df.item(0, 'c6')
    temp = pl.DataFrame({
        'c14': ['A', 'B', 'C', 'O'],
        'count': [0, 0, 0, 0],
    }).lazy()
    df = (
        df
        .lazy()
        .filter(pl.col('c1') == 'F@')
        .select('c14')
        .groupby('c14')
        .agg(pl.count())
    )
    df = (
        df.join(temp, on='c14', how='outer')
        .fill_null(0)
        .sort('c14')
        .select(pl.col('count') + pl.col('count_right'))
        .collect()
    )
    df = df.transpose(column_names=['A', 'B', 'C', 'O']).with_columns([
        pl.Series('symbol', [symbol]),
        pl.Series('date', [date]),
    ]).select(['date', 'symbol', 'A', 'B', 'C', 'O'])
    return df


if __name__ == '__main__':
    dir = "./scratch/spxw_nov"
    dates = get_trading_days(20210104, 20210108)
    f = open("counts.csv", "a")
    f.write("date,symbol,A,B,C,O\n")

    for date in dates:
        for path in glob.glob(f"{dir}/{date}/*.parquet"):
            df = count_cc(path, date)
            df.write_csv(f, has_header=False)
