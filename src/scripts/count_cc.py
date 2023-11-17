import glob
from tqdm import tqdm
import polars as pl
from polars import DataFrame


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
    files = glob.glob(f'{dir}/*.parquet.gzip', recursive=True)
    f = open("counts.csv", "a")
    f.write("date,symbol,A,B,C,O\n")
    for file in tqdm(files):
        print(file)
        date = file.split('/')[-3]
        df = count_cc(file, date)
        df.write_csv(f, has_header=False)
