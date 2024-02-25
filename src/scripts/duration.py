import os

from src import get_trading_days, MPApply
from src.quote import find_turning_quotes
import polars as pl

DATA_DIR = "scratch/spxw_main"
OUT_DIR = "scratch/spxw_202309_out_test"


def generate(date: str, in_path: str):
    df = pl.scan_parquet(in_path)
    df = find_turning_quotes(df).collect()
    with open(f"{OUT_DIR}/{date}.csv", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        df.write_parquet(f)
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    args = []
    for date in get_trading_days("20230117", "20230120"):
        args += [(date, filename) for filename in os.listdir(f"{DATA_DIR}/{date}/SPXW")]

    MPApply(args).apply(generate, 8)
