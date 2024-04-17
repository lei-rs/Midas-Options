import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

from .helpers import get_trading_days

CHUNK = 10000
SYMBOL = None
OUT_DIR = None
SKIP_DIR = None


class ProcReader:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
        )

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise StopIteration
            else:
                return line.strip()


class Product:
    def __init__(self, product, data, date):
        self.product = product
        self.data = [data]
        self.date = date

    def append(self, twxm):
        self.data.append(twxm)

    def check(self):
        if len(self.data) > CHUNK:
            self._write()

    def write(self):
        if len(self.data) > 0:
            self._write()

    def _write(self):
        df = pd.DataFrame(self.data, columns=[f"c{i}" for i in range(1, 16)])
        symbol = self.data[0][5].split("_")[0]
        os.makedirs(f"{OUT_DIR}/{self.date}/{symbol}/", exist_ok=True)
        path = f"{OUT_DIR}/{self.date}/{symbol}/{self.product}.parquet.gzip"

        if not os.path.isfile(path):
            df.to_parquet(path, index=None, engine="fastparquet", compression="gzip")
        else:
            df.to_parquet(
                path, index=None, engine="fastparquet", compression="gzip", append=True
            )

        self.data = []


class SplitProducts:
    def __init__(self, twxm: ProcReader, date):
        self._twxm = twxm
        self.products = {}
        self.date = date
        self.whitelist = None
        if SKIP_DIR:
            whitelist = list(pd.read_csv(SKIP_DIR.format(date))["symbol"])
            self.whitelist = set(
                [s.replace("_", "").replace(" ", "") for s in whitelist]
            )

    def _save_data(self, product: str, row: str):
        if product in self.products:
            p = self.products[product]
            p.append(row)
        else:
            p = Product(product, row, self.date)
            self.products[product] = p

    def execute(self):
        for twxm_byte in self._twxm:
            twxm = twxm_byte.decode("utf-8").split(" ")
            product = twxm[5].replace("_", "")

            if self.whitelist and product in self.whitelist:
                self._save_data(product, twxm)
                self.products[product].check()

        for k, v in self.products.items():
            v.write()


def worker(symbol: str, date: str):
    twxm = ProcReader(f"twxm {date} opra {symbol}_*")
    SplitProducts(twxm, date).execute()


def download(date_range: str, workers: int = None):
    try:
        dates = get_trading_days(*date_range.split("-"))
    except ValueError:
        raise ValueError("Invalid date range format")

    args = zip([SYMBOL] * len(dates), dates)
    if workers:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = []
        for arg in args:
            results.append(executor.submit(worker, *arg))
        for res in tqdm(results):
            print(res.result())
        executor.shutdown(wait=True)
    else:
        for arg in tqdm(args):
            worker(*arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Midas Downloader")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to store the downloaded data",
    )
    parser.add_argument(
        "symbol",
        type=str,
        help="Underlying to download",
    )
    parser.add_argument(
        "date_range",
        type=str,
        help="Date range to download"
             "Format: YYYYMMDD-YYYYMMDD",
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="Skip dates in the format YYYYMMDD,YYYYMMDD",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use",
    )
    args = parser.parse_args()

    SYMBOL = args.symbol
    OUT_DIR = args.output_dir
    SKIP_DIR = args.skip
    download(args.date_range)
