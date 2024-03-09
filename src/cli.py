import cmd
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path

import pandas
import polars as pl
from tqdm import tqdm

from .turning import generate_turning

PROC_FN = {
    "turning": generate_turning,
}


def save_result(df: pandas.DataFrame | pl.DataFrame, path_in: str, path_out: str):
    path_in = path_in.split("/")
    date = path_in[-3]
    under = path_in[-2]
    path_out = Path(path_out) / f"{date}_{under}.csv"

    if not os.path.exists(path_out):
        f = open(path_out, "w")
        f.write(",".join(df.columns) + "\n")
        f.close()

    if isinstance(df, pandas.DataFrame):
        df.to_csv(path_out, index=False, header=False, mode="a")
    elif isinstance(df, pl.DataFrame):
        f = open(path_out, "ab")
        df.write_csv(f, include_header=False)
        f.close()
    else:
        raise ValueError(f"Invalid dataframe type: {type(df)}")


def par_generate(fn):
    def par_fn(lock, path_in: str, path_out: str):
        try:
            df = fn(path_in)
            with lock:
                save_result(df, path_in, path_out)

        except Exception as e:
            return e

    return par_fn


class Midas(cmd.Cmd):
    def do_generate(self, args):
        try:
            args = args.split()
            if len(args) == 3:
                in_dir, out_dir, kind = args
                workers = None
            else:
                in_dir, out_dir, kind, workers = args
                workers = int(workers)
        except ValueError:
            raise ValueError("Invalid arguments")

        if not os.path.exists(in_dir):
            raise FileNotFoundError(f"Directory {in_dir} does not exist")
        if os.path.exists(out_dir):
            raise FileExistsError(f"Directory {out_dir} already exists")
        if kind not in PROC_FN:
            raise ValueError(f"Invalid report type to generate: {kind}")

        raw_files = list(Path(in_dir).rglob("*.parquet.gzip"))
        if not len(raw_files):
            raise FileNotFoundError(f"No files to process in {in_dir}")

        fn = PROC_FN[kind]
        if workers and workers > 1:
            pool = ProcessPoolExecutor()
            m = Manager()
            lock = m.Lock()
            for result in tqdm(
                pool.map(
                    par_generate(fn),
                    [lock] * len(raw_files),
                    [str(file) for file in raw_files],
                    [out_dir] * len(raw_files),
                ),
                total=len(raw_files),
            ):
                if result:
                    raise result
        else:
            for file in tqdm(raw_files):
                df = fn(str(file))
                save_result(df, str(file), out_dir)


if __name__ == "__main__":
    Midas().cmdloop()
