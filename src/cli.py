import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, get_context
from pathlib import Path
from itertools import repeat

import argparse
import pandas
import polars as pl
from tqdm import tqdm

from .mbm import generate_mbm
from .turning import generate_turning
from .helpers import polars_generate


def save_result(df: pandas.DataFrame | pl.DataFrame, path_in: str, path_out: str):
    path_in = path_in.split("/")
    date = path_in[-3]
    under = path_in[-2]
    path_out = Path(path_out) / f"{date}_{under}.csv"

    if not os.path.exists(path_out):
        f = open(path_out, "x")
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


PROC_FN = {
    "turning": polars_generate(generate_turning),
    "mbm": polars_generate(generate_mbm),
}

PROC_FN += {
    "turning_mp": par_generate(PROC_FN["turning"]),
    "mbm_mp": par_generate(PROC_FN["mbm"]),
}


def iter_raw(in_dir: str):
    in_dir = Path(in_dir)
    for date in os.listdir(in_dir):
        for under in os.listdir(in_dir / date):
            for file in os.listdir(in_dir / date / under):
                yield str(in_dir / date / under / file)


def do_generate(in_dir: str, out_dir: str, kind: str, workers: int = None):
    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Directory {in_dir} does not exist")
    os.makedirs(out_dir, exist_ok=True)
    if kind not in PROC_FN:
        raise ValueError(f"Invalid report type to generate: {kind}")

    raw_files = list(iter_raw(in_dir))
    if not len(raw_files):
        raise FileNotFoundError(f"No files to process in {in_dir}")

    if workers and workers > 1:
        fn = PROC_FN[kind + "_mp"]
        context = get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)
        m = Manager()
        lock = m.Lock()
        for result in tqdm(
            pool.map(
                fn,
                repeat(lock, len(raw_files)),
                raw_files,
                repeat(out_dir, len(raw_files)),
            ),
            total=len(raw_files),
        ):
            if result:
                raise result
    else:
        fn = PROC_FN[kind]
        for file in tqdm(raw_files):
            df = fn(str(file))
            save_result(df, str(file), out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Midas Options CLI")
    parser.add_argument(
        "in_dir",
        type=str,
        help="Directory containing the raw OPRA data",
    )
    parser.add_argument(
        "out_dir",
        type=str,
        help="Directory to save the generated reports",
    )
    parser.add_argument(
        "kind",
        type=str,
        help="Type of report to generate",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use for multiprocessing",
    )
    args = parser.parse_args()
    do_generate(args.in_dir, args.out_dir, args.kind, args.workers)
