# Midas Options CLI

# Installation

First set up the required python environment with the following commands:

```shell
conda create --name ENVNAME python=3.10
conda activate ENVNAME
pip install polars pandas fastparquet tqdm
```

Then clone the ``src`` directory into your home directory and you can start using the CLI.

# Downloading

Arguments:

- Output Directory:
  The directory where raw data should be stored.
- Symbol:
  The symbol for the underlying.
- Date Range:
  The inclusive range for which to download data, in the following format: YYYYMMDD-YYYYMMDD.
- Whitelist (Optional):
  A template string for the file names of symbols to download. The CLI will fill in the date where
  the curly braces are. For example if the symbol files
  are ``symbols/SPXW_20230101, symbols/SPXW_20230101, ...``, use ``symbols/SPXW_{}`` as the
  argument.
- Workers (Optional):
  A number indicating how many parralel processes to run.

Example:

```shell
python -m src.download ./output SPXW 20240101-20240102 --whitelist ./symbols/SPXW_{}.csv --workers 8
```

This means download raw SPXW data from Jan 1st 2024 to Jan 2nd 2024 (inclusive) into the ``output``
directory. We ignore symbols not in the ``symbols/SPXW_{date}.csv`` file. This runs 8 parallel
processes at a time.

# Processing
Arguments:
- Input Directory: Directory containing the raw OPRA data.
- Output Directory: The directory where raw data should be stored.
- Kind: Type of algorithm to run. Options:
  - ``turning``: Market turning
  - ``mbm``: Minute by minute
  - ``count``: Condition quote counts
  - ``index``: Index trade report

Example:

```shell
python -m src.proc ./raw_data ./raw_data_out mbm --workers 8
```

This means generate the minute by minute report for all the data in ``raw_data`` and save it
to ``raw_data_out`` using 8 parallel process at a time.

**Note**:
Every algorithm except the index trade report is written in polars. Polars is a multithreaded, so
this could be very CPU intensive (but also fast) when using multiple workers. You can prefix the
command with ``POLARS_MAX_THREADS=n`` like so:

```shell
POLARS_MAX_THREADS=n python -m src.proc ./raw_data ./raw_data_out mbm --workers m
```

to alter the CPU usage. In the above modified example, n*m threads will be used, which the Jupyter
server only has 64 of. 