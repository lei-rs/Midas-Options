from datetime import timedelta
from typing import Set, Optional, Dict

import polars as pl
from polars import DataFrame, Series, LazyFrame


def _pivot_and_fill(quotes: LazyFrame) -> LazyFrame:
    """
    Pivot the quotes table by exchange and fill nulls
    :param quotes:
    :return:
    """
    quotes = (
        (
            quotes.collect()
            .pivot(
                values=["bid_size", "bid_price", "ask_size", "ask_price", "condition"],
                index="time",
                columns="exchange",
                aggregate_function="last",
            )
            .lazy()
            .select(pl.all().forward_fill())
        )
        .with_columns(
            [
                pl.col("^(bid_size|ask_size).*$").fill_null(0),
                pl.col("^condition.*$").fill_null("K"),
            ]
        )
        .filter(pl.all_horizontal(pl.col("^(bid_size|ask_size).*$") != 0))
    )
    return quotes.select(sorted(quotes.columns))


def generate_nbbo(quotes: LazyFrame) -> LazyFrame:
    """
    Generate NBBO indicators for each exchange
    :param quotes:
    :return:
    """
    quotes = quotes.filter(pl.col("type") == "F@")

    # Pivot and fill nulls
    quotes = _pivot_and_fill(quotes)

    # Calculate NBBO
    quotes = quotes.with_columns(
        [
            pl.max_horizontal("^bid_price.*$").alias("nbb"),
            pl.min_horizontal("^ask_price.*$").alias("nbo"),
            pl.col("^condition.*$"),
        ]
    )

    # Calculate NBBO indicators
    quotes = quotes.with_columns(
        [
            pl.when(pl.col("^bid_price.*$") == pl.col("nbb"))
            .then(1)
            .otherwise(0)
            .name.map(lambda c: f"{c[-1]}_nbb_ind"),
            pl.when(pl.col("^ask_price.*$") == pl.col("nbo"))
            .then(1)
            .otherwise(0)
            .name.map(lambda c: f"{c[-1]}_nbo_ind"),
        ]
    )

    quotes = quotes.select(
        [
            pl.col("time"),
            pl.col("nbb"),
            pl.col("nbo"),
            pl.col("^condition.*$"),
            pl.col("^.*_nbb_ind$"),
            pl.col("^.*_nbo_ind$"),
        ]
    )

    return quotes


def _get_directions(trades: DataFrame, raw: DataFrame) -> Series:
    """
    Get the direction of each trade
    :param trades:
    :param raw:
    :return:
    """
    fq_idxs = (trades[:, "index"] - 1).to_list()
    trade_price = trades[:, "bid_price"]
    bids = raw[fq_idxs, "bid_price"]
    asks = raw[fq_idxs, "ask_price"]
    return pl.select(
        pl.when(trade_price == bids)
        .then(pl.lit("b"))
        .when(trade_price == asks)
        .then(pl.lit("a"))
    ).to_series()


def _find_q1(df: DataFrame, tidx: int) -> int:
    ex = df.item(tidx, "exchange")
    if ex in {"N"}:
        return tidx - 1


def _find_q5(df: DataFrame, q1_idx: int, direction: str) -> None | int:
    """
    Finds quote type 5. The first quote posted for a given trade.
    :param df:
    :param q1_idx: The index of the quote corresponding to the trade.
    :param direction: The direction of the trade.
    :return:
    """
    q5 = q1_idx
    first = df[q5]
    for i in range(-1, q5, -1):
        curr = df[i]
        if curr["type"] == "FT":
            continue
        elif (
            direction == "a"
            and first["ask_price", "condition"] == curr["ask_price", "condition"]
        ) or (
            direction == "b"
            and first["bid_price", "condition"] == curr["bid_price", "condition"]
        ):
            q5 = i
        else:
            break

    if df.item(q5, "type") == "FT":
        return None
    return q5 - 1


def _se_analyze_ex(ex: DataFrame, nbbo: DataFrame) -> DataFrame:
    nbbo = nbbo.with_row_index()
    time_to_idx = dict(nbbo.select("time", "index").iter_rows())
    ex = ex.sort("time").with_row_index()
    trades = ex.filter(pl.col("type").eq("FT") & pl.col("condition").is_in({"I", "S"}))
    trades = trades.with_columns(
        [
            _get_directions(trades, ex).alias("direction"),
        ]
    )

    info = [None] * len(trades)
    for i, trade in enumerate(trades.select("index", "direction").iter_rows()):
        q1_idx = _find_q1(ex, trade[0])
        q5_idx = _find_q5(ex, q1_idx, trade[1])
        if q5_idx is not None:
            q5 = list(ex.row(q5_idx))
        else:
            q5 = [None] * len(ex.columns)
        q5_time = ex.item(q5_idx, "time") if q5_idx is not None else None
        if q5_time in time_to_idx:
            nbbo_idx = time_to_idx[q5_time]
            nbbo_row = list(nbbo.row(nbbo_idx))
        else:
            nbbo_row = [None] * len(nbbo.columns)
        info[i] = q5 + nbbo_row

    q5_cols = [f"q5_{c}" for c in ex.columns]
    cols = q5_cols + nbbo.columns
    info = DataFrame(info)
    info.columns = cols
    info = info.drop(
        [
            "q5_index",
            "time",
            "index",
        ]
    )
    trades = trades.drop(["index", "ask_size", "ask_price"]).rename(
        {"bid_size": "size", "bid_price": "price"}
    )
    final = pl.concat([trades, info], how="horizontal")
    return final


def analyze_executions(raw: LazyFrame, subset: Optional[Set] = None) -> Dict:
    nbbo = generate_nbbo(raw).collect()
    if subset is not None:
        raw = raw.filter(pl.col("exchange").is_in(subset))
    by_ex = raw.collect().partition_by("exchange", as_dict=True)
    by_ex = {k: _se_analyze_ex(v, nbbo) for k, v in by_ex.items()}
    return by_ex


def time_at_nbbo(quotes: pl.DataFrame):
    quotes = quotes.with_columns(
        pl.date(1970, 1, 1).dt.combine(pl.col("c2")).alias("c2")
    )
    rng = pl.date_range(
        quotes.select(pl.first("c2").dt.truncate("1m")).item(),
        quotes.select(pl.last("c2").dt.truncate("1m")).item() + timedelta(minutes=1),
        "1m",
        eager=True,
    )[1:]
    rng = rng.filter(~rng.is_in(quotes["c2"]))
    times = pl.DataFrame({"c2": rng})
    quotes = pl.concat([quotes.lazy(), times.lazy()], how="diagonal").sort("c2")
    quotes = quotes.select(pl.all().forward_fill())

    frac_nbbo = quotes.select(
        [
            pl.col("c2").diff().shift(-1).dt.epoch().alias("diff"),
            pl.col("c2").dt.truncate(timedelta(minutes=1)).alias("floor"),
            pl.col("^.*_nbb_ind$"),
            pl.col("^.*_nbo_ind$"),
        ]
    )

    frac_nbbo = frac_nbbo.with_columns(
        [
            pl.col("^.*_nbb_ind$").mul(pl.col("diff")),
            pl.col("^.*_nbo_ind$").mul(pl.col("diff")),
        ]
    )

    frac_nbbo = (
        frac_nbbo.groupby("floor")
        .agg(
            [
                pl.col("^.*_nbb_ind$").sum() / 60000000,
                pl.col("^.*_nbo_ind$").sum() / 60000000,
            ]
        )
        .collect()
    )

    return frac_nbbo
