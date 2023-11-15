from datetime import timedelta
from typing import Set, Optional

import numpy as np
import pandas as pd
import polars as pl
from polars import DataFrame, Series

from src.helpers import cast_quotes, cast_trades


def _pivot_and_fill(quotes: pl.DataFrame):
    quotes = quotes.pivot(
        values=['c7', 'c8', 'c11', 'c12', 'c15'],
        index='c2',
        columns='c10',
        aggregate_function='last'
    ).lazy().select(pl.all().forward_fill())
    quotes = quotes.with_columns([
        pl.col('^(c7|c11).*$').fill_null(0),
    ])
    quotes = quotes.filter(
        (pl.all(pl.col('^c7.*$') == 0) | pl.all(pl.col('^c11.*$') == 0)).is_not()
    )
    quotes = quotes.with_columns([
        pl.col('^c8.*$').map_dict({None: 0.0}, default=pl.first()),
        pl.col('^c12.*$').map_dict({0.0: np.inf, None: np.inf}, default=pl.first()),
        pl.col('^c15.*$').fill_null('K'),
    ])
    quotes = quotes.with_columns([
        pl.max(pl.col('^c8.*$')).alias('nbb'),
        pl.min(pl.col('^c12.*$')).alias('nbo'),
    ])
    return quotes.collect().select(sorted(quotes.columns))


def generate_nbbo(quotes: pl.DataFrame):
    quotes = _pivot_and_fill(quotes)

    bid_cols = [col for col in quotes.columns if col.startswith('c8_')]
    ask_cols = [col for col in quotes.columns if col.startswith('c12_')]

    quotes = quotes.lazy()

    for col in bid_cols:
        quotes = quotes.with_columns(
            [pl.when(pl.col(col) == pl.col('nbb')).then(1).otherwise(0).alias(f'{col[-1]}_nbb_ind')]
        )

    for col in ask_cols:
        quotes = quotes.with_columns(
            [pl.when(pl.col(col) == pl.col('nbo')).then(1).otherwise(0).alias(f'{col[-1]}_nbo_ind')]
        )

    quotes = quotes.select([
        pl.col('c2').cast(pl.Time),
        pl.col('nbb'),
        pl.col('nbo'),
        pl.col('^c15.*$'),
        pl.col('^.*_nbb_ind$'),
        pl.col('^.*_nbo_ind$')
    ])

    return quotes.collect()


def time_at_nbbo(quotes: pl.DataFrame):
    quotes = quotes.with_columns(pl.date(1970, 1, 1).dt.combine(pl.col('c2')).alias('c2'))
    rng = pl.date_range(
        quotes.select(pl.first('c2').dt.truncate('1m')).item(),
        quotes.select(pl.last('c2').dt.truncate('1m')).item() + timedelta(minutes=1),
        '1m',
        eager=True,
    )[1:]
    rng = rng.filter(~rng.is_in(quotes['c2']))
    times = pl.DataFrame({'c2': rng})
    quotes = pl.concat([quotes.lazy(), times.lazy()], how='diagonal').sort('c2')
    quotes = quotes.select(pl.all().forward_fill())

    frac_nbbo = quotes.select([
        pl.col('c2').diff().shift(-1).dt.epoch().alias('diff'),
        pl.col('c2').dt.truncate(timedelta(minutes=1)).alias('floor'),
        pl.col('^.*_nbb_ind$'),
        pl.col('^.*_nbo_ind$')
    ])

    frac_nbbo = frac_nbbo.with_columns([
        pl.col('^.*_nbb_ind$').mul(pl.col('diff')),
        pl.col('^.*_nbo_ind$').mul(pl.col('diff')),
    ])

    frac_nbbo = frac_nbbo.groupby('floor').agg([
        pl.col('^.*_nbb_ind$').sum() / 60000000,
        pl.col('^.*_nbo_ind$').sum() / 60000000,
    ]).collect()

    return frac_nbbo


class ExecutionAnalysis:
    def __init__(self, raw: pl.DataFrame, subset: Optional[Set] = None):
        self.nbbo_buckets = self._generate_nbbo(raw)
        raw = raw.with_columns(
            pl.when(pl.col('c10') == '-').then(pl.col('c12')).otherwise(pl.col('c10')).alias('c10'),
        )
        self.by_exchange = raw.partition_by('c10', as_dict=True)
        for k, v in self.by_exchange.items():
            if subset is not None and k not in subset:
                continue
            v = v.sort('c2').with_columns(
                pl.arange(0, v.height).alias('index')
            )
            self.by_exchange[k] = cast_quotes(v)

    def find_q6(self, idx: int, direction: str, ex: str):
        raw = self.by_exchange[ex]
        first = q6 = raw[idx - 1]
        for i in range(-1, idx - 1, -1):
            curr = raw[i]
            if curr['c8'] == 'FT':
                continue
            elif direction == 's' and first['c12', 'c15'] == curr['c12', 'c15']:
                q6 = curr
            elif direction == 'b' and first['c8', 'c15'] == curr['c8', 'c15']:
                q6 = curr
            else:
                break
        return q6.to_numpy().flatten()[[1, 6, 7, 10, 11, 14]]

    @staticmethod
    def _get_directions(trades: DataFrame, raw: DataFrame) -> Series:
        fq_idxs = (trades[:, 'index'] - 1).to_series().to_list()
        prices = trades[:, 'c9'].to_series()
        bids = raw[fq_idxs, 'c8'].to_series()
        asks = raw[fq_idxs, 'c12'].to_series()
        return (
            pl.when(prices.eq(bids)).then('b')
            .when(prices.eq(asks)).then('s')
        )

    @staticmethod
    def _generate_nbbo(raw: pl.DataFrame) -> pd.DataFrame:
        raw = raw.filter(pl.col('c1') == 'F@')
        raw = cast_quotes(raw)
        raw = generate_nbbo(raw)
        return time_at_nbbo(raw)

    def _exchange_report(self, ex: str):
        raw = self.by_exchange[ex]
        info = []
        trades = raw.filter(pl.col('c1').eq('FT') & pl.col('c13').is_in({'I', 'S'}))
        trades = cast_trades(trades)
        trades = trades.with_columns([
            self._get_directions(trades, raw).alias('dir'),
        ])

        for trade in trades.iter_rows():
            info.append(self.find_q6(trade[-2], trade[-1], ex))
        info = np.asarray(info)
        return np.concatenate([trades.to_numpy(), info], axis=1)
