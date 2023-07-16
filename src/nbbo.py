import numpy as np
import polars as pl


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
        pl.col('^(c7|c11).*$'),
        pl.col('^c15.*$'),
        pl.col('^.*_nbb_ind$'),
        pl.col('^.*_nbo_ind$')
    ])

    return quotes.collect()
