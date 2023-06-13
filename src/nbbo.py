import numpy as np
import pandas as pd
import polars as pl


def _pivot_and_fill(quotes: pl.DataFrame):
    return quotes.pivot(
        values=['c7', 'c8', 'c11', 'c12', 'c15'],
        index='c2',
        columns='c10',
        aggregate_function='first'
    ).lazy().select(pl.all().forward_fill()).fill_nan(0).collect()


def generate_nbbo(quotes: pl.DataFrame):
    quotes = _pivot_and_fill(quotes)
    time = quotes['c2'].cast(pl.Time).to_list()
    pd_quotes: pd.DataFrame = quotes.to_pandas()
    bids = pd_quotes.filter(regex='c8').fillna(0).values
    asks = pd_quotes.filter(regex='c12').fillna(np.inf).values
    bid_idx = np.argmax(bids, axis=1)
    ask_idx = np.argmin(asks, axis=1)
    length = list(range(len(pd_quotes)))
    nbbo = pl.DataFrame({
        'time': time,
        'bid_size': pd_quotes.filter(regex='c7').values[length, bid_idx],
        'bid_price': pd_quotes.filter(regex='c8').values[length, bid_idx],
        'ask_size': pd_quotes.filter(regex='c11').values[length, ask_idx],
        'ask_price': pd_quotes.filter(regex='c12').values[length, ask_idx],
    })
    return nbbo, bids, asks
