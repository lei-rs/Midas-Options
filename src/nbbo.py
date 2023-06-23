from typing import List, Dict

import numpy as np
import polars as pl


EXCHANGES = ['A', 'B', 'C', 'D', 'E', 'H', 'I', 'J', 'M', 'N', 'P', 'Q', 'T', 'W', 'X', 'Z']
QUOTES_COL_TO_REGEX = {
    'bids': '^c8.*$',
    'asks': '^c12.*$',
    'bid_sizes': '^c7.*$',
    'ask_sizes': '^c11.*$',
    'conditions': '^c15.*$',
}


class NBBO:
    def __init__(self, length):
        columns = [
            'bids',
            'asks',
            'bid_sizes',
            'ask_sizes',
            'bid_exchange',
            'ask_exchange',
            'bid_condition',
            'ask_condition',
        ]

        self.length = length
        self.data = {c: [None] * length for c in columns}
        self.i = 0

    @staticmethod
    def _proc_col(col: np.ndarray, idx: List, length: int):
        return col[np.arange(length), idx]

    def update(self, data: dict, bid_idx: List, ask_idx: List):
        length = len(bid_idx)
        for k, v in data.items():
            idx = bid_idx if k.startswith('bid') else ask_idx
            data = [EXCHANGES[i] for i in idx] if k.endswith('exchange') else self._proc_col(v, idx, length)
            self.data[k][self.i:self.i+length] = data
        self.i += length

    def to_dataframe(self):
        return pl.DataFrame(self.data)


def _slice_to_arrays(s: pl.DataFrame) -> Dict:
    s = {k: s.select(pl.col(v)).to_numpy() for k, v in QUOTES_COL_TO_REGEX.items()}
    con = s.pop('conditions')
    s.update({'bid_condition': con, 'ask_condition': con, 'bid_exchange': None, 'ask_exchange': None})
    return s


def _get_info(s: pl.DataFrame):
    s = _slice_to_arrays(s)
    bid_idx = np.nanargmax(s['bids'], axis=1)
    ask_idx = np.nanargmin(s['asks'], axis=1)
    return s, bid_idx, ask_idx


def _generate_nbbo(quotes: pl.DataFrame):
    length = len(quotes)
    nbbo = NBBO(length)
    for s in quotes.iter_slices(10000):
        nbbo.update(*_get_info(s))
    return nbbo


def _pivot_and_fill(quotes: pl.DataFrame):
    quotes = quotes.pivot(
        values=['c7', 'c8', 'c11', 'c12', 'c15'],
        index='c2',
        columns='c10',
        aggregate_function='first'
    ).lazy().select(pl.all().forward_fill())
    quotes = quotes.with_columns([
        pl.col('^(c7|c11).*$').fill_null(0),
    ])
    quotes = quotes.filter(
        (pl.all(pl.col('^c7.*$') == 0) | pl.all(pl.col('^c11.*$') == 0)).is_not()
    )
    quotes = quotes.with_columns([
        pl.col('^c8.*$').map_dict({0.0: None}, default=pl.first()),
        pl.col('^c12.*$').map_dict({0.0: None}, default=pl.first()),
        pl.col('^c15.*$').fill_null('K'),
    ])
    return quotes.collect().select(sorted(quotes.columns))


def generate_nbbo(quotes: pl.DataFrame):
    quotes = _pivot_and_fill(quotes)
    time = quotes['c2'].cast(pl.Time)
    nbbo = _generate_nbbo(quotes).to_dataframe()
    nbbo = nbbo.with_columns(time.alias('time'))
    return nbbo
