import numpy as np
import numpy.typing as npt
import polars as pl


class QuoteReport:
    def __init__(self, quotes: pl.DataFrame | npt.NDArray):
        self.quotes = quotes
        if isinstance(quotes, np.ndarray):
            self.quotes = pl.from_numpy(quotes)
        self.quotes = self.quotes.filter(pl.col('c1') != 'FT')

    def _add_floor_rows(self):
        bounds = self.quotes[[0, -1], 'c2'].select(pl.col('c2').dt.truncate('1m'))
        rng = pl.date_range(
            bounds[0, 'c2'],
            pl.select(bounds[-1, 'c2'] + pl.duration(minutes=1)).item(),
            '1m',
            eager=True,
        )[1:]
        rng = rng.filter(~rng.is_in(self.quotes['c2']))
        df2 = pl.DataFrame({'c2': rng})
        quotes = pl.concat([self.quotes, df2], how='diagonal').sort('c2')
        self.quotes = quotes.select(pl.all().forward_fill())

    def _add_floor_cols(self):
        self.quotes = self.quotes.with_columns([
            pl.col('c2').dt.truncate('1m').alias('floor'),
        ])

    def _add_next_col(self):
        self.quotes = self.quotes.with_columns([
            pl.col('c2').diff().shift(-1).alias('dif'),
        ])

    def generate(self):
        self._add_floor_rows()
        self._add_floor_cols()
        self._add_next_col()
        agg = (
            self.quotes
            .lazy()
            .groupby('floor', 'c15')
            .agg([pl.col('dif').sum() / pl.duration(minutes=1)])
            .collect()
            .pivot(values='dif', index='floor', columns='c15')
            .sort('floor')
        )
        agg.with_columns(
            pl.col('floor') + pl.duration(minutes=1),
        )
        idx = agg['floor'].to_list()
        quotes = self.quotes.filter(
            pl.col('c2').is_in(idx),
        )
        agg = agg.select(['floor', 'A', 'B', 'C', 'O'])
        agg = pl.concat([agg, quotes.select(
            pl.col('c6').alias('symbol'),
            pl.col('c7').alias('bid_size'),
            pl.col('c8').alias('bid_price'),
            pl.col('c11').alias('ask_size'),
            pl.col('c12').alias('ask_price'),
        )], how='horizontal').sort('floor')
        agg = agg.with_columns([pl.col('floor').cast(pl.Time)])
        return agg
