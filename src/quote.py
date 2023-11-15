import polars as pl


def _add_floor_rows(quotes):
    bounds = quotes[[0, -1], 'c2'].select(pl.col('c2').dt.truncate('1m'))
    rng = pl.date_range(
        bounds[0, 'c2'],
        pl.select(bounds[-1, 'c2'] + pl.duration(minutes=1)).item(),
        '1m',
        eager=True,
    )[1:]
    rng = rng.filter(~rng.is_in(quotes['c2']))
    df2 = pl.DataFrame({'c2': rng})
    quotes = pl.concat([quotes.lazy(), df2.lazy()], how='diagonal').sort('c2')
    quotes = quotes.select(pl.all().forward_fill())
    return quotes


def generate_qr(quotes: pl.DataFrame):
    quotes = _add_floor_rows(quotes)

    quotes = quotes.with_columns([
        pl.col('c2').dt.truncate('1m').alias('floor'),
        pl.col('c2').diff().shift(-1).alias('dif')
    ]).collect()

    agg = (
        quotes
        .lazy()
        .groupby('floor', 'c15')
        .agg([pl.col('dif').sum() / pl.duration(minutes=1)])
        .collect()
        .pivot(values='dif', index='floor', columns='c15')
        .sort('floor')
    ).with_columns(
        pl.col('floor') + pl.duration(minutes=1),
    )

    idx = agg['floor'].to_list()
    quotes = quotes.filter(
        pl.col('c2').is_in(idx),
    )

    agg = pl.concat([
        agg.select([
            'floor',
            'A',
            'B',
            'C',
            'O'
        ]),
        quotes.select(
            pl.col('c6').alias('symbol'),
            pl.col('c7').alias('bid_size'),
            pl.col('c8').alias('bid_price'),
            pl.col('c11').alias('ask_size'),
            pl.col('c12').alias('ask_price'),
        )],
        how='horizontal').sort('floor')[:-1]

    return agg.with_columns([pl.col('floor').cast(pl.Time)])


