import datetime as dt

import polars as pl


def check_dir(tx_price, midpoint):
    if tx_price > midpoint:
        return 'b'
    elif tx_price < midpoint:
        return 's'
    elif tx_price == midpoint:
        return 'm'


def convert_time(block_time: str):
    if block_time is None:
        return None
    time = dt.datetime.fromtimestamp(int(block_time[0:5]))
    time = time + dt.timedelta(hours=5)
    return str(time.time()) + '.' + block_time[5:]


def get_trading_days(date_start, date_end):
    date_start = dt.datetime.strptime(date_start, "%Y%m%d")
    date_end = dt.datetime.strptime(date_end, "%Y%m%d")

    result = []
    while date_start <= date_end:
        if date_start.weekday() in (0, 1, 2, 3, 4):
            result.append(date_start.strftime("%Y%m%d"))

        date_start += dt.timedelta(days=1)

    return result


def cast_types(df: pl.DataFrame):
    return df.with_columns([
        pl.from_epoch(pl.col('c2'), time_unit='us'),
        pl.col('c7').cast(pl.UInt16, strict=False),
        pl.col('c8').cast(pl.Float32, strict=False),
        pl.col('c11').cast(pl.UInt16, strict=False),
        pl.col('c12').cast(pl.Float32, strict=False),
    ])
