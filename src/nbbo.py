from typing import Set, Optional, Dict

import polars as pl
from polars import DataFrame, LazyFrame


def _pivot_and_fill(quotes: DataFrame) -> LazyFrame:
    quotes = quotes.select(
        [
            "seq",
            "bid_size",
            "bid_price",
            "ask_size",
            "ask_price",
            "exchange",
        ]
    ).pivot(
        ["bid_size", "bid_price", "ask_size", "ask_price"],
        "seq",
        "exchange",
    )
    return (
        quotes.lazy()
        .select(pl.all().forward_fill())
        .with_columns(
            [
                pl.col("^bid_price.*$").fill_null(0),
                pl.col("^ask_price.*$").fill_null(float("inf")),
            ]
        )
        .select(sorted(quotes.columns))
    )


def generate_nbbo(quotes: DataFrame) -> LazyFrame:
    quotes = quotes.filter(pl.col("type") == "F@").with_columns(
        pl.when(pl.col("ask_price") == 0.).then(float("inf")).otherwise(pl.col("ask_price")).alias("ask_price"),
    )

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
            pl.concat_str(
                pl.when(pl.col("^bid_price.*$") == pl.col("nbb")).then(1).otherwise(0)
            ).alias("nbb_ex"),
            pl.concat_str(
                pl.when(pl.col("^ask_price.*$") == pl.col("nbo")).then(1).otherwise(0)
            ).alias("nbo_ex"),
        ]
    )

    # Select final columns
    quotes = quotes.select(
        [
            pl.col("seq").shift(-1),
            pl.col("nbb"),
            pl.col("nbo"),
            pl.col("nbb_ex"),
            pl.col("nbo_ex"),
        ]
    )

    return quotes


class EquityReport:
    def __init__(self, df: LazyFrame):
        df = (
            df.filter(
                pl.col("type").eq("F@")
                | (
                    pl.col("type").eq("FT")
                    & pl.col("condition").is_in({"I", "S", "a", "j"})
                )
            )
        ).collect()
        self.df = df.lazy().sort("exchange", "seq").with_row_index().collect()
        self.nbbo = generate_nbbo(df).with_columns(pl.col("seq").shift(-1)).collect()
        self.def_i = len(self.df)

    def check_trade(self, trade_i: int):
        t_price = self.df.item(trade_i, "bid_price")
        t_size = self.df.item(trade_i, "bid_size")
        t_ex = self.df.item(trade_i, "exchange")

        if (
            self.df.item(trade_i - 1, "type") == "FT"
            or self.df.item(trade_i + 1, "type") == "FT"
            or self.df.item(trade_i - 1, "exchange") != t_ex
            or self.df.item(trade_i + 1, "exchange") != t_ex
        ):
            return False

        bs1, bp1, as1, ap1 = self.df.row(trade_i - 1)[5:9]
        bs2, bp2, as2, ap2 = self.df.row(trade_i + 1)[5:9]
        return (
            (t_price == bp1 == bp2 and bs2 + t_size == bs1)
            or (t_price == ap1 == ap2 and as2 + t_size == as1)
            or (t_price != bp2 and (t_size, t_price) == (bs1, bp1))
            or (t_price != ap2 and (t_size, t_price) == (as1, ap1))
        )

    def find_q1(self, trade_i: int):
        trade_price = self.df.item(trade_i, "bid_price")
        trade_ex = self.df.item(trade_i, "exchange")
        nasdaq = {"H", "I", "J", "Q", "T", "X"}
        limit = 1
        if trade_ex in {"C", "E", "W", "Z"}.union(nasdaq):
            limit = 2
        quotes_found = 0

        for i in range(trade_i - 1, -1, -1):
            if self.df.item(i, "exchange") != trade_ex:
                return None
            if self.df.item(i, "type") == "FT":
                continue
            quotes_found += 1
            """
            if trade_ex in nasdaq:
                bid_price = self.df.item(i, "bid_price")
                ask_price = self.df.item(i, "ask_price")
                if trade_price == bid_price or trade_price == ask_price:
                    return i
            """
            if quotes_found == limit:
                return i
        return None

    def find_q5(self, trade_i: int, q1_i: int):
        trade_price = self.df.item(trade_i, "bid_price")
        bid_price = self.df.item(q1_i, "bid_price")
        ask_price = self.df.item(q1_i, "ask_price")
        first_cond = self.df.item(q1_i, "condition")
        if first_cond == "A":
            expected_cond = {"A"}
        else:
            expected_cond = {"C", first_cond}

        if trade_price == bid_price:
            side = "b"
        elif trade_price == ask_price:
            side = "a"
        else:
            return self.def_i, self.def_i

        q5_i = q1_i
        prev = self.df.row(q5_i)
        for i in range(q5_i - 1, -1, -1):
            curr = self.df.row(i)
            if curr[9] != prev[9]:
                return -1, -1
            if curr[1] == "FT":
                continue
            if (
                curr[10] not in expected_cond
                or (side == "b" and (curr[6] != prev[6]))
                or (side == "a" and (curr[8] != prev[8]))
            ):
                return i, prev[0]
            else:
                prev = curr
        return self.def_i, self.def_i

    def others(self, q_i: int):
        ret = []
        ex = self.df.item(q_i, "exchange")

        for i in range(q_i - 1, -1, -1):
            if self.df.item(i, "exchange") != ex:
                break
            if self.df.item(i, "type") == "F@":
                ret.append(i)
            if len(ret) == 5:
                break
        if len(ret) < 5:
            ret.extend([self.def_i] * (5 - len(ret)))
        ret = ret[::-1]

        for i in range(q_i + 1, len(self.df)):
            if self.df.item(i, "exchange") != ex:
                break
            if self.df.item(i, "type") == "F@":
                ret.append(i)
            if len(ret) == 9:
                break
        if len(ret) < 9:
            ret.extend([self.def_i] * (9 - len(ret)))

        return ret

    def generate(self) -> DataFrame:
        before_q5 = []
        q5 = []
        others = [[] for _ in range(9)]

        for trade_i in (
            self.df.filter(pl.col("type").eq("FT")).select("index").to_series()
        ):
            q1_i = self.find_q1(trade_i)
            if q1_i is None:
                before_q5.append(self.def_i)
                q5.append(self.def_i)
                for l in others:
                    l.append(self.def_i)
                continue
            temp = self.find_q5(trade_i, q1_i)
            before_q5.append(temp[0])
            q5.append(temp[1])
            for i, j in enumerate(self.others(q1_i)):
                others[i].append(j)

        self.df.vstack(pl.DataFrame({c: None for c in self.df.columns}), in_place=True)

        trades = (
            self.df.lazy()
            .select(["type", "time", "bid_size", "bid_price", "exchange", "condition"])
            .filter(pl.col("type").eq("FT"))
            .drop("type")
            .collect()
        )
        trades.columns = [f"trade_{c}" for c in trades.columns]

        q5 = self.df.select(
            ["seq", "time", "bid_size", "bid_price", "ask_size", "ask_price", "condition"]
        )[q5, :]
        q5 = (
            q5.lazy().join(self.nbbo.lazy(), on="seq", how="left").drop("seq").collect()
        )
        q5.columns = [f"q5_{c}" for c in q5.columns]

        before_q5 = self.df.select(["time", "bid_size", "bid_price", "ask_size", "ask_price",
                                    "condition"
                                    ])[before_q5, :]
        before_q5.columns = [f"before_q5_{c}" for c in before_q5.columns]

        quotes = [trades, q5, before_q5]
        for offset, l in zip([-5, -4, -3, -2, -1, "+1", "+2", "+3", "+4"], others):
            temp = self.df.select(
                ["time", "bid_size", "bid_price", "ask_size", "ask_price", "condition"]
            )[l, :]
            temp.columns = [f"q_t{offset}_{c}" for c in temp.columns]
            quotes.append(temp)

        final = pl.concat(quotes, how="horizontal")
        return final


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
