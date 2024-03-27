import datetime as dt

import numpy as np
import pandas as pd

from .helpers import convert_time, check_dir


class IndexGenerator:
    def __init__(self, data):
        self.data = data.loc[
            (data["c1"] == "F@") | (data["c13"].isin(["I", "S"]))
        ].reset_index(drop=True)
        self.quotes = self.data.loc[self.data["c1"] == "F@"].reset_index(drop=True)
        self.quotes.iloc[:, [6, 10]] = self.quotes.iloc[:, [6, 10]].astype(int)
        self.quotes.iloc[:, [7, 11]] = self.quotes.iloc[:, [7, 11]].astype(float)
        self.quotes_ar = self.quotes.to_numpy()
        self.trades = self.data.loc[self.data["c13"].isin(["I", "S"])]
        self.trades.iloc[:, 7] = self.trades.iloc[:, 7].astype(int)
        self.trades.iloc[:, 8] = self.trades.iloc[:, 8].astype(float)
        self.trades = self.trades.reset_index().to_numpy()
        self.num_orders = len(self.quotes_ar)
        self.num_trades = len(self.trades)
        self.tr = IndexTradeReport()
        self.last_loc = 0
        self.check = set(self.trades[:, 0].flatten() - np.arange(0, self.num_trades))

    def check_tx(self, ind, tx_amount, tx_price):
        if ind + 1 == self.num_orders:
            return False

        tx = np.asarray([tx_amount, tx_price])
        bid = self.quotes_ar[ind, [6, 7]]
        ask = self.quotes_ar[ind, [10, 11]]
        next_bid = self.quotes_ar[ind + 1, [6, 7]]
        next_ask = self.quotes_ar[ind + 1, [10, 11]]

        if (np.array_equal(bid, tx) and tx[1] != next_bid[1]) or (
            np.array_equal(ask, tx) and tx[1] != next_ask[1]
        ):
            return True

        elif (tx[0] + next_bid[0] == bid[0] and (tx[1] == bid[1] == next_bid[1])) or (
            tx[0] + next_ask[0] == ask[0] and (tx[1] == ask[1] == next_ask[1])
        ):
            return True

        return False

    def match_tx(self, ind, tx_amount, tx_price):
        for i in range(ind, ind - 7, -1):
            if self.check_tx(i, tx_amount, tx_price):
                return i

        # Forward Search
        for i in range(ind + 1, self.num_orders):
            if i in self.check:
                break

            if self.check_tx(i, tx_amount, tx_price):
                return i

    def find_ind(self, ind, tx_amount, tx_price):
        new_ind = self.match_tx(ind, tx_amount, tx_price)

        if new_ind is None:
            return ind, 1

        else:
            return new_ind, 0

    def find_q2(self, ind):
        for i in range(ind, -1, -1):
            order = self.quotes_ar[i]

            if order[14] == "A":
                return order

        return [None] * 15

    def find_q3_q6(self, ind, direction):
        i = -1
        q3 = self.quotes_ar[ind]
        q6 = [None] * 15

        for i in range(ind - 1, -1, -1):
            if i == 0:
                q3 = self.quotes_ar[i]

            if direction == "s":
                curr = self.quotes_ar[ind, [7, 8]]
                prev = self.quotes_ar[i, [7, 8]]

            else:
                curr = self.quotes_ar[ind, [11, 12]]
                prev = self.quotes_ar[i, [11, 12]]

            if not np.array_equal(curr, prev):
                q3 = self.quotes_ar[i + 1]
                q6 = self.quotes_ar[i]

                break

        if i > -1:
            q6 = self.quotes_ar[i]

        return q3, q6

    def find_q4(self, ind):
        for i in range(ind + 1, self.num_orders):
            order = self.quotes_ar[i]

            if order[14] == "A":
                return order

        return [None] * 15

    def find_q5(self, ind):
        for i in range(ind + 1, self.num_orders):
            order = self.quotes_ar[i]

            if order[0] == "F@":
                return order

        return [None] * 15

    def generate_trade(self, ind, new_ind, tx_price, start, indicator, fin=None):
        quotes = {
            "q1": self.quotes_ar[new_ind],
            "q2": self.find_q2(new_ind),
            "q4": self.find_q4(new_ind),
            "q5": self.find_q5(new_ind),
        }

        q3, q6 = self.find_q3_q6(
            new_ind,
            check_dir(tx_price, (float(quotes["q1"][7]) + float(quotes["q1"][11])) / 2),
        )
        quotes = {**quotes, "q3": q3, "q6": q6}

        for offset in (-3, -2, -1, 0, 1, 2):
            if 0 <= new_ind + offset < self.num_orders:
                q = self.quotes_ar[ind + offset]
            else:
                q = [None] * 15
            quotes[f"q_t{str(offset)}"] = q

        if fin is None:
            tx = self.trades[start][1:]
            self.tr.add_trade(tx, quotes, indicator)

        else:
            for i in range(start, fin):
                tx = self.trades[i][1:]
                self.tr.add_trade(tx, quotes, indicator)

    def generate_tr(self):
        i = 0

        while i < self.num_trades:
            curr_trade = self.trades[i]
            curr_ind = curr_trade[0]
            tx_amount = curr_trade[8]
            tx_price = curr_trade[9]
            new_ind, indicator = self.find_ind(curr_ind - i - 1, tx_amount, tx_price)

            if (
                indicator
                and (i != self.num_trades - 1)
                and (curr_trade[0] == self.trades[i + 1][0] - 1)
            ):
                next_trade = self.trades[i + 1]
                curr_ind = curr_trade[0]
                agg_amount = curr_trade[8]
                tx_price = curr_trade[9]
                count = 1

                while (curr_trade[0] == next_trade[0] - 1) and (
                    curr_trade[9] == next_trade[9]
                ):
                    agg_amount += next_trade[8]
                    count += 1

                    if i + count == self.num_trades:
                        break

                    next_trade = self.trades[i + count]

                new_ind, indicator = self.find_ind(
                    curr_ind - i - 1, agg_amount, tx_price
                )
                self.last_loc = new_ind
                self.generate_trade(
                    curr_ind - i - 1, new_ind, tx_price, i, indicator, i + count
                )

                i += count

            else:
                self.last_loc = new_ind
                self.generate_trade(curr_ind - i - 1, new_ind, tx_price, i, indicator)
                i += 1

        return self.tr.finalize()


class IndexTradeReport:
    def __init__(self):
        self.quote_cols = [
            "bb",
            "bb_size",
            "bo",
            "bo_size",
            "quote_ts",
            "quote_condition",
        ]
        self.trade_cols = [
            "option",
            "indicator",
            "trade_time",
            "trade_size",
            "trade_price",
            "midpoint",
            "trade_code",
            "time_to_fill",
        ]
        self.quote_data = dict()
        self.trade_data = []

    def add_trade(self, tx, quotes, indicator):
        midpoint = (quotes["q1"][7] + quotes["q1"][11]) / 2

        t1 = quotes["q1"][1]
        t2 = quotes["q3"][1]

        if t1 is None or t2 is None:
            dif = None
        else:
            dif = str(
                (
                    dt.datetime.strptime(convert_time(t1), "%H:%M:%S.%f")
                    - dt.datetime.strptime(convert_time(t2), "%H:%M:%S.%f")
                ).total_seconds()
            )

        for k, v in quotes.items():
            if not k in self.quote_data:
                self.quote_data[k] = []
            self.quote_data[k].append(
                [v[7], v[6], v[11], v[10], convert_time(v[1]), v[14]]
            )

        self.trade_data.append(
            [tx[5], indicator, convert_time(tx[1]), tx[7], tx[8], midpoint, tx[12], dif]
        )

    def finalize(self):
        array = np.hstack(
            tuple(
                [
                    np.asarray(d)
                    for d in [self.trade_data] + list(self.quote_data.values())
                ]
            )
        )
        return pd.DataFrame(
            array,
            columns=self.trade_cols
            + [
                f"{prefix}_{col}"
                for prefix in self.quote_data.keys()
                for col in self.quote_cols
            ],
        )


def generate_index_report(in_path: str) -> pd.DataFrame:
    return IndexGenerator(pd.read_parquet(in_path)).generate_tr()
