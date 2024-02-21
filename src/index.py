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
        q1 = self.quotes_ar[new_ind]
        q2 = self.find_q2(new_ind)
        q3, q6 = self.find_q3_q6(
            new_ind, check_dir(tx_price, (float(q1[7]) + float(q1[11])) / 2)
        )
        q4 = self.find_q4(new_ind)
        q5 = self.find_q5(new_ind)
        q7 = [None] * 15
        q8 = self.quotes_ar[ind]

        if ind - 1 >= 0:
            q7 = self.quotes_ar[ind - 1]

        if fin is None:
            tx = self.trades[start][1:]
            self.tr.add_trade(tx, [q1, q2, q3, q4, q5, q6, q7, q8], indicator)

        else:
            for i in range(start, fin):
                tx = self.trades[i][1:]
                self.tr.add_trade(tx, [q1, q2, q3, q4, q5, q6, q7, q8], indicator)

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

        return pd.DataFrame(self.tr.trade_report, columns=self.tr.header)


class IndexTradeReport:
    def __init__(self):
        quotes = [
            [
                f"q{i}_bb",
                f"q{i}_bb_size",
                f"q{i}_bo",
                f"q{i}_bo_size",
                f"q{i}_quote_ts",
                f"q{i}_quote_condition",
            ]
            for i in range(1, 9)
        ]
        self.header = [
            "option",
            "indicator",
            "trade_time",
            "trade_size",
            "trade_price",
            "midpoint",
            "trade_code",
        ]
        self.header += [item for sublist in quotes for item in sublist]
        self.header.insert(25, "time_to_fill")
        self.trade_report = None
        self.flag = True

    def add_trade(self, tx, quotes, indicator):
        midpoint = (quotes[0][7] + quotes[0][11]) / 2
        ts = [convert_time(x[1]) for x in [tx] + quotes]

        if quotes[0][1] is None or quotes[2][1] is None:
            dif = None
        else:
            dif = str(
                (
                    dt.datetime.strptime(ts[1], "%H:%M:%S.%f")
                    - dt.datetime.strptime(ts[3], "%H:%M:%S.%f")
                ).total_seconds()
            )

        quotes = [
            [q[7], q[6], q[11], q[10], ts[i + 1], q[14]] for i, q in enumerate(quotes)
        ]
        row = [tx[5], indicator, ts[0], tx[7], tx[8], midpoint, tx[12]]
        row += [item for sublist in quotes for item in sublist]
        row.insert(25, dif)

        if self.flag:
            self.trade_report = np.array([row])
            self.flag = False
        else:
            self.trade_report = np.vstack((self.trade_report, row))

        return
