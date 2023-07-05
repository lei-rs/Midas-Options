import src.midas
from src import MPApply, get_trading_days


if __name__ == '__main__':
    src.midas.DATA_DIR = 'scratch/su_spy_2023_2'
    src.midas.SYMBOL_DIR = 'symbols/for_su/spy_{}.csv'
    MPApply(['SPY'], get_trading_days('20230103', '20230112')).apply(src.midas.download, 4)
