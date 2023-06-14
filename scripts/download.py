import src.midas
from src import MPApply, get_trading_days


if __name__ == '__main__':
    src.midas.DATA_DIR = 'spx_opra_raw'
    MPApply(['SPX'], get_trading_days(20190102, 20211231)).apply(src.midas.download, 1)
