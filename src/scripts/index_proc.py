import os
import src.midas
from src import MPApply, get_trading_days


if __name__ == '__main__':
    src.midas.DATA_DIR = 'scratch/spy_jan_17'
    src.midas.OUT_DIR = 'scratch/spy_jan_out'    
    
    args = []
    for date in get_trading_days('20230117', '20230120'):
        args += [('SPY', date, filename) for filename in os.listdir(f'{src.midas.DATA_DIR}/{date}/SPY')]
    
    MPApply(args).apply(src.midas.index_worker, 8)
