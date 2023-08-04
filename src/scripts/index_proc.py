import os
import src.midas
from src import MPApply, get_trading_days


if __name__ == '__main__':
    src.midas.DATA_DIR = 'scratch/spy23_25-27'
    src.midas.OUT_DIR = 'scratch/spy23_25-27_out'    
    
    args = [('SPY', '20230127', filename) for filename in os.listdir(f'{src.midas.DATA_DIR}/20230127/SPY')]
    
    MPApply(args).apply(src.midas.index_worker, 4)
