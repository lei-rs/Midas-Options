from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import subprocess
import os


CHUNK = 1000
DATA_DIR = None 
SYMBOL_DIR = None


class ProcReader:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE,
        )

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise StopIteration
            else:
                return line.strip()


class Product:
    def __init__(self, product, data, date):
        self.product = product
        self.data = [data]
        self.date = date

    def append(self, twxm):
        self.data.append(twxm)

    def check(self):
        if len(self.data) > CHUNK:
            self._write()

    def write(self):
        if len(self.data) > 0:
            self._write()

    def _write(self):
        df = pd.DataFrame(self.data, columns=[f'c{i}' for i in range(1, 16)])
        symbol = self.data[0][5].split('_')[0]
        os.makedirs(f'{DATA_DIR}/{self.date}/{symbol}/', exist_ok=True)
        path = f'{DATA_DIR}/{self.date}/{symbol}/{self.product}.parquet.gzip'
        
        if not os.path.isfile(path):
            df.to_parquet(path, index=None, engine='fastparquet', compression='gzip')
        else:
            df.to_parquet(path, index=None, engine='fastparquet', compression='gzip', append=True)

        self.data = []


class SplitProducts:
    def __init__(self, twxm, date):
        self._twxm = twxm
        self.products = {}
        self.date = date
        with_trades = list(pd.read_csv(SYMBOL_DIR.format(date))['symbol'])
        self.with_trades = set([s.replace('_', '').replace(' ', '') for s in with_trades])

    def _save_data(self, product: str, row: str):
        if product in self.products:
            p = self.products[product]
            p.append(row)

        else:
            p = Product(product, row, self.date)
            self.products[product] = p

    def execute(self):
        for twxm_byte in self._twxm:
            twxm = twxm_byte.decode('utf-8').split(' ')
            product = twxm[5].replace('_', '')

            if product in self.with_trades:
                self._save_data(product, twxm)
                self.products[product].check()

        for k, v in self.products.items():
            v.write()


def download(symbol, date):
    twxm = ProcReader(f'twxm {date} opra {symbol}_*')
    SplitProducts(twxm, date).execute()


class MPApply:
    def __init__(self, symbols, date_range):
        self._symbols = symbols
        self._date_range = date_range

    def apply(self, func, total_procs):
        os.makedirs(f'{DATA_DIR}/', exist_ok=True)
        executor = ProcessPoolExecutor(max_workers=total_procs)
        results = []
        for symbol in self._symbols:
            for date in self._date_range:
                results.append(executor.submit(func, symbol, date))
        
        for res in tqdm(results):
            print(res.result())
            
        executor.shutdown(wait=True)


'''class Generator:
    def __init__(self):
        self._date_range = os.listdir(f'{DATA_DIR}/')

    def _generate_tr(self, symbol, k, date):
        v = pd.read_parquet(f'{DATA_DIR}/{date}/{symbol}/{k}')
        params = {
            'mode': 'a',
            'index': False,
            'float_format': '%.3f',
            'header': False
        }

        if symbol in ('SPX', 'SPXW'):
            generator = IndexGenerator(v, k)
            generator.generate_tr().to_csv(f'{OUT_DIR}/{date}/tr_{symbol}.csv', **params)

        else:
            generator = EquityGenerator(v)
            generator.generate_tr().to_csv(f'{OUT_DIR}/{date}/tr_{symbol}.csv', **params)

    def __call__(self, total_procs):
        pool = multiprocessing.Pool(total_procs)

        for i, date in enumerate(self._date_range):
            os.makedirs(f'{OUT_DIR}/{date}', exist_ok=True)
            symbols = os.listdir(f'{DATA_DIR}/{date}/')

            for s in symbols:
                for k in os.listdir(f'{DATA_DIR}/{date}/{s}/'):
                    pool.apply_async(self._generate_tr, args=[s, k, date])

        pool.close()
        pool.join()'''
