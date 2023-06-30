import select_base as sb
import numpy as np
import pandas as pd

class read_data:
    
    def __init__(self,csv_data=False):
        print('\n Reading data... \n')
        if csv_data:
            self.df = pd.read_csv('data/sales_data.csv',index_col=0)
        else:
            SQL_conn = sb.read_data_SQL()
            self.df = SQL_conn.read_table(save_csv=True)

if __name__ == '__main__':
    rd = read_data()