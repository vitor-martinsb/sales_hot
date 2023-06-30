import mysql.connector
import os
import shutil
import pandas as pd


class read_data_SQL:
    
    def __init__(self,host='interview-2.ck1h5ksgzpiq.us-east-1.rds.amazonaws.com',port='3306',user='hotinterview',password='6cT4jk9QWPhQC9KXWKDd',database='innodb'):
        
        self.cnx = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )

    def read_table(self, table_name = 'sales_data',save_csv=False):
        print('\n *** Reading table ' + table_name + ' ***')
        cursor = self.cnx.cursor()

        query = 'SELECT * FROM ' + table_name
        cursor.execute(query)

        rows = cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]

        df = pd.DataFrame(rows, columns=columns)

        if save_csv:
            try:
                df.to_csv('data/sales_data.csv')
            except:
                try:
                    shutil.rmtree('data')
                except:
                    os.mkdir('data')
                    df.to_csv('data/sales_data.csv')
        print('\n*** FINISH ***')
        cursor.close()
        self.cnx.close()
        print('\n*** Connection close ***')

        return df

# if __name__ == '__main__':
#     print('\n Begin... \n')

#     SQL_conn = read_data_SQL()
#     df = SQL_conn.read_table(save_csv=False)

#     print('\n Finish! \n')