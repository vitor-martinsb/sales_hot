import mysql.connector
import os
import shutil
import pandas as pd


class read_data_SQL:

    """
    Class for reading data from a SQL table.

    Parameters:
        host (str): The SQL database host.
        port (str): The port for the SQL database connection.
        user (str): The username for the SQL database connection.
        password (str): The password for the SQL database connection.
        database (str): The name of the SQL database.

    Attributes:
        cnx (mysql.connector.connection.MySQLConnection): The connection to the SQL database.

    Methods:
        read_table(table_name='sales_data', save_csv=False): Reads a specific table from the database and returns a DataFrame.

    Example usage:
        # Create an instance of the class and read a table from the database
        reader = read_data_SQL(host='interview-2.ck1h5ksgzpiq.us-east-1.rds.amazonaws.com',
                               port='3306',
                               user='hotinterview',
                               password='6cT4jk9QWPhQC9KXWKDd',
                               database='innodb')
        df = reader.read_table(table_name='sales_data', save_csv=True)
    """
    
    def __init__(self,host='interview-2.ck1h5ksgzpiq.us-east-1.rds.amazonaws.com',port='3306',user='hotinterview',password='6cT4jk9QWPhQC9KXWKDd',database='innodb'):
        """
        Initializes the read_data_SQL object and establishes a connection to the SQL database.

        Parameters:
            host (str): The SQL database host.
            port (str): The port for the SQL database connection.
            user (str): The username for the SQL database connection.
            password (str): The password for the SQL database connection.
            database (str): The name of the SQL database.
        """
        self.cnx = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )

    def read_table(self, table_name = 'sales_data',save_csv=False):
        """
        Reads a specific table from the SQL database and returns a DataFrame.

        Parameters:
            table_name (str): The name of the table to read. Defaults to 'sales_data'.
            save_csv (bool): Flag indicating whether to save the DataFrame as a CSV file.
                             If True, the DataFrame will be saved as 'data/sales_data.csv'.
                             Defaults to False.

        Returns:
            pandas.DataFrame: The DataFrame containing the read data.

        """
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