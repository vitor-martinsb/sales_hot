import select_base as sb
import numpy as np
import pandas as pd
from tqdm import tqdm
from RFM import RFM
import matplotlib.pyplot as plt
from datetime import datetime, date

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns

class read_data:
    """
    A class for reading data from either a CSV file or a SQL table.

    Parameters:
        csv_data (bool): Flag indicating whether to read data from a CSV file or not.
                         If True, data is read from a CSV file. If False, data is read
                         from a SQL table.

    Attributes:
        df (pandas.DataFrame): The DataFrame containing the read data.

    Example usage:
        # Read data from CSV file
        reader = read_data(csv_data=True)
        df_from_csv = reader.df

        # Read data from SQL table
        reader = read_data(csv_data=False)
        df_from_sql = reader.df
    """
    
    def __init__(self,csv_data=True):

        """
        Initializes the read_data object and reads data from either a CSV file or a SQL table.

        Parameters:
            csv_data (bool): Flag indicating whether to read data from a CSV file or not.
                             If True, data is read from a CSV file. If False, data is read
                             from a SQL table.

        """

        print('\n Reading data... \n')
        if csv_data:
            self.df = pd.read_csv('data/sales_data.csv',index_col=0)
        else:
            SQL_conn = sb.read_data_SQL()
            self.df = SQL_conn.read_table(save_csv=True)
        print('\n FINISH !!! \n')

    def val_faturation(self, media = 100, desvio_padrao = 10):
        """
        Calculates and plots the valuation and faturation of a set of products.

        Args:
            media (float): Mean value to be used in the calculation. Default is 100.
            desvio_padrao (float): Standard deviation to be used in the calculation. Default is 10.

        Returns:
            tuple: Two DataFrames containing the valuation and faturation based on product categories and niches, respectively.
        """

        def plot_bar(df,xlabel='Nicho',media=100,desvio_padrao=10):
            
            """
            Plots a bar chart and line chart showing the valuation and faturation.

            Args:
                df (DataFrame): DataFrame containing the data for plotting.
                xlabel (str): Label for the x-axis. Default is 'Nicho'.
                media (float): Mean value used in the calculation. Default is 100.
                desvio_padrao (float): Standard deviation used in the calculation. Default is 10.
            """

            x = df.iloc[:, 0]
            y = 100 * (df.iloc[:, 2].to_numpy()) / df.iloc[:, 2].sum()
            z = df.iloc[:, 1].to_numpy()

            fig, ax1 = plt.subplots()
            ax1.set_xlabel(xlabel,fontsize=14,fontweight='bold')
            ax1.set_ylabel('Quantidade (%)',fontsize=12,fontweight='bold')
            ax1.bar(x, y, color='#f04e23',label='Quantidade de vendas')
            ax1.set_xticklabels(x, rotation='vertical',fontsize=6)
  
            for i, v in enumerate(y):
                ax1.annotate(str(round(v, 2)) + '%', xy=(i, v), ha='center', va='bottom', fontsize=10)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Faturamento (media = {} e dp = {})'.format(media,desvio_padrao),fontsize=12,fontweight='bold')
            ax2.plot(x, z, color='#19A86E',label='Faturamento',linewidth=2.5)
            ax2.set_xticklabels(x, rotation='vertical',fontsize=6)

            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            fig.tight_layout()

            ax1.tick_params(axis='x', labelsize=8)
            ax1.tick_params(axis='y', labelsize=12)
            ax2.tick_params(axis='y', labelsize=12)

            plt.show()

        
        sell_data = self.df[['product_category','product_niche','purchase_value']]
        sell_data['purchase_value'] = (sell_data['purchase_value'] * desvio_padrao) + media

        vet_product_category = []
        vet_product_category_sum = []
        vet_product_category_len = []
        for category, group in tqdm(sell_data.groupby('product_category')):
            vet_product_category.append(category)
            vet_product_category_sum.append(group['purchase_value'].sum())
            vet_product_category_len.append(len(group['purchase_value']))

        df_product_category_value = pd.DataFrame({'product_by_category': vet_product_category,
                                                  'product_by_category_value': vet_product_category_sum,
                                                  'product_by_category_len': vet_product_category_len})
        plot_bar(df_product_category_value,xlabel='Categoria')
        
        vet_niche_category = []
        vet_niche_category_sum = []
        vet_niche_category_len = []
        for category, group in tqdm(sell_data.groupby('product_niche')):
            vet_niche_category.append(category)
            vet_niche_category_sum.append(group['purchase_value'].sum())
            vet_niche_category_len.append(len(group['purchase_value']))

        df_product_niche_value = pd.DataFrame({'product_by_category': vet_niche_category,
                                                'product_by_category_value': vet_niche_category_sum,
                                                'product_by_category_len': vet_niche_category_len})
        plot_bar(df_product_niche_value,xlabel='Nicho')

        return df_product_category_value, df_product_niche_value
    
    def seg_client(self, media = 100, desvio_padrao = 10, read_RFM_data=True):
        """
        Segment the clients based on RFM (Recency, Frequency, Monetary) analysis.

        Args:
            media (int): Mean value for scaling the purchase value (default: 100).
            desvio_padrao (int): Standard deviation for scaling the purchase value (default: 10).
            read_RFM_data (bool): Flag to read RFM data from file or perform RFM analysis (default: True).
        """
        df_cliente = self.df[['buyer_id','purchase_date','purchase_value']]
        df_cliente['purchase_date'] = pd.to_datetime(df_cliente['purchase_date'])
        df_cliente['purchase_value'] = (df_cliente['purchase_value'] * desvio_padrao) + media

        if read_RFM_data:
            df_rfm = pd.read_csv('data/RFM.csv',index_col=0)
        else:

            vet_cliente = []
            vet_mon = []
            vet_fre = []
            vet_rec = []

            for cliente, group in tqdm(df_cliente.groupby('buyer_id')):
                group.sort_values(by='purchase_date',ascending=False)
                vet_cliente.append(cliente)
                vet_mon.append(group['purchase_value'].sum())
                vet_fre.append(len(group['purchase_value']))
                closest_date = group['purchase_date'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
                closest_date = datetime.strptime(closest_date, '%Y-%m-%d %H:%M:%S')
                vet_rec.append((date.today() - closest_date.date()).days)

            df_rfm = pd.DataFrame({'buyer_id':vet_cliente,'R':vet_rec,'F':vet_fre,'M':vet_mon})
            
            df_rfm.to_csv('data/RFM.csv')
  
        seg_RFM = RFM(ID=df_rfm['buyer_id'].to_numpy(),recency=df_rfm['R'].to_numpy(),frequency=df_rfm['F'].to_numpy(),monetary=df_rfm['M'].to_numpy(),n_quantis=4)
        self.RFM_segment = seg_RFM.evaluate_RFM_quantis(plot_data=False)

        segt_map = {
            r'[4][1-2]': 'Hibernando',
            r'[3-4][3]': 'Sonolento',
            r'[3-4]4': 'Campeão Ausente',
            r'[2-3][1-2]': 'Promissor',

            r'23': 'Campeão',
            r'24': 'Campeão',

            r'[1-2][1]': 'Novo',
            r'[1-2][2]': 'Promissor',
            r'1[3-4]': 'Campeão'
        }

        vetor_media_FM = np.round(self.RFM_segment[['F','M']].mean(axis=1).to_numpy(),decimals=0)
        vetor_media_FM = {'FM':vetor_media_FM.astype(int)}
        df_vetor_media_FM = pd.DataFrame(vetor_media_FM,self.RFM_segment.index)
        
        df_vetor_media_FM = pd.DataFrame(vetor_media_FM,self.RFM_segment.index)
        self.RFM_segment['RFM_class'] = self.RFM_segment['R'].map(str) + df_vetor_media_FM['FM'].map(str)
        self.RFM_segment['RFM_class'] = self.RFM_segment['RFM_class'].replace(segt_map, regex=True)

        seg_RFM.boxplot_RFM(opt_RFM = 0,column_name='RFM_class',filliers = False)
        seg_RFM.boxplot_RFM(opt_RFM = 1,column_name='RFM_class',filliers = False)
        seg_RFM.boxplot_RFM(opt_RFM = 2,column_name='RFM_class',filliers = False)

        self.RFM_segment.to_csv('data/client_segmentation.csv')
        print('Finish segmentation')

    def feature_analysis(self):
        """

        Perform feature analysis by selecting relevant features based on variance threshold and plot feature importance.

        """
        threshold = 0.1  # Example threshold value
        variance_selector = VarianceThreshold(threshold)

        X = self.df
        X = X.sort_values(by='buyer_id')
        seg = pd.read_csv('data/client_segmentation.csv',index_col=0)
        seg['buyer_id'] = seg.index.values
        
        X_categorical = X[['product_category', 'product_niche', 'purchase_device']]
        X_numeric = X[['buyer_id','purchase_value', 'affiliate_commission_percentual']]
        X_numeric['affiliate_commission_percentual'] = X_numeric['affiliate_commission_percentual'] / 100

        X_encoded = pd.get_dummies(X_categorical, dtype=float)
        X = pd.concat([X_encoded, X_numeric], axis=1)
        X = pd.DataFrame(X).dropna()

        target_variable = pd.merge(X,seg,how='left',on='buyer_id')
        X = X.drop(columns = 'buyer_id')
        target_variable = target_variable[['buyer_id','RFM_class']]

        # Fit the selector to the data
        variance_selector.fit(X)

        # Get the selected feature indices
        selected_indices = variance_selector.get_support(indices=True)

        # Get the selected feature names
        selected_features = X.columns[selected_indices]

        # Print the selected features
        print("Selected Features:")
        for feature in selected_features:
            print(feature)

        # Perform feature importance using a Random Forest model
        model = RandomForestRegressor()

        target_variable['RFM_class'] = target_variable['RFM_class'].replace('Campeão Ausente',0)
        target_variable['RFM_class'] = target_variable['RFM_class'].replace('Campeão',1)
        target_variable['RFM_class'] = target_variable['RFM_class'].replace('Promissor',2)
        target_variable['RFM_class'] = target_variable['RFM_class'].replace('Novo',3)
        target_variable['RFM_class'] = target_variable['RFM_class'].replace('Hibernando',4)

        model.fit(X[selected_features], target_variable['RFM_class'].to_numpy(dtype=int))

        # Get feature importance
        importance = model.feature_importances_

        # Sort feature importance in descending order
        sorted_indices = np.argsort(importance)[::-1]
        sorted_features = X.columns[sorted_indices]

        # Plot feature importance
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(sorted_features)), 100 * importance[sorted_indices],color='#f04e23')
        plt.xticks(range(len(sorted_features)), sorted_features, rotation='vertical')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

        print('Finish')

    def seg_prod(self):
        data = self.df
        data = data.sort_values(by='buyer_id')
        seg = pd.read_csv('data/client_segmentation.csv',index_col=0)
        seg['buyer_id'] = seg.index
        data = pd.merge(data,seg,how='left',on='buyer_id')

        for col in ['product_category', 'product_niche', 'purchase_device']:
            plt.figure(figsize=(8, 6))
            pivot_table = data.pivot_table(index='RFM_class', columns=col, aggfunc='size', fill_value=0)
            pivot_table_normalized = pivot_table / pivot_table.values.sum()
            print("*** Absolute number from {}: {}***".format(col,pivot_table.values.sum()))
            sns.heatmap(pivot_table_normalized, cmap='magma', annot=True, fmt='.1%')
            plt.xlabel(col)
            plt.ylabel('RFM')
            plt.tight_layout()
            plt.show()

# if __name__ == '__main__':
#     rd = read_data()
#     rd.val_faturation()
#     rd.seg_client()
#     rd.seg_prod()
#     rd.feature_analysis()
    

