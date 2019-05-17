# -*- coding: utf-8 -*-
"""
Created on Fri April 26 13:06:44 2019

@author: ryokugyu
"""
import numpy as np
import pandas as pd
from scipy.io import arff
import fancyimpute
from sklearn.preprocessing import Imputer
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

def load_arff_raw_data():
    N=5
    return [arff.loadarff('data/' + str(i+1) + 'year.arff') for i in range(N)]

def load_dataframes():
    return [pd.DataFrame(data_i_year[0]) for data_i_year in load_arff_raw_data()]

# Set the column headers from X1 ... X64 and the class label as Y, for all the 5 dataframes.
def set_new_headers(dataframes):
    cols = ['X' + str(i+1) for i in range(len(dataframes[0].columns)-1)]
    cols.append('Y')
    for df in dataframes:
        df.columns = cols

# dataframes is the list of pandas dataframes for the 5 year datafiles.  
dataframes = load_dataframes()

# Set the new headers for the dataframes. The new headers will have the renamed set of feature (X1 to X64)
set_new_headers(dataframes)    

df = pd.DataFrame(dataframes[0])

df.to_csv('./tempfile.csv', index=False)


# Convert the dtypes of all the columns (other than the class label columns) to float.
def convert_columns_type_float(dfs):
    for i in range(5):
        index = 1
        while(index<=63):
            colname = dfs[i].columns[index]
            col = getattr(dfs[i], colname)
            dfs[i][colname] = col.astype(float)
            index+=1
            
convert_columns_type_float(dataframes) 

def convert_class_label_type_int(dfs):
    for i in range(len(dfs)):
        col = getattr(dfs[i], 'Y')
        dfs[i]['Y'] = col.astype(int)
        
convert_class_label_type_int(dataframes)

def perform_mean_imputation(dfs):
        imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0)
        mean_imputed_dfs = [pd.DataFrame(imputer.fit_transform(df)) for df in dfs]
        for i in range(len(dfs)):
            mean_imputed_dfs[i].columns = dfs[i].columns   
        return mean_imputed_dfs

mean_imputed_dataframes = perform_mean_imputation(dataframes)


def perform_knn_imputation(dfs):
    knn_imputed_datasets = [fancyimpute.KNN(k=100,verbose=True).fit_transform(dfs[i]) for i in range(len(dfs))]
    return [pd.DataFrame(data=knn_imputed_datasets[i]) for i in range(len(dfs))]
    
knn_imputed_dataframes = perform_knn_imputation(dataframes)
set_new_headers(knn_imputed_dataframes)

X = mean_imputed_dataframes[0][["X1", "X2", "X6", "X7","X9","X10", "X34"]]
y = dataframes[0]['Y']

X.to_csv('./X.csv', header=False, index=False)
y.to_csv('./y.csv', header=False, index=False)


#converting to csv
mean_imputed_dataframes = pd.DataFrame(mean_imputed_dataframes[0])
mean_imputed_dataframes.to_csv('./mean_imputed_dataframes.csv', header=False)


knn_imputed_dataframes = pd.DataFrame(knn_imputed_dataframes[0])
knn_imputed_dataframes.to_csv('./knn_imputed_dataframes.csv')
