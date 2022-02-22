from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from utils.utils import remove_outliers_iqr


def make_pca(dataframe):
    # Process categorical data
    cat_col_names = dataframe.select_dtypes(exclude=np.number).columns.to_list()
    dataframe.dropna(subset=cat_col_names, inplace=True)
    # Process numeric data
    numeric_cols = dataframe.select_dtypes(include=np.number).columns.to_list()
    remove_outliers_iqr(dataframe, columns=numeric_cols, iqr_multiplier=2)
    numeric_df = dataframe.select_dtypes(include=np.number)
    if numeric_df.isna().sum().sum() > 0:
        numeric_df = numeric_df.apply(lambda x: x.fillna(np.mean(x)))
    categorical_df = dataframe.select_dtypes(exclude=np.number)
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(data=scaled_data, columns=numeric_df.columns)
    # PCA
    pca = PCA()
    pca_data = pca.fit_transform(scaled_df)
    pca_df = pd.DataFrame(pca_data)
    pca_columns = [f'PCA_{i}' for i in list(pca_df.columns)]
    pca_col_mapper = dict(zip(list(pca_df.columns), pca_columns))
    pca_df.rename(columns=pca_col_mapper, inplace=True)
    # Output
    dataframe.reset_index(drop=True, inplace=True)
    output_df = pd.concat([dataframe, pca_df], axis=1)
    return output_df, categorical_df.columns.to_list(), pca_df.columns.to_list()
