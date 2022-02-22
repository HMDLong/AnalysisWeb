import numpy as np


def del_from_list(array, value):
    new_array = array.copy()
    new_array.remove(value)
    return new_array


def remove_outliers_iqr(data, columns=None, iqr_multiplier=1.5):
    if columns is None:
        columns = list(data.columns)
    for column in columns:
        q1 = np.percentile(data[column], 25, method='midpoint')
        q3 = np.percentile(data[column], 75, method='midpoint')
        iqr = q3 - q1
        upper_outliers = data.loc[data[column] > (q3 + iqr_multiplier*iqr)]
        lower_outliers = data.loc[data[column] < (q1 - iqr_multiplier*iqr)]
        data.drop(upper_outliers.index, inplace=True)
        data.drop(lower_outliers.index, inplace=True)
