import pandas as pd
import numpy as np
from utils.utils import remove_outliers_iqr


class MultiCorrespondence:
    def __init__(self):
        self.component = None
        self.loadings = None
        self.name = 'Multiple Correspondence Analysis'

    def fit(self, dataframe):
        pass
