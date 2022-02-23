from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from utils.utils import remove_outliers_iqr, del_from_list
import streamlit as st
import plotly.express as px


class GeneralAnalysis:
    def __init__(self):
        self.dataframe = None
        self.categorical_cols = None
        self.numeric_cols = None

    def parse(self, datafile):
        dataframe = pd.read_csv(datafile)
        return dataframe

    def clean(self, dataframe):
        # drop all NaN categorical var
        self.categorical_cols = dataframe.select_dtypes(exclude=np.number).columns.to_list()
        dataframe.dropna(subset=self.categorical_cols, inplace=True)
        for col in self.categorical_cols:
            dataframe[col].str.strip()
        # Process numeric data
        self.numeric_cols = dataframe.select_dtypes(include=np.number).columns.to_list()
        remove_outliers_iqr(dataframe, columns=self.numeric_cols, iqr_multiplier=2)
        dataframe.reset_index(drop=True, inplace=True)
        for col in self.numeric_cols:
            if dataframe[col].isna().sum() > 0:
                dataframe[col].fillna(np.mean(dataframe[col]), inplace=True)
        self.dataframe = dataframe

    def fit(self):
        pass

    def draw(self):
        pass

    def empty(self):
        pass


class PrincipalComponent(GeneralAnalysis):
    def __init__(self):
        super().__init__()
        self.pca = None
        self.component = None
        self.loadings = None
        self.pca_cols = None
        self.pca_features = None
        self.x_component = None
        self.y_component = None
        self.hue_col = None
        self.name = 'Principal Component Analysis'

    def fit(self, datafile):
        if self.component is None:
            dataframe = self.parse(datafile)
            self.clean(dataframe)
            numeric_df = self.dataframe[self.numeric_cols]
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            scaled_df = pd.DataFrame(data=scaled_data, columns=numeric_df.columns)
            # PCA
            self.pca = PCA()
            self.component = self.pca.fit_transform(scaled_df)
            self.loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
            self.pca_cols = [f'PCA_{i}' for i in range(0, self.pca.n_components_)]
            self.pca_features = self.numeric_cols

    def draw(self, chart_col, setting_col):
        self.x_component = setting_col.selectbox('X Component', self.pca_cols)
        self.y_component = setting_col.selectbox('Y Component', del_from_list(self.pca_cols, self.x_component))
        self.hue_col = setting_col.selectbox('Categorizing field', self.categorical_cols)
        # Components and Loadings Plot
        output_df = pd.concat([self.dataframe, self.components_to_df()], axis=1)
        fig = px.scatter(data_frame=output_df,
                         x=self.x_component, y=self.y_component,
                         color=self.hue_col,
                         title='Component and Loading Plot')
        for i, feature in enumerate(self.pca_features):
            fig.add_shape(type='line',
                          x0=0, y0=0,
                          x1=self.loadings[i, self.pca_cols.index(self.x_component)],
                          y1=self.loadings[i, self.pca_cols.index(self.y_component)])
            fig.add_annotation(x=self.loadings[i, self.pca_cols.index(self.x_component)],
                               y=self.loadings[i, self.pca_cols.index(self.y_component)],
                               ax=0, ay=0,
                               xanchor='center', yanchor='bottom',
                               text=feature)
        chart_col.plotly_chart(fig)
        # Loadings
        loadings_col, loading_plot_col = st.columns([1, 2])
        loadings_col.write('Loadings')
        loadings_col.dataframe(self.loadings_to_df())
        # Explained Variances
        each_var_col, cumul_var_col = st.columns([1, 1])
        exp_var_cumul = np.cumsum(self.pca.explained_variance_ratio_)
        cumul_var_col.plotly_chart(
            px.area(x=range(1, exp_var_cumul.shape[0] + 1),
                    y=exp_var_cumul,
                    labels={"x": "# Components", "y": "Explained Variance"},
                    title='Cumulative explained variance'
            )
        )
        each_var_col.plotly_chart(
            px.bar(pd.DataFrame(data=self.pca.explained_variance_ratio_,
                                index=self.pca_cols),
                   title='Explained variance')
        )

    def components_to_df(self):
        pca_df = pd.DataFrame(self.component)
        pca_columns = [f'PCA_{i}' for i in list(pca_df.columns)]
        pca_col_mapper = dict(zip(list(pca_df.columns), pca_columns))
        pca_df.rename(columns=pca_col_mapper, inplace=True)
        return pca_df

    def loadings_to_df(self):
        return pd.DataFrame(data=self.loadings, index=self.pca_features, columns=self.pca_cols)

    def empty(self):
        self.dataframe = None
        self.pca = None
        self.component = None
        self.loadings = None
        self.pca_cols = None
        self.pca_features = None
        self.categorical_cols = None
        self.numeric_cols = None
        self.x_component = None
        self.y_component = None
        self.hue_col = None

