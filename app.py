import streamlit as st
import pandas as pd
from analysis.pca import make_pca
import plotly.express as xp
from utils.utils import del_from_list

st.set_page_config(layout="wide")

if __name__ == '__main__':
    mode_col, chart_col, setting_col = st.columns([1, 4, 1])
    analysis_mode = mode_col.selectbox('Analysis mode', ['PCA', 'Something more'])

    chart_col.title('Principal Component Analysis')
    setting_col.title('Option')

    uploaded_file = setting_col.file_uploader("Choose file")

    if uploaded_file is None:
        chart_col.header("Please upload your data")
    else:
        raw_df = pd.read_csv(uploaded_file)
        output_df, cat_cols, pca_cols = make_pca(raw_df)
        x_component = setting_col.selectbox('X Component', pca_cols)
        y_component = setting_col.selectbox('Y Component', del_from_list(pca_cols, x_component))
        hue_col = setting_col.selectbox('Categorizing field', cat_cols)
        chart_col.plotly_chart(xp.scatter(data_frame=output_df, x=x_component, y=y_component, color=hue_col))



