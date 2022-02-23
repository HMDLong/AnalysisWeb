import streamlit as st
from analysis.pca import PrincipalComponent
from analysis.mca import MultiCorrespondence

st.set_page_config(layout="wide")


def make_engine(mode):
    # Control coupling
    if mode == 'PCA':
        return PrincipalComponent()
    if mode == 'MCA':
        return MultiCorrespondence()
    else:
        return None


if __name__ == '__main__':
    mode_col, _, upload_col = st.columns([1, 3, 3])
    uploaded_file = upload_col.file_uploader("Choose file")
    analysis_mode = mode_col.selectbox('Analysis mode', ['PCA', 'MCA', 'Something more'])
    analysis_engine = make_engine(analysis_mode)
    chart_col, setting_col = st.columns([4, 1])

    chart_col.title(analysis_engine.name)
    setting_col.title('Option')

    if uploaded_file is None:
        chart_col.header("Please upload your data")
        analysis_engine.empty()
    else:
        analysis_engine.fit(uploaded_file)
        analysis_engine.draw(chart_col, setting_col)




