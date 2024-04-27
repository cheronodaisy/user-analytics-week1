'''
Streamlit app to visualize output of data analysis
'''
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)


df = load_data('../schema/cleaned_dataset.csv')

add_selectbox = st.sidebar.selectbox(
    "Contact Us",
    ("Email", "Home phone", "Mobile phone")
)

st.title('User Analysis Visualization')

ul_dl_fields = {
    'Social Media': ['Social Media DL (Bytes)', 'Social Media UL (Bytes)'],
    'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
    'Email': ['Email DL (Bytes)', 'Email UL (Bytes)'],
    'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
    'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)']
}

total_data_per_app = {}
for app, columns in ul_dl_fields.items():
    total_data_per_app[app] = df[columns[0]].sum() + df[columns[1]].sum()

total_data_per_app_df = pd.DataFrame.from_dict(total_data_per_app, orient='index', columns=['Total Data Volume (Bytes)'])

top_3_apps = total_data_per_app_df.nlargest(3, 'Total Data Volume (Bytes)')

plt.figure(figsize=(10, 6))
plt.bar(top_3_apps.index, top_3_apps['Total Data Volume (Bytes)'], color='skyblue')
plt.title('Top 3 Most Used Applications')
plt.xlabel('Application')
plt.ylabel('Total Data Volume (Bytes)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')

st.pyplot(plt)

st.title('Univariate and Bivariate Plots')


left_column, right_column, col3 = st.columns(3)
with left_column:
    st.image('../schema/output.png', caption='Gaming volume vs Total volume')

with right_column:
    st.image('../schema/email.png', caption='Email volume vs Total volume')
with col3:
    st.image('../schema/pairplots.png', caption='Pair Plots')
