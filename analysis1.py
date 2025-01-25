import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.impute import SimpleImputer
from statsmodels.stats.weightstats import ztest


def perform_pca_analysis(elective_df):
    """Perform PCA analysis on the elective course data"""
    numerical_cols = ['S', 'A', 'B', 'C', 'D', 'E']
    data = elective_df[numerical_cols]

    # Handle missing values by imputing with the mean
    imputer = SimpleImputer(strategy="mean")
    data_imputed = imputer.fit_transform(data)

    # Normalization
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_imputed)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(normalized_data)

    # Create visualization
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Course Code'] = elective_df['Course Code']
    pca_df['Faculty Name'] = elective_df['Faculty Name']

    fig = px.scatter(pca_df, x='PC1', y='PC2',
                    hover_data=['Course Code', 'Faculty Name'],
                    title='PCA Analysis of Courses',
                    template='plotly_dark')

    return fig

def perform_hypothesis_test(elective_df):
    """Perform hypothesis testing on the grade distribution"""
    s_count = elective_df['S'].sum()
    a_count = elective_df['A'].sum()
    total_count = s_count + a_count
    z_stat, p_value = ztest(elective_df[['S', 'A']].sum(axis=1), value=total_count)
    return z_stat, p_value