import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

"""
# ANALISIS DATA KRIMINAL
Kelompok 9 - Akuisisi Data B
"""

# Data Loading and Preprocessing Functions
def load_and_preprocess_data(data):
    df = pd.read_csv(data)
    
    # Handle missing dates
    df['tanggal_pengaduan'] = pd.to_datetime(df['tanggal_pengaduan'], errors='coerce')
    
    # Fill missing dates with the median date for each periode_data
    df['tanggal_pengaduan'] = df.groupby('periode_data')['tanggal_pengaduan'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Clean other columns
    df['wilayah'] = df['wilayah'].fillna('Unknown')
    df['lokasi_pengaduan'] = df['lokasi_pengaduan'].fillna('Unknown')
    df['asal_pengaduan'] = df['asal_pengaduan'].fillna('Unknown')
    df['jenis_kriminal'] = df['jenis_kriminal'].fillna('Unknown')
    df['jumlah_pengaduan'] = df['jumlah_pengaduan'].fillna(df['jumlah_pengaduan'].mean())
    
    return df

# Time Series Analysis Functions
def prepare_time_series_data(df):
    daily_crimes = df.groupby('tanggal_pengaduan')['jumlah_pengaduan'].sum().reset_index()
    daily_crimes.columns = ['ds', 'y']
    return daily_crimes

def perform_time_series_analysis(df):
    ts_data = prepare_time_series_data(df)
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(ts_data)
    
    future_dates = model.make_future_dataframe(periods=30)
    forecast = model.predict(future_dates)
    return forecast

# Association Rule Mining Functions
def prepare_transaction_data(df):
    # Create transaction matrix
    transaction_data = pd.crosstab(index=df['wilayah'], 
                                 columns=df['jenis_kriminal'])
    return transaction_data

def mine_association_rules(transaction_data):
    # Generate frequent itemsets
    frequent_itemsets = apriori(transaction_data, min_support=0.1, use_colnames=True)
    
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    return rules

# Main Application
def main():
    st.title("Analisis Data Kriminal Jakarta")
    
    # File upload
    data = st.file_uploader("Upload data CSV kamu", type="csv")
    
    if data is not None:
        # Load and preprocess data
        df = load_and_preprocess_data(data)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Data Overview", 
            "Time Series Analysis", 
            "Association Analysis",
            "Statistical Insights"
        ])
        
        # Tab 1: Data Overview
        with tab1:
            st.header("Overview Data")
            st.write("Preview Dataset:")
            st.dataframe(df)
            
            st.subheader("Statistik Deskriptif")
            st.write(df.describe())
            
            # Distribution of crime types
            fig = px.pie(df, names='jenis_kriminal', title='Distribusi Jenis Kriminal')
            st.plotly_chart(fig)
            
            # Crime by region
            fig = px.bar(df.groupby('wilayah')['jumlah_pengaduan'].sum().reset_index(),
                        x='wilayah', y='jumlah_pengaduan',
                        title='Jumlah Kasus per Wilayah')
            st.plotly_chart(fig)
        
        # Tab 2: Time Series Analysis
        with tab2:
            st.header("Analisis Time Series")
            
            # Perform time series analysis
            forecast = perform_time_series_analysis(df)
            
            # Plot forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                   name='Prediksi'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                                   fill=None, name='Upper Bound'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                                   fill='tonexty', name='Lower Bound'))
            fig.update_layout(title='Prediksi Jumlah Kasus 30 Hari ke Depan',
                            xaxis_title='Tanggal',
                            yaxis_title='Jumlah Kasus')
            st.plotly_chart(fig)
            
            # Seasonality components
            st.subheader("Komponen Musiman")
            fig = plt.figure(figsize=(12, 8))
            plt.plot(forecast['ds'], forecast['yearly'], label='Yearly')
            plt.plot(forecast['ds'], forecast['weekly'], label='Weekly')
            plt.legend()
            st.pyplot(fig)
        
        # # Tab 3: Association Analysis
        # with tab3:
        #     st.header("Analisis Asosiasi")
            
        #     transaction_data = prepare_transaction_data(df)
        #     rules = mine_association_rules(transaction_data)
            
        #     st.write("Association Rules yang Ditemukan:")
        #     st.dataframe(rules)
            
        #     # Visualize rules
        #     fig = px.scatter(rules, x='support', y='confidence',
        #                    size='lift', color='lift',
        #                    title='Association Rules Visualization')
        #     st.plotly_chart(fig)
        
        # Tab 4: Statistical Insights
        with tab4:
            st.header("Insight Statistik")
            
            # Crime trends over time
            monthly_crimes = df.groupby(df['tanggal_pengaduan'].dt.to_period('M'))['jumlah_pengaduan'].sum()
            fig = px.line(x=monthly_crimes.index.astype(str), y=monthly_crimes.values,
                         title='Tren Kriminalitas per Bulan')
            st.plotly_chart(fig)
            
            # Crime type by reporting source
            crime_source = pd.crosstab(df['jenis_kriminal'], df['asal_pengaduan'])
            fig = px.imshow(crime_source, 
                          title='Heatmap Jenis Kriminal vs Asal Pengaduan')
            st.plotly_chart(fig)
            
            # Statistical tests
            # st.subheader("Uji Chi-Square: Hubungan antara Wilayah dan Jenis Kriminal")
            # contingency_table = pd.crosstab(df['wilayah'], df['jenis_kriminal'])
            # chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            # st.write(f"P-value: {p_value:.4f}")
            # if p_value < 0.05:
            #     st.write("Terdapat hubungan signifikan antara wilayah dan jenis kriminal")
            # else:
            #     st.write("Tidak terdapat hubungan signifikan antara wilayah dan jenis kriminal")

if __name__ == "__main__":
    main()