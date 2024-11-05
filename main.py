import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
# ANALISIS DATA KRIMINAL
sub text
"""

data = st.file_uploader("Upload data CSV kamu", type="csv")

if data is not None:
    try:
        data = pd.read_csv(data)
    except Exception as e:
        st.error(f"Error membaca file: {str(e)}")

tab1, tab2, tab3 = st.tabs(["Display Data", "Pre-processing dan Analysis", "Visualization"])

with tab1:
    if st.button("Display Data", type="primary"):
        if data is not None:
            st.write("Data preview:")
            st.dataframe(data)
            buffer = data.info(buf=None)  # Capture the info output
            st.text(buffer)

        else:
            st.warning("Silakan unggah file CSV terlebih dahulu")

# Tab 2: Pre-processing dan Analysis
with tab2:
    if st.button("Analysis", type="primary"):
        if data is not None:
            st.write("Analisis Data:")
            data.isnull() .sum()
            null_row = data[data.isnull().any(axis=1)]
            st.dataframe(null_row) 
        else:
            st.warning("Silakan unggah file CSV terlebih dahulu")

# Tab 3: Visualization
# with tab3:
    # if data is not None:
    #     columns = data.columns.tolist()
    #     x_axis1 = st.selectbox("Pilih x-axis:", options=columns, index=None)
    #     y_axis1 = st.selectbox("Pilih y-axis:", options=columns, index=None)
    
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(x_axis1, y_axis1, marker='o', color='b', label='Sine Wave')
    #     plt.xlabel('X-axis')
    #     plt.ylabel('Y-axis')
    #     plt.title('Contoh Grafik Garis')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # else:
    #     st.warning("Silakan unggah file CSV terlebih dahulu")