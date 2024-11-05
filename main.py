import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import io

"""
# ANALISIS DATA KRIMINAL
Kelompok 9 - Akuisisi Data B
"""

data = st.file_uploader("Upload data CSV kamu", type="csv")

if data is not None:
    try:
        data = pd.read_csv(data)
    except Exception as e:
        st.error(f"Error membaca file: {str(e)}")

tab1, tab2, tab3 = st.tabs(["Display Data", "Pre-processing dan Analysis", "Visualization"])

# Tab 1: Display data aja (optional)
with tab1:
    if st.button("Display Data", type="primary"):
        if data is not None:
            st.write("Data preview:")
            st.dataframe(data)
            buffer = io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)  
            # st.text(buffer)

        else:
            st.warning("Silakan unggah file CSV terlebih dahulu")

# Tab 2: Pre-processing dan Analysis
with tab2:
    if st.button("Analysis", type="primary"):
        if data is not None:
        # Menampilkan data null ---------------------------------------
            #Progress Bar
            st.subheader("Missing Data:")
            progress_text = "Menganalisis data kamu..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()

            #Menampilkan data
            data.isnull() .sum()
            null_row = data[data.isnull().any(axis=1)]
            st.dataframe(null_row) 

        # Drop NA ---------------------------------------
            #Progress Bar
            progress_text = "Membersihkan data..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()

            #Drop baris kosong dan tampilkan
            data_bersih = data.dropna().fillna(0)
            st.subheader("Data sudah dibersihkan!")
            st.dataframe(data_bersih)

            #Verifikasi tidak ada missing value yang masih ada
            null_row2 = data_bersih[data_bersih.isnull().any(axis=1)]
            if not null_row2.empty:
                st.write("Rows with missing data after cleaning:")
                st.dataframe(null_row2)
            else:
                st.write("No missing data found after cleaning.")
        else:
            st.warning("Silakan unggah file CSV terlebih dahulu")

# Tab 3: Visualisasi
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