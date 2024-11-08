import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import seaborn as sns


"""
# ANALISIS DATA KRIMINAL
Kelompok 9 - Akuisisi Data B
"""

# Deklarasi
data = st.file_uploader("Upload data CSV kamu", type="csv")
data_bersih = None


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
            # buffer = io.StringIO()
            # data.info(buf=buffer)
            # s = buffer.getvalue()
            # st.text(s)  
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
                st.write("Tidak ditemukan missing value setelah dibersihkan.")
            
            #Clustering 
            if 'wilayah' in data_bersih.columns and 'jumlah_pengaduan' in data_bersih.columns:
                # Agregasi data untuk mendapatkan total jumlah pengaduan per wilayah
                total_pengaduan = data_bersih.groupby('wilayah', as_index=False)['jumlah_pengaduan'].sum()

                # Ubah kolom 'wilayah' menjadi nilai numerik menggunakan Label Encoding
                le = LabelEncoder()
                total_pengaduan['wilayah_encoded'] = le.fit_transform(total_pengaduan['wilayah'])

                # Persiapkan fitur untuk clustering
                X = total_pengaduan[['wilayah_encoded', 'jumlah_pengaduan']]

                # Pilihan jumlah cluster
                num_clusters = st.slider("Pilih jumlah cluster", 2, 10, 3)

                # Tentukan model K-Means
                kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                total_pengaduan['cluster'] = kmeans.fit_predict(X)

                # Buat dataframe keterangan wilayah
                wilayah_map = total_pengaduan[['wilayah', 'wilayah_encoded']].drop_duplicates().sort_values(by='wilayah_encoded')

                # Visualisasi hasil clustering
                st.subheader("Hasil Clustering Berdasarkan Total Jumlah Pengaduan")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=total_pengaduan, x='wilayah_encoded', y='jumlah_pengaduan', hue='cluster', palette='viridis', ax=ax)
                ax.set_title("Clustering Wilayah Berdasarkan Total Jumlah Pengaduan")
                ax.set_xlabel("Wilayah (Encoded)")
                ax.set_ylabel("Total Jumlah Pengaduan")
                st.pyplot(fig)

                # Tampilkan data dengan cluster
                st.subheader("Data dengan Cluster")
                st.write(total_pengaduan)

                # Tampilkan keterangan kode wilayah
                st.subheader("Keterangan Kode Wilayah")
                st.write(wilayah_map)
            # if 'wilayah' in data_bersih.columns and 'jumlah_pengaduan' in data_bersih.columns:
            #     data_bersih = data_bersih[['wilayah', 'jumlah_pengaduan']]

            #     le = LabelEncoder()
            #     data['wilayah_encoded'] = le.fit_transform(data['wilayah'])

            #     X = data[['wilayah_encoded', 'jumlah_pengaduan']]

            #     num_clusters = st.slider("Pilih jumlah cluster", 2, 10, 3)

            #     kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            #     data['cluster'] = kmeans.fit_predict(X)

            #     wilayah_map = pd.DataFrame({
            #     'Wilayah': data['wilayah'],
            #     'Kode Wilayah': data['wilayah_encoded']
            #     }).drop_duplicates().sort_values(by='Kode Wilayah')
                
            #     st.subheader("Hasil Clustering")
            #     fig, ax = plt.subplots(figsize=(10, 6))
            #     sns.scatterplot(data=data, x='wilayah_encoded', y='jumlah_pengaduan', hue='cluster', palette='viridis', ax=ax)
            #     ax.set_title("Clustering Wilayah Berdasarkan Jumlah Pengaduan")
            #     ax.set_xlabel("Wilayah (Encoded)")
            #     ax.set_ylabel("Jumlah Pengaduan")
            #     st.pyplot(fig)

            #     st.subheader("Data dengan Cluster")
            #     st.write(data)

            #     st.subheader("Keterangan Kode Wilayah")
            #     st.write(wilayah_map)

            else:
                st.error("File CSV tidak memiliki kolom 'wilayah' dan 'jumlah_pengaduan' yang diperlukan.")

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