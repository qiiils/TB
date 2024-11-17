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

tab1, tab2, tab3, tab4 = st.tabs(["Display Data", "Pre-processing", "Analysis Data", "Visualization"])

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
            
            # Summary Data ---------------------------------------
            st.subheader("Summary Dataset")
            st.write("Dataset ini terdiri dari 378 baris dan 7 kolom. Atribut yang tersedia :")
            st.write(" - Periode Data")
            st.write(" - Wilayah")
            st.write(" - Lokasi Pengaduan") 
            st.write(" - Tanggal Pengaduan")
            st.write(" - Asal Pengaduan")
            st.write(" - Tanggal Pengaduan")
            st.write(" - Jenis Kriminalitas")
            st.write(" - Jumlah Pengaduan")

            # Statistik Data -----------------------------------------------------
            st.subheader("Statistik Deskriptif")
            st.write(data_bersih.describe())
            
            # Tipe Data -------------------------------------------------------------------
            st.subheader("Tipe Data Setiap Kolom")
            st.write(data_bersih.dtypes)


data_bersih1 = data_bersih


#Tab 3
with tab3:
    if st.button("Analysis"):
        if data_bersih1 is not None:
            #Clustering 
            st.write(data_bersih.columns)
            if 'wilayah' in data_bersih.columns and 'jumlah_pengaduan' in data_bersih.columns:
                # Agregasi data untuk mendapatkan total jumlah pengaduan per wilayah
                total_pengaduan = data_bersih.groupby(['wilayah', 'jenis_kriminal'], as_index=False)['jumlah_pengaduan'].sum()

                # Ubah kolom 'wilayah' menjadi nilai numerik menggunakan Label Encoding
                le_wilayah = LabelEncoder()
                le_kriminalitas = LabelEncoder()
                total_pengaduan['wilayah_encoded'] = le_wilayah.fit_transform(total_pengaduan['wilayah'])
                total_pengaduan['jenis_kriminalitas_encoded'] = le_kriminalitas.fit_transform(total_pengaduan['jenis_kriminal'])

                # Persiapkan fitur untuk clustering
                X = total_pengaduan[['wilayah_encoded', 'jumlah_pengaduan']]

                # Pilihan jumlah cluster
                num_clusters = st.slider("Pilih jumlah cluster", 2, 10, 3)

                # Tentukan model K-Means
                kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                total_pengaduan['cluster'] = kmeans.fit_predict(X)

                # Tampilkan data wilayah dan jenis kriminalitas yang terkluster
                st.subheader("Data dengan Cluster")
                st.write(total_pengaduan)

                # Buat dataframe keterangan wilayah
                wilayah_map = total_pengaduan[['wilayah', 'wilayah_encoded']].drop_duplicates().sort_values(by='wilayah_encoded')

                # Visualisasi hasil clustering berdasarkan wilayah dan jenis kriminalitas
                st.subheader("Hasil Clustering Berdasarkan Wilayah dan Jenis Kriminalitas")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=total_pengaduan, x='wilayah_encoded', y='jenis_kriminalitas_encoded', size='jumlah_pengaduan', hue='cluster', palette='viridis', ax=ax, sizes=(20, 200))
                ax.set_title("Clustering Berdasarkan Wilayah dan Jenis Kriminalitas")
                ax.set_xlabel("Wilayah (Encoded)")
                ax.set_ylabel("Jenis Kriminalitas (Encoded)")
                st.pyplot(fig)

                # Buat tabel keterangan kode wilayah dan jenis kriminalitas
                st.subheader("Keterangan Kode Wilayah dan Jenis Kriminalitas")
                wilayah_map = pd.DataFrame({'Wilayah': total_pengaduan['wilayah'], 'Kode Wilayah': total_pengaduan['wilayah_encoded']}).drop_duplicates().sort_values(by='Kode Wilayah')
                kriminalitas_map = pd.DataFrame({'Jenis Kriminalitas': total_pengaduan['jenis_kriminal'], 'Kode Kriminalitas': total_pengaduan['jenis_kriminalitas_encoded']}).drop_duplicates().sort_values(by='Kode Kriminalitas')
                st.write("Kode Wilayah:", wilayah_map)
                st.write("Kode Jenis Kriminalitas:", kriminalitas_map)

                describe = data_bersih.describe()
                st.write(describe)

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

# Tab 4 : Visualisasi
with tab4:
    pilih_visualisasi = st.selectbox(
        "Pilih visualisasi:",
        ["Trend waktu pengaduan", "Asal Pengaduan", "Trend Jenis Kriminalitas"]
    )

    # pilih_visualisasi = st.selectbox(
    #     "Pilih wilayah:",
    #     # data_bersih['wilayah']
    # )

        # if data_bersih is not None:
        # sd
        # else:
        # st.warning("Silakan unggah file CSV terlebih dahulu")
