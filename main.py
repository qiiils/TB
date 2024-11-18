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

# Deklarasi -------------------------------------------------------------------
data = st.file_uploader("Upload data CSV kamu", type="csv")

if "data_bersih" not in st.session_state:
    st.session_state.data_bersih = None


if data is not None:
    try:
        data = pd.read_csv(data)
        data_bersih = data.dropna().fillna(0)
    except Exception as e:
        st.error(f"Error membaca file: {str(e)}")

tab1, tab2, tab3, tab4 = st.tabs(["Display Data", "Pre-processing", "Analysis Data", "Visualization"])

# Tab 1: Display data aja (optional) -------------------------------------------------------------------
with tab1:
    if st.button("Display Data", type="primary"):
        if data is not None:
            st.write("Data preview:")
            st.dataframe(data)
        else:
            st.warning("Silakan unggah file CSV terlebih dahulu")

# Tab 2: Pre-processing 
with tab2:
    if st.button("Analysis", type="primary"):
        if data is not None:
        # Summary Data -------------------------------------------------------------------
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

            # Statistik Data -------------------------------------------------------------------
            st.subheader("Statistik Deskriptif")
            st.write(data.describe())
            
            # Tipe Data -------------------------------------------------------------------
            st.subheader("Tipe Data Setiap Kolom")
            st.write(data.dtypes)

        # Menampilkan data null -------------------------------------------------------------------
            #Progress Bar -------------------------------------------------------------------
            st.subheader("Missing Data:")
            progress_text = "Menganalisis data kamu..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()

            #Menampilkan data -------------------------------------------------------------------
            data.isnull() .sum()
            null_row = data[data.isnull().any(axis=1)]
            st.dataframe(null_row) 

        # Drop NA -------------------------------------------------------------------
            #Progress Bar -------------------------------------------------------------------
            progress_text = "Membersihkan data..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()

            #Drop baris kosong dan tampilkan -------------------------------------------------------------------
            # data_bersih = data.dropna().fillna(0)
            st.subheader("Data sudah dibersihkan!")
            st.dataframe(data_bersih)

            st.session_state["data_bersih"] = data_bersih

            #Verifikasi tidak ada missing value yang masih ada -------------------------------------------------------------------
            null_row2 = data_bersih[data_bersih.isnull().any(axis=1)]
            if not null_row2.empty:
                st.write("Rows with missing data after cleaning:")
                st.dataframe(null_row2)
            else:
                st.write("Tidak ditemukan missing value setelah dibersihkan.")
            
            
#Tab 3: Clustering
with tab3:
    # if st.button("Perform Clustering", type="primary"):
        if "data_bersih" in st.session_state and st.session_state["data_bersih"] is not None:
            data_bersih = st.session_state["data_bersih"]
            
            progress_text = "Mempersiapkan clustering..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty() 

            if 'wilayah' in data_bersih.columns and 'jenis_kriminal' in data_bersih.columns and 'jumlah_pengaduan' in data_bersih.columns:
                # Aggregate data for clustering
                total_pengaduan = data_bersih.groupby(['wilayah', 'jenis_kriminal'], as_index=False)['jumlah_pengaduan'].sum()

                # Label Encoding -------------------------------------------------------------------
                le_wilayah = LabelEncoder()
                le_kriminalitas = LabelEncoder()
                total_pengaduan['wilayah_encoded'] = le_wilayah.fit_transform(total_pengaduan['wilayah'])
                total_pengaduan['jenis_kriminalitas_encoded'] = le_kriminalitas.fit_transform(total_pengaduan['jenis_kriminal'])

                # Fitur clustering -------------------------------------------------------------------
                X = total_pengaduan[['wilayah_encoded', 'jumlah_pengaduan']]

                # Pilih berapa banyak cluster -------------------------------------------------------------------
                num_clusters = st.slider("Pilih jumlah cluster", 2, 10, 3)

                # Apply KMeans -------------------------------------------------------------------
                kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                total_pengaduan['cluster'] = kmeans.fit_predict(X)

                # Menampilkan hasil clustering -------------------------------------------------------------------
                st.subheader("Data dengan Cluster")
                st.write(total_pengaduan)

                # Wilayah and Jenis Kriminalitas code -------------------------------------------------------------------
                wilayah_map = pd.DataFrame({'Wilayah': total_pengaduan['wilayah'], 'Kode Wilayah': total_pengaduan['wilayah_encoded']}).drop_duplicates().sort_values(by='Kode Wilayah')
                kriminalitas_map = pd.DataFrame({'Jenis Kriminalitas': total_pengaduan['jenis_kriminal'], 'Kode Kriminalitas': total_pengaduan['jenis_kriminalitas_encoded']}).drop_duplicates().sort_values(by='Kode Kriminalitas')

                # Visualisasi clustering -------------------------------------------------------------------
                st.subheader("Hasil Clustering Berdasarkan Wilayah dan Jenis Kriminalitas")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=total_pengaduan, x='wilayah_encoded', y='jenis_kriminalitas_encoded', size='jumlah_pengaduan', hue='cluster', palette='viridis', ax=ax, sizes=(20, 200))
                ax.set_title("Clustering Berdasarkan Wilayah dan Jenis Kriminalitas")
                ax.set_xlabel("Wilayah (Encoded)")
                ax.set_ylabel("Jenis Kriminalitas (Encoded)")
                st.pyplot(fig)

                # Keterangan kode -------------------------------------------------------------------
                st.subheader("Keterangan Kode Wilayah dan Jenis Kriminalitas")
                st.write("Kode Wilayah:", wilayah_map)
                st.write("Kode Jenis Kriminalitas:", kriminalitas_map)

                # Observasi Clustering ---------------------------------------------------------
                st.subheader("Observasi Clustering (3 cluster)")
                st.write(" - Kluster 0 (ungu tua): Berisi kombinasi wilayah dan jenis kriminalitas yang umumnya memiliki jumlah laporan moderat hingga rendah, terlihat dari ukuran titik yang lebih kecil.")
                st.write(" - Kluster 1 (tosca): Pada beberapa titik, kluster ini memiliki ukuran yang lebih besar, yang menunjukkan wilayah dan jenis kriminalitas dengan jumlah laporan yang tinggi.")
                st.write(" - Kluster 2 (kuning): Titik-titik dalam kluster ini lebih banyak tersebar dan umumnya memiliki ukuran yang lebih kecil, menunjukkan kombinasi wilayah dan jenis kriminalitas dengan jumlah laporan yang lebih rendah.")

            else:
                st.error("File CSV tidak memiliki kolom 'wilayah', 'jenis_kriminal', dan 'jumlah_pengaduan' yang diperlukan.")
        else:
            st.warning("Silakan unggah file CSV terlebih dahulu")

           
# Tab 4 : Visualisasi
with tab4:
    data_bersih['tanggal_pengaduan'] = pd.to_datetime(data_bersih['tanggal_pengaduan'], errors='coerce')
    data_bersih['bulan_pengaduan'] = data_bersih['tanggal_pengaduan'].dt.month

    x_axis_options = ["periode_data", "wilayah", "jenis_kriminal", "asal_pengaduan", "tanggal_pengaduan"]
    y_axis_options = ["jumlah_pengaduan"]
    
    x_axis = st.selectbox("Pilih X-axis:", x_axis_options)
    y_axis = st.selectbox("Pilih Y-axis:", y_axis_options)
    
    # Pilihan jenis visualisasi
    pilih_visualisasi = st.selectbox(
        "Pilih jenis visualisasi:",
        ["Jumlah Pengaduan per Tahun", "Jumlah Pengaduan berdasarkan Wilayah", 
         "Distribusi Jenis Kriminal", "Asal Pengaduan per Jenis Kriminal", "Trend Bulanan Pengaduan"]
    )

    if pilih_visualisasi == "Jumlah Pengaduan per Tahun":
        if x_axis == "periode_data" and y_axis == "jumlah_pengaduan":
            data_grouped = data.groupby(x_axis)[y_axis].sum().reset_index()
            fig, ax = plt.subplots()
            ax.plot(data_grouped[x_axis], data_grouped[y_axis], marker='o', linestyle='-')
            ax.set_title("Jumlah Pengaduan per Tahun")
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Jumlah Pengaduan")
            st.pyplot(fig)
        else:
            st.warning("Pilih 'periode_data' sebagai X-axis dan 'jumlah_pengaduan' sebagai Y-axis untuk visualisasi ini.")
    
    elif pilih_visualisasi == "Jumlah Pengaduan berdasarkan Wilayah":
        if x_axis == "wilayah" and y_axis == "jumlah_pengaduan":
            data_grouped = data.groupby(x_axis)[y_axis].sum().reset_index()
            fig, ax = plt.subplots()
            ax.bar(data_grouped[x_axis], data_grouped[y_axis], color='skyblue')
            ax.set_title("Jumlah Pengaduan berdasarkan Wilayah")
            ax.set_xlabel("Wilayah")
            ax.set_ylabel("Jumlah Pengaduan")
            ax.set_xticklabels(data_grouped[x_axis], rotation=45, ha="right")
            st.pyplot(fig)
        else:
            st.warning("Pilih 'wilayah' sebagai X-axis dan 'jumlah_pengaduan' sebagai Y-axis untuk visualisasi ini.")
    
    elif pilih_visualisasi == "Distribusi Jenis Kriminal":
        if x_axis == "jenis_kriminal" and y_axis == "jumlah_pengaduan":
            data_grouped = data.groupby(x_axis)[y_axis].sum().reset_index()
            fig, ax = plt.subplots()
            ax.bar(data_grouped[x_axis], data_grouped[y_axis], color='salmon')
            ax.set_title("Distribusi Jenis Kriminal")
            ax.set_xlabel("Jenis Kriminal")
            ax.set_ylabel("Jumlah Pengaduan")
            ax.set_xticklabels(data_grouped[x_axis], rotation=45, ha="right")
            st.pyplot(fig)
        else:
            st.warning("Pilih 'jenis_kriminal' sebagai X-axis dan 'jumlah_pengaduan' sebagai Y-axis untuk visualisasi ini.")
    
    elif pilih_visualisasi == "Asal Pengaduan per Jenis Kriminal":
        if x_axis == "asal_pengaduan" and y_axis == "jumlah_pengaduan":
            data_grouped = data.groupby(x_axis)[y_axis].sum().reset_index()
            fig, ax = plt.subplots()
            ax.bar(data_grouped[x_axis], data_grouped[y_axis], color='green')
            ax.set_title("Asal Pengaduan per Jenis Kriminal")
            ax.set_xlabel("Asal Pengaduan")
            ax.set_ylabel("Jumlah Pengaduan")
            ax.set_xticklabels(data_grouped[x_axis], rotation=45, ha="right")
            st.pyplot(fig)
        else:
            st.warning("Pilih 'asal_pengaduan' sebagai X-axis dan 'jumlah_pengaduan' sebagai Y-axis untuk visualisasi ini.")

    elif pilih_visualisasi == "Trend Bulanan Pengaduan":
        if x_axis == "tanggal_pengaduan" and y_axis == "jumlah_pengaduan":
            # Menghitung jumlah pengaduan per bulan
            trend_bulanan = data_bersih.groupby("bulan_pengaduan")[y_axis].sum().reset_index()
            
            # Visualisasi
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=trend_bulanan, x="bulan_pengaduan", y=y_axis, marker="o", ax=ax)
            ax.set_title("Trend Pengaduan Bulanan")
            ax.set_xlabel("Bulan")
            ax.set_ylabel("Jumlah Pengaduan")
            st.pyplot(fig)
        else:
            st.warning("Pilih 'tanggal_pengaduan' sebagai X-axis dan 'jumlah_pengaduan' sebagai Y-axis untuk visualisasi ini.")
