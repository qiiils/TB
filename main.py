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
            st.write("Dataset ini terdiri dari 376 baris dan 7 kolom. Atribut yang tersedia :")
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

                st.subheader("Dimensi data")
                st.write("Dimensi data sebelum dibersihkan : ", data.shape)
                st.write("Dimensi data setelah dibersihkan : ", data_bersih.shape)

                st.download_button(
                    label="Download Data Bersih",
                    data=data_bersih.to_csv(index=False).encode('utf-8'),
                    file_name="(Clean Data)_Data_Kriminalitas_yang_Telah_Tertangani_di_Jakarta.csv",
                    mime="text/csv"
                )
            # More about data (gak dipakai)
            # Heatmap korelasi 
            # st.subheader("Correlation Heatmap")
            # numeric_data = data_bersih.select_dtypes(include=['float', 'int'])  # Select only numeric columns
            # if not numeric_data.empty:
            #     fig, ax = plt.subplots()
            #     sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
            #     st.pyplot(fig)

            
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
                # Aggregate data untuk clustering
                total_pengaduan = data_bersih.groupby(['wilayah', 'jenis_kriminal'], as_index=False)['jumlah_pengaduan'].sum()

                # Label Encoding -------------------------------------------------------------------
                le_wilayah = LabelEncoder()
                le_kriminalitas = LabelEncoder()
                total_pengaduan['wilayah_encoded'] = le_wilayah.fit_transform(total_pengaduan['wilayah'])
                total_pengaduan['jenis_kriminalitas_encoded'] = le_kriminalitas.fit_transform(total_pengaduan['jenis_kriminal'])

                # Fitur clustering -------------------------------------------------------------------
                X = total_pengaduan[['wilayah_encoded', 'jumlah_pengaduan']]

                # Pilih berapa banyak cluster -------------------------------------------------------------------
                num_clusters = st.slider("Pilih jumlah cluster", 2, 4, 3)

                # Apply KMeans -------------------------------------------------------------------
                kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                total_pengaduan['cluster'] = kmeans.fit_predict(X)

                # Menampilkan  -------------------------------------------------------------------
                st.subheader("Data dengan Cluster")
                st.write(total_pengaduan)

                # Wilayah and Jenis Kriminalitas code -------------------------------------------------------------------
                wilayah_map = pd.DataFrame({'Wilayah': total_pengaduan['wilayah'], 'Kode Wilayah': total_pengaduan['wilayah_encoded']}).drop_duplicates().sort_values(by='Kode Wilayah')
                kriminalitas_map = pd.DataFrame({'Jenis Kriminalitas': total_pengaduan['jenis_kriminal'], 'Kode Kriminalitas': total_pengaduan['jenis_kriminalitas_encoded']}).drop_duplicates().sort_values(by='Kode Kriminalitas')
   
                # Keterangan kode -------------------------------------------------------------------
                st.subheader("Keterangan Kode Wilayah dan Jenis Kriminalitas")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Kode Wilayah:", wilayah_map)
                with col2: 
                    st.write("Kode Jenis Kriminalitas:", kriminalitas_map)

                # Visualisasi clustering -------------------------------------------------------------------
                # st.subheader("Hasil Clustering Berdasarkan Wilayah dan Jenis Kriminalitas")
                # pivot_table = total_pengaduan.pivot_table(index='wilayah', columns='jenis_kriminal', values='jumlah_pengaduan', aggfunc='sum')
                # plt.figure(figsize=(12, 8))
                # sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt=".0f")
                # plt.title("Jumlah Pengaduan per Wilayah dan Jenis Kriminalitas")
                # st.pyplot()

                plt.figure(figsize=(10, 6))
                sns.barplot(data=total_pengaduan, x="wilayah", y="jumlah_pengaduan", hue="cluster", palette="viridis")
                plt.title("Jumlah Pengaduan per Wilayah Berdasarkan Cluster")
                plt.xlabel("Wilayah")
                plt.ylabel("Jumlah Pengaduan")
                plt.xticks(rotation=45)
                st.pyplot(plt)

                # sns.pairplot(total_pengaduan, vars=["wilayah_encoded", "jumlah_pengaduan"], hue="cluster", palette="viridis")
                # st.pyplot()

                # fig, ax = plt.subplots(figsize=(10, 6))
                # sns.scatterplot(data=total_pengaduan, x='wilayah_encoded', y='jenis_kriminalitas_encoded', size='jumlah_pengaduan', hue='cluster', palette='viridis', ax=ax, sizes=(20, 200))
                # ax.set_title("Clustering Berdasarkan Wilayah dan Jenis Kriminalitas")
                # ax.set_xlabel("Wilayah (Encoded)")
                # ax.set_ylabel("Jenis Kriminalitas (Encoded)")
                # st.pyplot(fig)


                # Observasi Clustering ---------------------------------------------------------
                # 2 cluster -------------------------------------------------------------------
                if num_clusters == 2: 
                    st.subheader("Observasi Clustering")
                    st.write(" - Kluster 0 (biru tua): Bar dalam kluster ini memiliki ukuran yang lebih pendek, hal ini mengindikasikan wilayah dan jenis kriminalitas dengan jumlah laporan yang rendah hingga moderat.")
                    st.write(" - Kluster 1 (tosca): Kluster ini memiliki bar dengan ukuran yang lebih tinggi, mengindikasikan wilayah dan jenis kriminalitas dengan jumlah laporan yang relatif tinggi.")
                # 3 cluster -------------------------------------------------------------------
                elif num_clusters == 3:
                    st.subheader("Observasi Clustering")
                    st.write(" - Kluster 0 (ungu tua): Berisi kombinasi wilayah dan jenis kriminalitas yang umumnya memiliki jumlah laporan moderat hingga rendah, terlihat dari ukuran bar yang agak pendek/menengah.")
                    st.write(" - Kluster 1 (tosca): Pada kluster ini, bar memiliki ukuran yang lebih tinggi dari kluster lainnya, yang menunjukkan wilayah dan jenis kriminalitas dengan jumlah laporan yang tinggi.")
                    st.write(" - Kluster 2 (kuning): Bar dalam kluster ini lebih banyak tersebar dan umumnya memiliki ukuran yang lebih pendek, menunjukkan kombinasi wilayah dan jenis kriminalitas dengan jumlah laporan yang lebih rendah.")
                # 4 cluster -------------------------------------------------------------------
                else:
                    st.subheader("Observasi Clustering")
                    st.write(" - Kluster 0 (ungu tua): Bar pada kluster ini memiliki ukuran menengah, yang menunjukkan bahwa wilayah dan jenis kriminalitas yang termasuk dalam kelompok ini memiliki jumlah pengaduan yang lebih sedikit dibandingkan kluster tertinggi, namun lebih banyak daripada kluster lain.")
                    st.write(" - Kluster 1 (biru tua): Ukuran bar pada kluster ini lebih tinggi dibandingkan dengan kluster lainnya, menandakan bahwa jumlah pengaduan yang ada pada wilayah yang memiliki kluster 1 tergolong relatif tinggi.")
                    st.write(" - Kluster 2 (tosca): Bar dengan penyebaran yang lebih banyak dibandingkan bar kluster lain, dengan ukuran yang beragam namun tidak mencapai di atas 5 pengaduan, tetapi sebagian besar berukuran kecil. Hal ini menunjukkan bahwa kombinasi wilayah dan jenis kriminalitas dalam kluster ini umumnya memiliki jumlah laporan yang lebih rendah, meskipun tersebar hampir merata di beberapa wilayah.")
                    st.write(" - Kluster 3 (kuning): Bar dalam kluster ini tidak memilki perbedaan ukuran bar yang signifikan, hampir sama.")
            else:
                st.error("File CSV tidak memiliki kolom 'wilayah', 'jenis_kriminal', dan 'jumlah_pengaduan' yang diperlukan.")
        else:
            st.warning("Silakan unggah file CSV terlebih dahulu")

           
# Tab 4 : Visualisasi -------------------------------------------------------------------
with tab4:
    if "data_bersih" in st.session_state and st.session_state["data_bersih"] is not None:
        data_bersih = st.session_state["data_bersih"]

        x_axis_options = ["periode_data", "wilayah", "jenis_kriminal", "asal_pengaduan", "tanggal_pengaduan"]
        y_axis_options = ["jumlah_pengaduan"]
        
        # Pilihan jenis visualisasi -------------------------------------------------------------------
        pilih_visualisasi = st.selectbox(
            "Pilih jenis visualisasi:",
            ["Jumlah Pengaduan per Tahun", "Jumlah Pengaduan berdasarkan Wilayah", 
            "Distribusi Jenis Kriminal", "Asal Pengaduan per Jenis Kriminal", "Trend Bulanan Pengaduan"]
        )
        
        x_axis = st.selectbox("Pilih X-axis:", x_axis_options)
        y_axis = st.selectbox("Pilih Y-axis:", y_axis_options)
        

        if pilih_visualisasi == "Jumlah Pengaduan per Tahun":
            if x_axis == "periode_data" and y_axis == "jumlah_pengaduan":
                data_grouped = data_bersih.groupby(x_axis)[y_axis].sum().reset_index()
                fig, ax = plt.subplots()
                ax.plot(data_grouped[x_axis], data_grouped[y_axis], marker='o', linestyle='-')
                ax.set_title("Jumlah Pengaduan per Tahun")
                ax.set_xlabel("Tahun")
                ax.set_ylabel("Jumlah Pengaduan")
                st.pyplot(fig)
            else:
                st.warning("Pilih 'periode_data' sebagai X-axis dan 'jumlah_pengaduan' sebagai Y-axis untuk visualisasi ini.")
        
        #Pilihan: Jumlah Pengaduan Berdasarkan Wilayah -------------------------------------------------------------------
        elif pilih_visualisasi == "Jumlah Pengaduan berdasarkan Wilayah":
            if x_axis == "wilayah" and y_axis == "jumlah_pengaduan":
                data_grouped = data_bersih.groupby(x_axis)[y_axis].sum().reset_index()
                fig, ax = plt.subplots()
                ax.bar(data_grouped[x_axis], data_grouped[y_axis], color='skyblue')
                ax.set_title("Jumlah Pengaduan berdasarkan Wilayah")
                ax.set_xlabel("Wilayah")
                ax.set_ylabel("Jumlah Pengaduan")
                ax.set_xticklabels(data_grouped[x_axis], rotation=45, ha="right")
                st.pyplot(fig)

                st.write("Penjelasan: ")
                st.write("Pada visualisasi Jumlah Pengaduan berdasarkan wilayah, dapat dilihat bahwa wilayah dengan jumlah pengaduan tertinggi adalah Kota Adm. Jakarta Timur, mencapai total 80 pengaduan. Sedangkan wilayah dengan jumlah pengaduan paling sedikit adalah Jakarta Pusat. Disarankan agar penjagaan ketertiban dan kemanan di wilayah Kota administratif, terlebih Kota Adm. Jakarta Timur, untuk ditingkatkan.")
            else:
                st.warning("Pilih 'wilayah' sebagai X-axis dan 'jumlah_pengaduan' sebagai Y-axis untuk visualisasi ini.")
        
        #Pilihan: Distribusi Jenis Kriminal -------------------------------------------------------------------
        elif pilih_visualisasi == "Distribusi Jenis Kriminal":
            if x_axis == "jenis_kriminal" and y_axis == "jumlah_pengaduan":
                data_grouped = data_bersih.groupby(x_axis)[y_axis].sum().reset_index()
                fig, ax = plt.subplots()
                ax.bar(data_grouped[x_axis], data_grouped[y_axis], color='salmon')
                ax.set_title("Distribusi Jenis Kriminal")
                ax.set_xlabel("Jenis Kriminal")
                ax.set_ylabel("Jumlah Pengaduan")
                ax.set_xticklabels(data_grouped[x_axis], rotation=45, ha="right")
                st.pyplot(fig)

                st.write("Penjelasan: ")
                st.write("Dari visualisasi yang tersedia dapat dilihat bahwa angka Tawuran dan Kriminalitas menjadi Top 2 Jenis Kriminal yang sering terjadi. Sedangkan pengaduan untuk jenis kriminal penipuan lebih jarang dilaporkan.")
            else:
                st.warning("Pilih 'jenis_kriminal' sebagai X-axis dan 'jumlah_pengaduan' sebagai Y-axis untuk visualisasi ini.")
        
        #Pilihan: Asal Pengaduan per Jenis Kriminal -------------------------------------------------------------------
        elif pilih_visualisasi == "Asal Pengaduan per Jenis Kriminal":
            if x_axis == "asal_pengaduan" and y_axis == "jumlah_pengaduan":
                data_grouped = data_bersih.groupby(x_axis)[y_axis].sum().reset_index()
                fig, ax = plt.subplots()
                ax.bar(data_grouped[x_axis], data_grouped[y_axis], color='green')
                ax.set_title("Asal Pengaduan per Jenis Kriminal")
                ax.set_xlabel("Asal Pengaduan")
                ax.set_ylabel("Jumlah Pengaduan")
                ax.set_xticklabels(data_grouped[x_axis], rotation=45, ha="right")
                st.pyplot(fig)

                st.write("Penjelasan: ")
                st.write("Dari visualisasi yang ditampilkan dapat diketahui bahwa orang-orang lebih nyaman jika melaporkan kriminalitas melaui aplikasi JAKI atau langsung melaporkan ke kantor kelurahan. Hampir nyaris tidak ada yang melaporkan kriminalitas melalui 112. Untuk itu, disarankan untuk dapat meningkatkan layanan pengaduan di aplikasi JAKI dan kantor kelurahan.")
            else:
                st.warning("Pilih 'asal_pengaduan' sebagai X-axis dan 'jumlah_pengaduan' sebagai Y-axis untuk visualisasi ini.")

        #Pilihan: Trend Bulanan Pengaduan -------------------------------------------------------------------
        elif pilih_visualisasi == "Trend Bulanan Pengaduan":
            if x_axis == "tanggal_pengaduan" and y_axis == "jumlah_pengaduan":
                # Convert tanggal_pengaduan ke datetime -------------------------------------------------------------------
                data_bersih['tanggal_pengaduan'] = pd.to_datetime(data_bersih['tanggal_pengaduan'], errors='coerce')
                data_bersih['bulan_pengaduan'] = data_bersih['tanggal_pengaduan'].dt.month
                    
                trend_bulanan = data_bersih.groupby("bulan_pengaduan")[y_axis].sum().reset_index()

                # Visualisasi -------------------------------------------------------------------
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=trend_bulanan, x="bulan_pengaduan", y=y_axis, marker="o", ax=ax)
                ax.set_title("Trend Pengaduan Bulanan")
                ax.set_xlabel("Bulan")
                ax.set_ylabel("Jumlah Pengaduan")
                st.pyplot(fig)

                st.write("Penjelasan: ")
                st.write("Pengaduan di bulan April terjadi kenaikan sampai hampir mencapai 80 pengaduan, namun turun kembali selama 4 bulan setelahnya. Akan tetapi, pada bulan september, pengaduan meningkat secara pesat, perlu penyelidikan lebih lanjut mengapa kriminalitas pada bulan September meningkat secara tidak normal.")
            else:
                st.warning("Pilih 'tanggal_pengaduan' sebagai X-axis dan 'jumlah_pengaduan' sebagai Y-axis untuk visualisasi ini.")

    else:
        st.warning("Silakan unggah file CSV terlebih dahulu")