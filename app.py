import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="COVID-19 Clustering & Classification", layout="wide")
st.sidebar.title("📊 COVID-19 Analysis")

# Dropdown Analisis
option = st.sidebar.selectbox("Pilih Analisis", ["Clustering (KMeans)", "Klasifikasi (Logistic Regression)"])

# Load dataset
try:
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()  # Hilangkan spasi di nama kolom
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

# Validasi kolom penting
required_columns = ['Total Cases', 'Total Deaths']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Kolom berikut tidak ditemukan dalam dataset: {', '.join(missing_columns)}")
    st.stop()

# Normalisasi fitur
X_raw = df[['Total Cases', 'Total Deaths']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Hitung inertia untuk berbagai jumlah cluster
inertias = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_raw)
    inertias.append(kmeans.inertia_)

# Hitung quantile
cases_q = df['Total Cases'].quantile([0.33, 0.66])
deaths_q = df['Total Deaths'].quantile([0.33, 0.66])

# Fungsi 'status' berdasarkan quantile untuk perhitungan klasifikasi
def assign_status(row):
    if row['Total Deaths'] > deaths_q[0.66] or row['Total Cases'] > cases_q[0.66]:
        return 'High'
    elif row['Total Deaths'] > deaths_q[0.33] or row['Total Cases'] > cases_q[0.33]:
        return 'Medium'
    else:
        return 'Low'

# Tambahkan kolom Status untuk klasifikasi
if 'Status' not in df.columns:
    df['Status'] = df.apply(assign_status, axis=1)

# 1. METODE CLUSTERING (KMEANS)
if option == "Clustering (KMeans)":
    st.title("🧪 Clustering COVID-19")

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Mapping label risiko berdasarkan urutan centroid
    centroids = kmeans.cluster_centers_
    centroid_sums = centroids.sum(axis=1)
    sorted_indices = centroid_sums.argsort()
    cluster_labels = {sorted_indices[0]: 'Low', sorted_indices[1]: 'Medium', sorted_indices[2]: 'High'}
    df['Cluster_Label'] = df['Cluster'].map(cluster_labels)


    # Elbow Method untuk menentukan jumlah cluster optimal
    st.write("Mengelompokkan provinsi berdasarkan jumlah kasus dan kematian COVID-19 di tahun 2022")

    # Plot Elbow
    fig_elbow, ax = plt.subplots()
    ax.plot(K_range, inertias, marker='o')
    ax.set_title("Elbow Method")
    ax.set_xlabel("Jumlah Cluster (k)")
    ax.set_ylabel("Inertia (Total Within-Cluster Sum of Squares)")
    st.pyplot(fig_elbow)

    st.markdown(""" Titik elbow pada grafik mengindikasikan jumlah cluster yang ideal dalam proses clustering.""")

    # Visualisasi hasil clustering
    plt.figure(figsize=(10, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis')
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
    plt.xlabel("Total Cases (scaled)")
    plt.ylabel("Total Deaths (scaled)")
    plt.title("Hasil Clustering dengan KMeans")
    plt.legend()
    st.pyplot(plt)

    # Menampilkan Hasil Semua Clustering
    st.write("### Hasil Clustering")
    if 'Location' in df.columns:
        st.dataframe(df[['Location', 'Total Cases', 'Total Deaths', 'Cluster_Label']])
    else:
        st.dataframe(df[['Location', 'Total Cases', 'Total Deaths', 'Cluster_Label']])
    
    # Tabel Jumlah data per cluster
    cluster_counts = df['Cluster_Label'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster (Risiko)', 'Jumlah Data']
    st.dataframe(cluster_counts, use_container_width=True, hide_index=True)

    # Filter berdasarkan nama provinsi
    if 'Location' in df.columns:
        prov_input = st.text_input("Filter Provinsi", value="")
        if prov_input:
            mask = df['Location'].str.contains(prov_input, case=False, na=False)
            df_filtered = df[mask].sort_values(by='Date').reset_index(drop=True)
            st.write(f"### Hasil Clustering untuk provinsi “{prov_input}”")
            st.dataframe(df_filtered[['Date', 'Location', 'Total Cases', 'Total Deaths', 'Cluster_Label']])


# 2. METODE KLASIFIKASI
elif option == "Klasifikasi (Logistic Regression)":
    st.title("📈 Klasifikasi COVID-19")

    # Encoding
    label_encoder = LabelEncoder()
    df['Status_encoded'] = label_encoder.fit_transform(df['Status'])

    X = X_scaled
    y = df['Status_encoded']

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    try:
        # Split data dengan stratifikasi label
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Cek jumlah kelas
        if len(set(y_train)) < 2:
            st.error("Data training hanya mengandung satu kelas. Tambahkan data yang lebih bervariasi.")
        else:
            # Latih model regresi logistik
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            # Evaluasi model
            report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].round(2)
            report_df['support'] = report_df['support'].astype(int)
            st.markdown("<h3 style='text-align: center;'>Hasil Evaluasi Model</h3>", unsafe_allow_html=True)
            st.dataframe(report_df, use_container_width=True)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            labels = label_encoder.classes_
            st.write("### Confusion Matrix")
            fig_cm, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig_cm)

            # 🔍 Prediksi Risiko Berdasarkan Input Manual
            st.markdown("---")
            st.subheader("🔍 Prediksi Potensi Risiko Covid-19")

            with st.form("manual_input_form"):
                total_cases_input = st.number_input("Total Kasus COVID-19", min_value=0, step=1)
                total_deaths_input = st.number_input("Total Kematian COVID-19", min_value=0, step=1)
                submitted = st.form_submit_button("Prediksi Risiko")

            if submitted:
                # Skala input seperti data training
                input_scaled = scaler.transform([[total_cases_input, total_deaths_input]])
                pred_encoded = model.predict(input_scaled)[0]
                pred_label = label_encoder.inverse_transform([pred_encoded])[0]

                st.success(f"✅ Prediksi Risiko: **{pred_label}**")

    except ValueError as e:
        st.error(f"Terjadi error saat training model: {e}")