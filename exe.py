import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Install scikit-learn
st.write("Installing required libraries...")
st.run("pip install -U scikit-learn")

# Load the dataset
df = pd.read_csv("penjualan_barang.csv")

# Streamlit App
st.title("Analisis Penjualan Barang")

# Display the first 5 rows of the dataset
st.subheader("5 Baris Pertama dari Dataset")
st.write(df.head())

# Data Preparation
st.subheader("Data Preparation")

# Handle missing values (if any)
df.dropna(inplace=True)

# Convert tanggal column to datetime
df['tanggal'] = pd.to_datetime(df['tanggal'])

# Perform other preprocessing as needed
# ...

# Modelling
st.subheader("Modelling")

# Select features for clustering (example: kuantum and nominal)
selected_feature = st.selectbox("Pilih Fitur untuk Clustering:", ['kuantum', 'nominal'])
features = df[[selected_feature]]

# Choose the number of clusters (adjust as needed)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the model
df['cluster'] = kmeans.fit_predict(features)

# Display clustered results
st.subheader("Hasil Clustering")
st.write(df[['nama.barang', 'cluster']])

# Additional Analysis
st.subheader("Analisis Tambahan")

# Nama barang dengan penjualan terbanyak
top_barang = df.groupby('nama.barang')[selected_feature].sum().sort_values(ascending=False).head(1)
st.write(f"Nama Barang dengan Penjualan Terbanyak ({selected_feature}):", top_barang.index[0])

# Pembeli yang membeli paling banyak
top_pembeli = df.groupby('nama.pembeli')[selected_feature].sum().sort_values(ascending=False).head(1)
st.write(f"Pembeli yang Membeli Paling Banyak ({selected_feature}):", top_pembeli.index[0])

# Tahun dengan penjualan terbanyak
df['tahun'] = df['tanggal'].dt.year
top_tahun = df.groupby('tahun')[selected_feature].sum().sort_values(ascending=False).head(1)
st.write(f"Tahun dengan Penjualan Terbanyak ({selected_feature}):", top_tahun.index[0])

# Visualization - Pie Chart for Top 5 Nama Pembeli dengan Pembelian Terbanyak
st.subheader(f"Visualisasi: Top 5 Nama Pembeli dengan Pembelian Terbanyak (Berdasarkan {selected_feature})")

# Group data by nama pembeli and calculate total selected feature
top_pembeli_feature = df.groupby('nama.pembeli')[selected_feature].sum().sort_values(ascending=False).head(5)

# Calculate purchase percentage
total_feature = top_pembeli_feature.sum()
top_pembeli_percentage = (top_pembeli_feature / total_feature) * 100

# Plot pie chart using seaborn color palette
plt.figure(figsize=(10, 8))
plt.pie(top_pembeli_percentage, labels=top_pembeli_percentage.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(top_pembeli_percentage)))

plt.title(f'Top 5 Nama Pembeli dengan Pembelian Terbanyak (Berdasarkan {selected_feature})')
st.pyplot(plt)  # Display the plot in Streamlit

# Visualization - Horizontal Bar Chart for Top 5 Nama Barang dengan Pembelian Terbanyak
st.subheader(f"Visualisasi: Top 5 Nama Barang dengan Pembelian Terbanyak (Berdasarkan {selected_feature})")

# Group data by nama barang and calculate total selected feature
top_barang_feature = df.groupby('nama.barang')[selected_feature].sum().sort_values(ascending=False).head(5)

# Plot horizontal bar chart using seaborn with "viridis" palette
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=top_barang_feature.values, y=top_barang_feature.index, palette="viridis")

# Add selected feature values next to the bars
for i, v in enumerate(top_barang_feature.values):
    ax.text(v + 0.1, i, f'${v:,.2f}', color='black', va='center')

plt.title(f'Top 5 Nama Barang dengan Pembelian Terbanyak (Berdasarkan {selected_feature})')
plt.xlabel(f'{selected_feature} Pembelian')
plt.ylabel('Nama Barang')
st.pyplot(plt)  # Display the plot in Streamlit

# Visualization - Bar Chart for Top 5 Nama Barang dengan Pembelian Terbanyak (Berdasarkan Persentase selected feature)
st.subheader(f"Visualisasi: Top 5 Nama Barang dengan Pembelian Terbanyak (Berdasarkan Persentase {selected_feature})")

# Group data by nama barang and calculate total selected feature
top_barang_feature = df.groupby('nama.barang')[selected_feature].sum().sort_values(ascending=False).head(5)

# Calculate purchase percentage
total_feature = top_barang_feature.sum()
top_barang_percentage = (top_barang_feature / total_feature) * 100

# Plot bar chart using seaborn with "viridis" palette
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=top_barang_percentage.values, y=top_barang_percentage.index, palette="viridis")

# Add percentage values above the bars
for i, v in enumerate(top_barang_percentage.values):
    ax.text(v + 0.1, i, f'{v:.1f}%', color='black', va='center')

plt.title(f'Top 5 Nama Barang dengan Pembelian Terbanyak (Berdasarkan Persentase {selected_feature})')
plt.xlabel(f'Persentase {selected_feature} Pembelian')
plt.ylabel('Nama Barang')
st.pyplot(plt)  # Display the plot in Streamlit

# Further analysis or visualizations as needed
# ...

# Evaluation
st.subheader("Evaluasi Model")

# Pilih fitur yang akan digunakan untuk clustering (contoh: kuantum dan nominal)
eval_features = df[['kuantum', 'nominal']]

# Coba beberapa jumlah klaster dan hitung Silhouette Score untuk masing-masing
st.write("Evaluasi Silhouette Score untuk Jumlah Klaster 2 hingga 5:")
for n_clusters in range(2, 6):
    kmeans_eval = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels_eval = kmeans_eval.fit_predict(eval_features)
    silhouette_avg_eval = silhouette_score(eval_features, cluster_labels_eval)
    st.write(f"Jumlah Klaster = {n_clusters}, Silhouette Score = {silhouette_avg_eval}")

# Tentukan jumlah klaster yang diinginkan
n_clusters_eval = st.slider("Pilih Jumlah Klaster untuk Evaluasi Silhouette Score:", 2, 10, 3)  # Sesuaikan dengan jumlah klaster yang diinginkan

# Fitting model K-means
kmeans_eval = KMeans(n_clusters=n_clusters_eval, random_state=42)
cluster_labels_eval = kmeans_eval.fit_predict(eval_features)

# Hitung Silhouette Score untuk seluruh data
silhouette_avg_eval = silhouette_score(eval_features, cluster_labels_eval)

st.write(f"Total Silhouette Score untuk Keseluruhan Cluster: {silhouette_avg_eval}")
