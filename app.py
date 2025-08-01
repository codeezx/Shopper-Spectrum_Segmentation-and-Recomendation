import streamlit as st
import pandas as pd
import pickle

# -------------------- Load Models --------------------
@st.cache_data
def load_similarity_matrix():
    return pd.read_csv("item_similarity_matrix.csv", index_col=0)

@st.cache_resource
def load_model():
    model = pickle.load(open("rfm_kmeans.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

similarity_df = load_similarity_matrix()
kmeans_model, scaler = load_model()

# -------------------- UI Setup --------------------
st.set_page_config(page_title="Shopper Spectrum", layout="centered")
st.title(" Shopper Spectrum - E-Commerce Dashboard")
st.markdown("Customer Segmentation and Product Recommendations")

# -------------------- Product Recommendation Module --------------------
st.header(" Product Recommendation")
product_code = st.text_input("Enter Product Code (e.g., 85123A)")

if st.button("Get Recommendations"):
    if product_code in similarity_df.columns:
        top_similar = similarity_df[product_code].sort_values(ascending=False)[1:6]
        st.success("Top 5 Recommended Products:")
        for i, item in enumerate(top_similar.index.tolist(), 1):
            st.markdown(f"{i}. **{item}**")
    else:
        st.warning(" Product not found. Please check the code.")

# -------------------- Customer Segmentation Module --------------------
st.header(" Customer Segmentation")
rec = st.number_input("Recency (days since last purchase)", 1, 1000, step=1, value=30)
freq = st.number_input("Frequency (number of purchases)", 1, 100, step=1, value=10)
mon = st.number_input("Monetary Value (total spend)", 1, 10000, step=10, value=500)

if st.button("Predict Customer Segment"):
    user_data = scaler.transform([[rec, freq, mon]])
    cluster_label = kmeans_model.predict(user_data)[0]
    segment_map = {
        0: "High-Value",
        1: "Regular",
        2: "Occasional",
        3: "At-Risk"
    }
    label = segment_map.get(cluster_label, f"Cluster {cluster_label}")
    st.success(f"Predicted Segment: **{label}**")
