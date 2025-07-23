import streamlit as st
import pandas as pd

st.set_page_config(page_title="Yolanda Risk Zone Classifier", layout="wide")
# Main Title
st.title("Yolanda Risk Zone Classifier -DIP Group 4 2025")

# Section 1: Map Viewer
with st.container():
    st.subheader("🗺️ Map Viewer (UIAxes)")
    st.markdown("- Himawari satellite image\n- Hazard overlays (PHIVOLCS, PAGASA)\n- Shelter markers (GeoAnalyticsPH)\n- Buffer zones (cyan overlay)")
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6e/Himawari8_true_color.png", caption="Sample Himawari Satellite Image", use_column_width=True)

# Section 2: Patch Selector & Classifier
with st.container():
    st.subheader("🟫 Patch Selector (Drop-down or File Browser)")
    patch = st.selectbox("Select Patch", ["Patch 1", "Patch 2", "Patch 3"])
    st.button("Classify Patch")
    st.markdown("### 🧠 Predicted Class Display: ")
    st.info("Predicted Class: [Placeholder]")

# Section 3: Metadata Viewer
with st.container():
    st.subheader("📋 Metadata Viewer (UITable)")
    st.markdown("- Hazard Score\n- Mean Elevation\n- Latitude / Longitude\n- Shelter Proximity\n- Timestamp")
    # Placeholder DataFrame
    df = pd.DataFrame({
        "Hazard Score": [0.8, 0.6, 0.9],
        "Mean Elevation": [12, 15, 8],
        "Latitude": [11.2, 11.3, 11.4],
        "Longitude": [124.9, 125.0, 125.1],
        "Shelter Proximity": [0.5, 0.7, 0.3],
        "Timestamp": ["2024-06-01", "2024-06-02", "2024-06-03"]
    })
    st.dataframe(df)

# Section 4: Batch Analysis Tab
with st.container():
    st.subheader("📈 Batch Analysis Tab")
    st.markdown("- Upload multiple patches\n- View classification heatmap\n- Export results to CSV")
    uploaded_files = st.file_uploader("Upload Patch Files", accept_multiple_files=True)
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")
    st.button("Export results to CSV")
    st.image("https://matplotlib.org/stable/_images/sphx_glr_image_001.png", caption="Sample Classification Heatmap", use_column_width=True) 