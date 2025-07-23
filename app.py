import streamlit as st

# Set page config
st.set_page_config(page_title="Yolanda Risk Zone Classifier", layout="wide")

# Reduce main block padding for a tighter layout
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.6rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
    }
    [data-testid="stSidebar"] {
        background: #181c23 !important;
        color: #fff !important;
    }
    /* Sidebar header styling */
    [data-testid="stSidebar"] h1 {
        background: linear-gradient(90deg, #4e73df 60%, #1cc88a 100%);
        color: #fff !important;
        padding: 1.2rem 1rem 1rem 1rem;
        border-radius: 14px 14px 0 0;
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: 1px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(44,62,80,0.07);
    }
    /* Sidebar subheader styling */
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #4e73df !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
        text-align: center;
        letter-spacing: 0.5px;
        font-style: italic;
        background: #eaf1fb;
        border-radius: 0 0 14px 14px;
        padding: 0.5rem 0.5rem 0.7rem 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header


# Sidebar with expanders for each feature
with st.sidebar:
    # Logo/Title
    st.markdown(
        """
        <div style='text-align:center; margin-bottom:1rem;'>
            <span style='font-size:1.5rem; font-weight:800; color:#4e73df;'>Yolanda</span> 
            <span style='font-size:1.5rem; color:#fff;'>Flood Risk Zone Classifier</span>
        </div>

        <div style='text-align:center; margin-bottom:1rem;'>
            <span style='font-size:1.1rem; color:#fff;'>Descritpion about the app....</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    

  
    with st.expander("🗺️ Overlays, Markers and Zones  View and toggle map overlays, shelter markers, and buffer zones for risk visualization." , expanded=True):
        st.markdown(
            """
            <div style='background:#23272f; border-radius:12px; box-shadow:0 2px 8px #0002; margin-bottom:1rem; padding:1.2rem 1.4rem; font-size:1.08rem;'>
                <b style='color:#1cc88a;'>🌀 Flood overlays</b><br>
                <span style='color:#fff;'>Map overlays for flood-prone areas.</span>
            </div>
            <div style='background:#23272f; border-radius:12px; box-shadow:0 2px 8px #0002; margin-bottom:1rem; padding:1.2rem 1.4rem; font-size:1.08rem;'>
                <b style='color:#f6c23e;'>🏠 Shelter markers</b><br>
                <span style='color:#fff;'>Locations of emergency shelters.</span>
            </div>
            <div style='background:#23272f; border-radius:12px; box-shadow:0 2px 8px #0002; margin-bottom:1rem; padding:1.2rem 1.4rem; font-size:1.08rem;'>
                <b style='color:#36b9cc;'>🔵 Buffer zones</b><br>
                <span style='color:#fff;'>Cyan overlays for buffer/safe zones.</span>
            </div>
            """,
            unsafe_allow_html=True
        )


    with st.expander("🧩 Patch Selectors Select or upload a patch for classification and risk analysis.", expanded=False):
        st.markdown(
            """
            <div style='background:#23272f; border-radius:12px; box-shadow:0 2px 8px #0002; margin-bottom:1rem; padding:1.2rem 1.4rem; font-size:1.08rem;'>
                <b style='color:#1cc88a;'>Select Patch</b><br>
                <span style='color:#fff;'>Choose a patch from the dropdown or upload your own.</span>
                <div style='margin-top:0.7rem;'>
                    <div style='margin-bottom:0.7rem;'>
                        <span style='color:#f6c23e;'>Dropdown:</span><br>
                        {dropdown}
                    </div>
                    <div style='margin-bottom:0.7rem;'>
                        <span style='color:#f6c23e;'>Upload:</span><br>
                        {uploader}
                    </div>
                    {button}
                    <div style='margin-top:0.7rem; color:#36b9cc;'><b>Predicted Class:</b> ...</div>
                </div>
            </div>
            """.format(
                dropdown=st.selectbox("", ["Patch 1", "Patch 2"], key="patch_select"),
                uploader=st.file_uploader("", key="patch_upload"),
                button=st.button("Classify Patch", key="classify_patch_btn")
            ),
            unsafe_allow_html=True
        )


    with st.expander("📋 Metadata Viewer   View detailed metadata for the selected patch, including hazard score and location info.", expanded=False):
        st.markdown(
            """
            <div style='background:#23272f; border-radius:12px; box-shadow:0 2px 8px #0002; margin-bottom:1rem; padding:1.2rem 1.4rem; font-size:1.08rem;'>
                <b style='color:#1cc88a;'>Hazard Score:</b> <span style='color:#fff;'>-</span><br>
                <b style='color:#f6c23e;'>Mean Elevation:</b> <span style='color:#fff;'>-</span><br>
                <b style='color:#36b9cc;'>Latitude / Longitude:</b> <span style='color:#fff;'>-</span><br>
                <b style='color:#e74a3b;'>Shelter Proximity:</b> <span style='color:#fff;'>-</span><br>
                <b style='color:#858796;'>Timestamp:</b> <span style='color:#fff;'>-</span>
            </div>
            """,
            unsafe_allow_html=True
        )


    with st.expander("📊 Batch Analysis  Upload and analyze multiple patches at once, view heatmaps, and export results.", expanded=False):
        st.markdown(
            """
            <div style='background:#23272f; border-radius:12px; box-shadow:0 2px 8px #0002; margin-bottom:1rem; padding:1.2rem 1.4rem; font-size:1.08rem;'>
                <b style='color:#1cc88a;'>Upload multiple patches</b><br>
                <span style='color:#fff;'>Select and upload several patches for batch analysis.</span>
                <div style='margin-top:0.7rem;'>
                    {batch_uploader}
                </div>
                <div style='margin:0.7rem 0; color:#36b9cc;'><b>View classification heatmap</b></div>
                {export_btn}
            </div>
            """.format(
                batch_uploader=st.file_uploader("", accept_multiple_files=True, key="batch_upload"),
                export_btn=st.button("Export results to CSV", key="export_csv_btn")
            ),
            unsafe_allow_html=True
        )

    # Note and Members
    st.markdown(
        """
        <div style='font-size:0.95rem; color:#aaa; font-style:italic; margin-bottom:0.5rem;'>
        This is for partial requirement for the course DIP Computer Engineering.<br><br>
        <b>Members:</b><br>
        <ul style='color:#fff; font-size:0.98rem; margin-top:0;'>
        <li>Mc Giberri M. Ginez</li>
        <li>Carlos San Gabriel</li>
        <li>Kurth Angelo Espiritu</li>
        <li>Mary Angelique Terre</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

 


# Map placeholder with white background
st.markdown(
    """
    <div style='background: white; border: 2px solid #ddd; height: calc(100vh - 60px); min-height: 400px; display: flex; align-items: center; justify-content: center; margin-top: 0px;'>
        <span style='color: #888; font-size: 1.5rem;'>[Map Placeholder]</span>
    </div>
    """,
    unsafe_allow_html=True
)

# (Optional) Footer
# st.markdown(
#     "<div style='background-color: #808080; padding: 20px; text-align: center; color: white; font-size: 18px; font-weight: bold;'>Footer Content Here</div>",
#     unsafe_allow_html=True
# )



