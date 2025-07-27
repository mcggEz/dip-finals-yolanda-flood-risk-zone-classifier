import streamlit as st
from features.overlays import show_overlays, render_overlay_main_content
from features.patch_selector import show_patch_selector, display_metadata_and_export, display_batch_metadata_and_export


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
            <span style='font-size:1rem; color:#fff;'>
                    An interactive dashboard for visualizing, analyzing, and classifying disaster risk zones affected by Typhoon Yolanda using satellite imagery, hazard overlays, and deep learning.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Overlay controls
    show_hazard, show_pagasa, show_evac, show_buffer, show_hazard_vs_warning, hazard_vs_warning_opacity, show_phivolcs_hazard, phivolcs_hazard_opacity, show_hazard_vs_warning_boundary = show_overlays()

    # Patch Selector
    show_patch_selector()

    # Note and Members
    st.markdown(
        """
        <div style='font-size:0.95rem; color:#aaa; font-style:italic; margin-bottom:0.5rem;'>
        This is for partial requirement for the Digital Image Processing course in Computer Engineering at Pamantasan ng Lungsod ng Maynila.<br><br>
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

# Main content
render_overlay_main_content(show_hazard, show_pagasa, show_evac, show_buffer, show_hazard_vs_warning, hazard_vs_warning_opacity, show_phivolcs_hazard, phivolcs_hazard_opacity, show_hazard_vs_warning_boundary)

# ---- Patch Analysis Results in Main Content ----
if (
    'patch_uploaded_files' in st.session_state and
    st.session_state['patch_uploaded_files'] and
    'patch_analysis_trigger' in st.session_state and
    st.session_state['patch_analysis_trigger']
):
    uploaded_files = st.session_state['patch_uploaded_files']
    trigger = st.session_state['patch_analysis_trigger']
    st.markdown("---")
    st.markdown("## 🖼️ Patch Analysis Results")
    if trigger == 'single' and len(uploaded_files) == 1:
        uploaded_file = uploaded_files[0]
        # Show preview
        from PIL import Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Patch", use_container_width=True)
        display_metadata_and_export(uploaded_file.name, "uploaded_file")
    elif trigger == 'batch' and len(uploaded_files) > 1:
        display_batch_metadata_and_export(uploaded_files)




