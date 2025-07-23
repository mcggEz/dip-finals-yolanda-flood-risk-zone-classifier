import streamlit as st

def show_batch_analysis():
    st.markdown(
        """
        <div class='sidebar-card' style='background:#4b2e5e;'>
            <b style='color:#a78bfa;'>Batch:</b> <span style='color:#fff;'>Upload and analyze multiple patches</span>
            <div style='margin-top:0.7rem;'>
            </div>
            <div style='margin:0.7rem 0; color:#36b9cc;'><b>View classification heatmap</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.file_uploader("Upload multiple patches", accept_multiple_files=True, key="batch_upload")
    st.button("Export results to CSV", key="export_csv_btn") 