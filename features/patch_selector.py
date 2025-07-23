import streamlit as st

def show_patch_selector():
    st.markdown(
        """
        <div class='sidebar-card' style='background:#2e5e22;'>
            <b style='color:#1cc88a;'>Patch:</b> <span style='color:#fff;'>Select or upload a patch for analysis</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.selectbox("Select Patch", ["Patch 1", "Patch 2"], key="patch_select")
    st.file_uploader("Or upload a patch", key="patch_upload")
    st.button("Classify Patch", key="classify_patch_btn")
    st.markdown(
        "<div style='margin-top:0.7rem; color:#36b9cc;'><b>Predicted Class:</b> ...</div>",
        unsafe_allow_html=True
    ) 