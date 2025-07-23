import streamlit as st

def show_metadata_viewer():
    st.markdown(
        """
        <div class='sidebar-card' style='background:#3a3a3a;'>
            <b style='color:#1cc88a;'>Hazard Score:</b> <span style='color:#fff;'>-</span><br>
            <b style='color:#f6c23e;'>Elevation:</b> <span style='color:#fff;'>-</span><br>
            <b style='color:#36b9cc;'>Coordinates:</b> <span style='color:#fff;'>-</span><br>
            <b style='color:#e74a3b;'>Proximity:</b> <span style='color:#fff;'>-</span><br>
            <b style='color:#858796;'>Timestamp:</b> <span style='color:#fff;'>-</span>
        </div>
        """,
        unsafe_allow_html=True
    ) 