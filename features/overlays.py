import streamlit as st

def show_overlays():
    st.markdown(
        """
        <div class='sidebar-card' style='background:#223a5e;'>
            <b style='color:#1cc88a;'>🌀 Overlay:</b> <span style='color:#fff;'>Flood-prone areas (map overlays)</span>
        </div>
        <div class='sidebar-card' style='background:#4e4e2e;'>
            <b style='color:#f6c23e;'>🏠 Marker:</b> <span style='color:#fff;'>Emergency shelter locations</span>
        </div>
        <div class='sidebar-card' style='background:#225e5e;'>
            <b style='color:#36b9cc;'>🔵 Zone:</b> <span style='color:#fff;'>Cyan buffer/safe zones</span>
        </div>
        """,
        unsafe_allow_html=True
    ) 