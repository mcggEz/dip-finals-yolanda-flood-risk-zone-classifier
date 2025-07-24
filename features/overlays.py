import streamlit as st
from streamlit_folium import st_folium
import folium

# Sample overlay data
FLOOD_POLYGON = [
    [13.0, 123.0], [13.0, 124.0], [12.0, 124.0], [12.0, 123.0], [13.0, 123.0]
]
SAFEZONE_POLYGON = [
    [11.5, 121.5], [11.5, 122.5], [10.5, 122.5], [10.5, 121.5], [11.5, 121.5]
]
SHELTER_POINTS = [
    [12.5, 122.5], [12.7, 122.7], [12.3, 122.3]
]

def show_overlays():
    st.markdown(
        """
        <div class='sidebar-card' style='background:#223a5e;'>
            <b style='color:#1cc88a;'>🌀 Overlay:</b> <span style='color:#fff;'>Flood-prone areas (map overlays)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_flood = st.checkbox("Show", key="show_flood")
    st.markdown(
        """
        <div class='sidebar-card' style='background:#4e4e2e;'>
            <b style='color:#f6c23e;'>🏠 Marker:</b> <span style='color:#fff;'>Emergency shelter locations</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_shelters = st.checkbox("Show", key="show_shelters")
    st.markdown(
        """
        <div class='sidebar-card' style='background:#225e5e;'>
            <b style='color:#36b9cc;'>🔵 Zone:</b> <span style='color:#fff;'>Cyan buffer/safe zones</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_buffer = st.checkbox("Show", key="show_buffer")
    return show_flood, show_shelters, show_buffer

def render_overlay_main_content(show_flood, show_shelters, show_buffer):
    # Center the map on the Philippines with ESRI satellite basemap
    m = folium.Map(location=[12.5, 122.5], zoom_start=6, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    # Flood-prone area polygon
    if show_flood:
        folium.Polygon(
            locations=[[13.0, 123.0], [13.0, 124.0], [12.0, 124.0], [12.0, 123.0]],
            color="blue",
            fill=True,
            fill_opacity=0.4,
            popup="Flood-prone Area"
        ).add_to(m)

    # Safe zone polygon
    if show_buffer:
        folium.Polygon(
            locations=[[11.5, 121.5], [11.5, 122.5], [10.5, 122.5], [10.5, 121.5]],
            color="cyan",
            fill=True,
            fill_opacity=0.4,
            popup="Safe Zone"
        ).add_to(m)

    # Emergency shelter markers
    if show_shelters:
        for lat, lon in [[12.5, 122.5], [12.7, 122.7], [12.3, 122.3]]:
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="green", icon="home"),
                popup="Emergency Shelter"
            ).add_to(m)

    st_folium(m, width=1200, height=550)

    # Overlay labels
    overlay_labels = []
    if show_flood:
        overlay_labels.append("🌀 Flood")
    if show_shelters:
        overlay_labels.append("🏠 Shelters")
    if show_buffer:
        overlay_labels.append("🔵 Buffer")
    if overlay_labels:
        st.markdown(f"<div style='color:#1cc88a; font-size:1.2rem;'>Active overlays: {', '.join(overlay_labels)}</div>", unsafe_allow_html=True) 