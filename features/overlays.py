import streamlit as st
from streamlit_folium import st_folium
import folium
import geopandas as gpd
import pandas as pd

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
    st.markdown(
        """
        <div class='sidebar-card' style='background:#5e2222;'>
            <b style='color:#e74c3c;'>🟥 Overlay:</b> <span style='color:#fff;'>Hazard Exposure Zones (PHIVOLCS)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_hazard = st.checkbox("Show", key="show_hazard")
    st.markdown(
        """
        <div class='sidebar-card' style='background:#228B22;'>
            <b style='color:#39e639;'>🟩 Marker:</b> <span style='color:#fff;'>Evacuation Centers and Relief Hubs</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_evac = st.checkbox("Show", key="show_evac")
    return show_flood, show_shelters, show_buffer, show_hazard, show_evac

def render_overlay_main_content(show_flood, show_shelters, show_buffer, show_hazard, show_evac):
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
            color="red",
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

    # Hazard Exposure Zones (PHIVOLCS) from shapefile
    if show_hazard:
        try:
            hazard_gdf = gpd.read_file('overlays/ph.shp')
            # Ensure CRS is set from .prj, or default to WGS84
            if hazard_gdf.crs is None:
                hazard_gdf.set_crs(epsg=4326, inplace=True)
            # Reproject to WGS84 for folium if needed
            if hazard_gdf.crs.to_epsg() != 4326:
                hazard_gdf = hazard_gdf.to_crs(epsg=4326)
            folium.GeoJson(
                hazard_gdf,
                name='Hazard Exposure Zones',
                style_function=lambda x: {
                    'fillColor': 'red',
                    'color': 'red',
                    'weight': 2,
                    'fillOpacity': 0.4
                },
                tooltip=folium.GeoJsonTooltip(fields=[col for col in hazard_gdf.columns if col != 'geometry'])
            ).add_to(m)
        except Exception as e:
            st.error(f"Error loading hazard shapefile: {e}")

    # Evacuation Centers and Relief Hubs from CSV
    if show_evac:
        try:
            evac_df = pd.read_csv('overlays/evacuation_centers.csv')
            for _, row in evac_df.iterrows():
                folium.CircleMarker(
                    location=[row['LAT'], row['LONG']],
                    radius=8,
                    color='green',
                    fill=True,
                    fill_color='green',
                    fill_opacity=0.8,
                    popup=f"<b>{row['NAME_1']}</b><br>Region: {row['REGION']}<br>Evac Centers: {row['Evac_Cntrs']}"
                ).add_to(m)
        except Exception as e:
            st.error(f"Error loading evacuation centers: {e}")

    st_folium(m, width=None, height=600)

    # Overlay labels
    overlay_labels = []
    if show_flood:
        overlay_labels.append("🌀 Flood")
    if show_shelters:
        overlay_labels.append("🏠 Shelters")
    if show_buffer:
        overlay_labels.append("🔵 Buffer")
    if show_hazard:
        overlay_labels.append("🟥 Hazard Zones")
    if show_evac:
        overlay_labels.append("🟩 Evacuation Centers")
    if overlay_labels:
        st.markdown(f"<div style='color:#1cc88a; font-size:1.2rem;'>Active overlays: {', '.join(overlay_labels)}</div>", unsafe_allow_html=True) 