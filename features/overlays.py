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
        <div class='sidebar-card' style='background:#5e2222;'>
            <b style='color:#e74c3c;'>🟥 Overlay:</b> <span style='color:#fff;'>Hazard overlays (PHIVOLCS, PAGASA)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_hazard = st.checkbox("Show", key="show_hazard")
    st.markdown(
        """
        <div class='sidebar-card' style='background:#228B22;'>
            <b style='color:#39e639;'>🟩 Marker:</b> <span style='color:#fff;'>Shelter markers (GeoAnalyticsPH)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_evac = st.checkbox("Show", key="show_evac")
    st.markdown(
        """
        <div class='sidebar-card' style='background:#225e5e;'>
            <b style='color:#36b9cc;'>🔵 Zone:</b> <span style='color:#fff;'>Buffer zones (cyan overlay)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_buffer = st.checkbox("Show", key="show_buffer")
    return show_hazard, show_evac, show_buffer

def render_overlay_main_content(show_hazard, show_evac, show_buffer):
    # Center the map on the Philippines with ESRI satellite basemap
    m = folium.Map(location=[12.5, 122.5], zoom_start=6, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    # Hazard overlays (PHIVOLCS, PAGASA) from shapefile
    if show_hazard:
        try:
            hazard_gdf = gpd.read_file('overlays/ph.shp')
            if hazard_gdf.crs is None:
                hazard_gdf.set_crs(epsg=4326, inplace=True)
            if hazard_gdf.crs.to_epsg() != 4326:
                hazard_gdf = hazard_gdf.to_crs(epsg=4326)
            folium.GeoJson(
                hazard_gdf,
                name='Hazard Overlays',
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

    # Shelter markers (GeoAnalyticsPH) from CSV
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

    # Buffer zones (cyan overlay) as circles
    if show_buffer:
        buffer_points = [
            [11.00, 122.80], [11.20, 124.90], [10.60, 125.10], [11.50, 125.10], [10.90, 122.70],
            [11.30, 123.00], [10.80, 124.00], [11.60, 124.20], [10.40, 123.50], [11.10, 125.30],
            [11.70, 124.80], [10.20, 122.90],
        ]
        for lat, lon in buffer_points:
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color='cyan',
                fill=True,
                fill_color='cyan',
                fill_opacity=0.6,
                popup="Safe Zone Buffer Point"
            ).add_to(m)

    st_folium(m, width=None, height=600)

    # Overlay labels
    overlay_labels = []
    if show_hazard:
        overlay_labels.append("🟥 Hazard Overlays")
    if show_evac:
        overlay_labels.append("🟩 Shelter Markers")
    if show_buffer:
        overlay_labels.append("🔵 Buffer Zones")
    if overlay_labels:
        st.markdown(f"<div style='color:#1cc88a; font-size:0.95rem;'>Active overlays: {', '.join(overlay_labels)}</div>", unsafe_allow_html=True)

    # Save Image button at the bottom
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)  # Spacer
    if st.button('💾 Save Image', key='save_image'):
        st.info('To save the map as an image, use your browser\'s screenshot or print-to-PDF feature. Direct image export is not supported in Streamlit-Folium.') 