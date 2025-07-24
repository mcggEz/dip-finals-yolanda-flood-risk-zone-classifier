import streamlit as st
import pydeck as pdk
import json


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
    # Load the GeoJSON polygon
    with open("c:/Users/mcgg/Downloads/Sketch.geojson") as f:
        geojson_data = json.load(f)

    # Extract bounds from the polygon
    coords = geojson_data['features'][0]['geometry']['coordinates'][0]
    min_lon = min([c[0] for c in coords])
    max_lon = max([c[0] for c in coords])
    min_lat = min([c[1] for c in coords])
    max_lat = max([c[1] for c in coords])
    bounds = [min_lon, min_lat, max_lon, max_lat]

    # Style from properties
    props = geojson_data['features'][0]['properties']
    fill_color = [255, 255, 255, int(float(props.get('fill-opacity', 0.4)) * 255)]
    stroke_color = [255, 204, 51, int(float(props.get('stroke-opacity', 1)) * 255)]
    stroke_width = int(props.get('stroke-width', 4))

    # BitmapLayer for your PNG
    bitmap_layer = pdk.Layer(
        "BitmapLayer",
        data=None,
        image="data/yolanda.png",
        bounds=bounds,
        opacity=0.7,
    )

    # GeoJsonLayer for your polygon
    geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_data,
        stroked=True,
        filled=True,
        get_fill_color=fill_color,
        get_line_color=stroke_color,
        get_line_width=stroke_width,
    )

    view_state = pdk.ViewState(
        longitude=(min_lon + max_lon) / 2,
        latitude=(min_lat + max_lat) / 2,
        zoom=5,
        pitch=0,
    )

    r = pdk.Deck(
        layers=[bitmap_layer, geojson_layer],
        initial_view_state=view_state,
        map_style=None,
    )

    st.pydeck_chart(r)

    overlay_labels = []
    if show_flood:
        overlay_labels.append("🌀 Flood")
    if show_shelters:
        overlay_labels.append("🏠 Shelters")
    if show_buffer:
        overlay_labels.append("🔵 Buffer")
    if overlay_labels:
        st.markdown(f"<div style='color:#1cc88a; font-size:1.2rem;'>Active overlays: {', '.join(overlay_labels)}</div>", unsafe_allow_html=True) 