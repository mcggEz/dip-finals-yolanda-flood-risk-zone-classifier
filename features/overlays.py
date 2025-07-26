import streamlit as st
from streamlit_folium import st_folium
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime
import math

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
            <b style='color:#e74c3c;'>🟥 Overlay:</b> <span style='color:#fff;'>Hazard overlays (PHIVOLCS)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_hazard = st.checkbox("Show", key="show_hazard")
    
    st.markdown(
        """
        <div class='sidebar-card' style='background:#1e3a5e;'>
            <b style='color:#3999e6;'>🔵 Marker:</b> <span style='color:#fff;'>PAGASA Warning Distribution Footprint</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_pagasa = st.checkbox("Show", key="show_pagasa")
    st.markdown(
        """
        <div class='sidebar-card' style='background:#228B22;'>
            <b style='color:#39e639;'>🟩 Marker:</b> <span style='color:#fff;'>Shelter markers (GeoAnalyticsPH)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_evac = st.checkbox("Show", key="show_evac")
    
    # PHIVOLCS Hazard Zones (Storm Surge & Rain-Induced Landslide)
    st.markdown(
        """
        <div class='sidebar-card' style='background:#8B0000;'>
            <b style='color:#fff;'>🔴 Overlay:</b> <span style='color:#fff;'>Rain-Induced Landslide Zones (PHIVOLCS)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_phivolcs_hazard = st.checkbox("Show Rain-Induced Landslide", key="show_phivolcs_hazard")
    phivolcs_hazard_opacity = st.slider("Rain-Induced Landslide Opacity", min_value=0.0, max_value=1.0, value=0.35, step=0.05, key="phivolcs_hazard_opacity")
    
    # Add the legend image here
  
    st.markdown("##### Rain-Induced Landslide Zones (PHIVOLCS)")
    
    # Custom legend UI - simplified single line
    st.markdown("""
    <div style=" padding: 8px; display: flex; align-items: center; gap: 15px; font-size: 11px;">
        <span><div style="display: inline-block; width: 15px; height: 15px; background-color: #D3D3D3; border: 1px solid #999; margin-right: 5px;"></div>No Data</span>
        <span><div style="display: inline-block; width: 15px; height: 15px; background-color: #FFFF00; border: 1px solid #999; margin-right: 5px;"></div>Low</span>
        <span><div style="display: inline-block; width: 15px; height: 15px; background-color: #800080; border: 1px solid #999; margin-right: 5px;"></div>Moderate</span>
        <span><div style="display: inline-block; width: 15px; height: 15px; background-color: #FF0000; border: 1px solid #999; margin-right: 5px;"></div>High</span>
        <span><div style="display: inline-block; width: 15px; height: 15px; background: repeating-linear-gradient(45deg, #FFFFFF, #FFFFFF 2px, #FF0000 2px, #FF0000 4px); border: 1px solid #999; margin-right: 5px;"></div>Accumulation</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(
        """
        <div class='sidebar-card' style='background:#b8860b;'>
            <b style='color:#fff;'>🟧 Overlay:</b> <span style='color:#fff;'>Hazard vs Warning Coverage (HazardHunterPH)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_hazard_vs_warning = st.checkbox("Show", key="show_hazard_vs_warning")
    hazard_vs_warning_opacity = st.slider("Overlay Opacity", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="hazard_vs_warning_opacity")
    st.markdown(
        """
        <div class='sidebar-card' style='background:#225e5e;'>
            <b style='color:#36b9cc;'>🔵 Zone:</b> <span style='color:#fff;'>Buffer zones (cyan overlay)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_buffer = st.checkbox("Show", key="show_buffer")
    return show_hazard, show_pagasa, show_evac, show_buffer, show_hazard_vs_warning, hazard_vs_warning_opacity, show_phivolcs_hazard, phivolcs_hazard_opacity

def render_overlay_main_content(show_hazard, show_pagasa, show_evac, show_buffer, show_hazard_vs_warning, hazard_vs_warning_opacity, show_phivolcs_hazard, phivolcs_hazard_opacity):
    # Center the map on the Philippines with ESRI satellite basemap
    m = folium.Map(location=[12.5, 122.5], zoom_start=6, tiles=None, min_zoom=5)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True,
        no_wrap=True  # Prevent map wrapping
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

    # PAGASA Warning Distribution Footprint from CSV
    if show_pagasa:
        try:
            pagasa_df = pd.read_csv('overlays/pagasa_warning_points.csv')
            for _, row in pagasa_df.iterrows():
                folium.CircleMarker(
                    location=[row['LAT'], row['LONG']],
                    radius=5,  # reduced from 7
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.7
                ).add_to(m)
        except Exception as e:
            st.error(f"Error loading PAGASA warning points: {e}")

    # Shelter markers (GeoAnalyticsPH) from CSV
    if show_evac:
        try:
            evac_df = pd.read_csv('overlays/evacuation_centers.csv')
            for _, row in evac_df.iterrows():
                folium.CircleMarker(
                    location=[row['LAT'], row['LONG']],
                    radius=6,  # reduced from 8
                    color='green',
                    fill=True,
                    fill_color='green',
                    fill_opacity=0.8
                ).add_to(m)
        except Exception as e:
            st.error(f"Error loading evacuation centers: {e}")

    # PHIVOLCS Hazard Zones (Storm Surge & Rain-Induced Landslide) as PNG
    if show_phivolcs_hazard:
        try:
            # Using rain_induced_landslide.png for the overlay
            # Updated bounds: top left (12.84616, 121.32461), bottom right (9.22908, 126.68594)
            phivolcs_bounds = [[9.22908, 121.32461], [12.84616, 126.68594]]  # [[south, west], [north, east]]
            folium.raster_layers.ImageOverlay(
                image='overlays/rain_induced_landslide.png',
                bounds=phivolcs_bounds,
                opacity=phivolcs_hazard_opacity,
                name='Rain-Induced Landslide Zones',
                interactive=False,
                cross_origin=False,
                zindex=8  # Place it below hazard_vs_warning_overlay if that's higher
            ).add_to(m)
        except Exception as e:
            st.error(f"Error loading Rain-Induced Landslide PNG: {e}")

    # Overlay Hazard Zones vs Warning Coverage (HazardHunterPH) as PNG
    if show_hazard_vs_warning:
        try:
            folium.raster_layers.ImageOverlay(
                image='overlays/hazard_vs_warning_overlay.png',
                bounds=[[9.34962, 119.38184], [13.01001, 126.2428]],  # Updated bounds: [[south, west], [north, east]]
                opacity=hazard_vs_warning_opacity,
                name='Hazard vs Warning Overlay',
                interactive=False,
                cross_origin=False,
                zindex=10
            ).add_to(m)
        except Exception as e:
            st.error(f"Error loading hazard vs warning overlay PNG: {e}")

    # Buffer zones (improved: buffer around each shelter marker)
    if show_buffer:
        try:
            evac_df = pd.read_csv('overlays/evacuation_centers.csv')
            buffer_radius = 0.05  # degrees, ~5.5km (decreased from 0.1)
            for _, row in evac_df.iterrows():
                folium.Circle(
                    location=[row['LAT'], row['LONG']],
                    radius=buffer_radius * 50000,  # convert degrees to meters
                    color='cyan',
                    fill=True,
                    fill_color='cyan',
                    fill_opacity=0.2,
                    weight=2
                ).add_to(m)
        except Exception as e:
            st.error(f"Error generating buffer zones: {e}")

    # Add interactive patch selection functionality only if enabled
    if st.session_state.get('patch_selection_enabled', False):
        add_patch_selection_to_map(m)
    
    # Render the map with click handling
    map_data = st_folium(
        m, 
        width=None, 
        height=600, 
        key="main_map",
        returned_objects=["last_clicked"]
    )
    
    # Store overlay states in session state for patch image generation
    st.session_state['active_hazard'] = show_hazard
    st.session_state['active_pagasa'] = show_pagasa
    st.session_state['active_evac'] = show_evac
    st.session_state['active_buffer'] = show_buffer
    st.session_state['active_hazard_vs_warning'] = show_hazard_vs_warning
    st.session_state['active_phivolcs_hazard'] = show_phivolcs_hazard
    
    # Overlay labels - render immediately after map
    overlay_labels = []
    if show_hazard:
        overlay_labels.append("🟥 Hazard Overlays")
    if show_pagasa:
        overlay_labels.append("🔵 PAGASA Warnings")
    if show_evac:
        overlay_labels.append("🟩 Shelter Markers")
    if show_buffer:
        overlay_labels.append("🔵 Buffer Zones")
    if show_hazard_vs_warning:
        overlay_labels.append("🟧 Hazard vs Warning Coverage")
    if show_phivolcs_hazard:
        overlay_labels.append("🔴 Rain-Induced Landslide Zones")
    if overlay_labels:
        st.markdown(f"<div style='color:#1cc88a; font-size:0.95rem;'>Active overlays: {', '.join(overlay_labels)}</div>", unsafe_allow_html=True)
    
    # Handle patch selection only if enabled
    if st.session_state.get('patch_selection_enabled', False) and map_data and map_data.get('last_clicked'):
        # Store the clicked location immediately
        clicked_location = map_data['last_clicked']
        
        # Calculate patch bounds (224x224 pixels equivalent)
        patch_size_degrees = 0.01  # Approximately 224x224 pixels in degrees
        center_lat = clicked_location['lat']
        center_lng = clicked_location['lng']
        
        # Calculate patch bounds
        patch_bounds = {
            'north': center_lat + patch_size_degrees/2,
            'south': center_lat - patch_size_degrees/2,
            'east': center_lng + patch_size_degrees/2,
            'west': center_lng - patch_size_degrees/2
        }
        
        # Store the selected patch in session state immediately
        st.session_state['selected_patch'] = {
            'center_lat': center_lat,
            'center_lng': center_lng,
            'bounds': patch_bounds,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Mark that we need to show the patch on next render
        st.session_state['show_patch_on_map'] = True
        
        # Show immediate feedback
        st.success(f"🎯 Patch selected at {center_lat:.4f}°, {center_lng:.4f}° - Updating map...")
        
        # Force a rerun to update the map with the new patch rectangle
        st.rerun()
    
    # Show debug info below map if user patch exists
    if 'selected_patch' in st.session_state:
        selected = st.session_state['selected_patch']
        
        # Add padding container
      
        
        st.info(f"🎯 User patch on map: Center {selected['center_lat']:.6f}°, {selected['center_lng']:.6f}°")
        
        col1, col2 = st.columns(2)
        
        with col1:
         
            st.subheader("📍 Patch Coordinates")
            st.write(f"**Center:** {selected['center_lat']:.6f}°, {selected['center_lng']:.6f}°")
            st.write(f"**North:** {selected['bounds']['north']:.6f}°")
            st.write(f"**South:** {selected['bounds']['south']:.6f}°")
            st.write(f"**East:** {selected['bounds']['east']:.6f}°")
            st.write(f"**West:** {selected['bounds']['west']:.6f}°")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            
            st.subheader("📐 Patch Properties")
            st.write(f"**Size:** 224x224 pixels equivalent")
            st.write(f"**Area:** ~0.0001 square degrees")
            st.write(f"**Timestamp:** {selected['timestamp']}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        display_patch_image(selected['center_lat'], selected['center_lng'], selected['bounds'])
        
        # Generate metadata for the selected patch
        generate_patch_metadata(selected['center_lat'], selected['center_lng'], selected['bounds'])

def add_patch_selection_to_map(m):
    """Add interactive patch selection functionality to the map"""
    

    
    # Add user-selected patch if exists
    if 'selected_patch' in st.session_state:
        selected = st.session_state['selected_patch']
        user_bounds = [
            [selected['bounds']['south'], selected['bounds']['west']],
            [selected['bounds']['north'], selected['bounds']['east']]
        ]
        
        # Create a transparent rectangle with just border for user selection
        folium.Rectangle(
            bounds=user_bounds,
            color='blue',
            weight=3,
            fillColor='blue',
            fillOpacity=0.0,
            dashArray='10, 10'
        ).add_to(m)
        
        # Also add a center marker for better visibility
        folium.Marker(
            location=[selected['center_lat'], selected['center_lng']],
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)



def display_patch_image(center_lat, center_lng, patch_bounds):
    """Display the patch image extracted from the map overlays using server-side rendering"""
    
    st.markdown("### 🖼️ Patch Image")
    
    # Show what overlays are active
    active_overlays = get_active_overlays()
    if active_overlays:
        st.info(f"🎯 **Attempting to capture overlays:** {', '.join(active_overlays)}")
        st.info("📋 **Overlay details:**")
        for overlay in active_overlays:
            if 'Hazard' in overlay:
                st.write(f"  🔴 {overlay}: Red polygons with 60% fill opacity")
            elif 'PAGASA' in overlay:
                st.write(f"  🔵 {overlay}: Blue circles with 90% fill opacity")
            elif 'Shelter' in overlay:
                st.write(f"  🟢 {overlay}: Green circles with 90% fill opacity")
            elif 'Buffer' in overlay:
                st.write(f"  🔵 {overlay}: Cyan circles with 30% fill opacity")
            elif 'Hazard vs Warning' in overlay:
                st.write(f"  🟧 {overlay}: Image overlay with opacity control")
            elif 'Rain-Induced' in overlay:
                st.write(f"  🔴 {overlay}: Image overlay with opacity control")
    else:
        st.warning("⚠️ **No overlays active** - Only base satellite imagery will be captured")
        st.info("💡 **Tip:** Enable overlays in the sidebar to capture them in the PNG.")
    
    try:
        # Try server-side rendering first (with Selenium)
        patch_img = capture_map_with_overlays(center_lat, center_lng)
        
        if patch_img:
            # Display the captured patch image with overlays
            caption = f"224x224 Pixel Patch at {center_lat:.4f}°, {center_lng:.4f}°"
            active_overlays = get_active_overlays()
            if active_overlays:
                caption += f" (with {', '.join(active_overlays)} overlays)"
            
            st.image(patch_img, caption=caption, use_container_width=False)
            
            # Add download button for the patch
            import io
            img_buffer = io.BytesIO()
            patch_img.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Patch PNG",
                data=img_data,
                file_name=f"patch_{center_lat:.4f}_{center_lng:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
            
            # Show overlay information
            if active_overlays:
                st.success(f"✅ **Real overlays captured:** {', '.join(active_overlays)}")
                st.info("💡 **Tip:** If overlays don't appear, try enabling them in the sidebar and clicking the map again.")
            else:
                st.info("📷 **Base satellite imagery** - no overlays active")
                st.info("💡 **Tip:** Enable overlays in the sidebar to capture them in the PNG.")
        else:
            # Fallback to tile-based method
            fallback_to_tile_method(center_lat, center_lng)
            
    except Exception as e:
        st.error(f"Error with server-side rendering: {e}")
        st.info("🔄 Falling back to tile-based method...")
        fallback_to_tile_method(center_lat, center_lng)

def capture_map_with_overlays(center_lat, center_lng, zoom_level=15):
    """Capture map with overlays using Selenium headless browser"""
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        from PIL import Image
        import io
        import os
        
        # Create headless Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=800,600")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        
        # Setup Chrome driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        try:
            # Instead of creating a separate map, let's try to capture the actual main map
            # We'll create a simplified version that matches the main map exactly
            html_content = create_main_map_html(center_lat, center_lng, zoom_level)
            
            # Save to temp file
            temp_file = "temp_map.html"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            # Load the map in browser
            driver.get(f"file:///{os.path.abspath(temp_file)}")
            driver.implicitly_wait(15)  # Wait longer for map to load
            
            # Additional wait for tiles and overlays to load
            import time
            time.sleep(5)  # Increased wait time
            
            # Additional wait after overlay detection
            time.sleep(3)
            
            # Force overlay visibility with JavaScript
            try:
                driver.execute_script("""
                    // Make all overlays more visible
                    document.querySelectorAll('path[stroke]').forEach(function(path) {
                        path.style.strokeWidth = '3px';
                        path.style.strokeOpacity = '0.8';
                    });
                    document.querySelectorAll('circle[stroke]').forEach(function(circle) {
                        circle.style.strokeWidth = '3px';
                        circle.style.strokeOpacity = '0.8';
                    });
                """)
                time.sleep(1)
            except Exception as e:
                st.warning(f"⚠️ Could not enhance overlay visibility: {str(e)}")
            
            # Take screenshot
            screenshot = driver.get_screenshot_as_png()
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(screenshot))
            
            # Crop to 224x224 patch area (center of screenshot)
            width, height = img.size
            center_x, center_y = width // 2, height // 2
            patch_size = 224
            
            # Calculate crop bounds
            left = max(0, center_x - patch_size // 2)
            top = max(0, center_y - patch_size // 2)
            right = min(width, left + patch_size)
            bottom = min(height, top + patch_size)
            
            # Crop the patch
            patch_img = img.crop((left, top, right, bottom))
            
            # Resize to exactly 224x224 if needed
            if patch_img.size != (224, 224):
                patch_img = patch_img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return patch_img
            
        finally:
            driver.quit()
            
    except Exception as e:
        st.error(f"Selenium error: {e}")
        return None

def create_main_map_html(center_lat, center_lng, zoom_level):
    """Create HTML with Folium map and active overlays"""
    
    # Create a new Folium map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom_level,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Satellite"
    )
    
    # Add ALL overlays that are active in the main map
    # This ensures we capture exactly what's visible on the main map
    
    # 1. Hazard overlays (PHIVOLCS) - EXACT same as main map
    if st.session_state.get('active_hazard', False):
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
                }
            ).add_to(m)
        except Exception as e:
            st.error(f"Error loading hazard shapefile: {e}")
    
    # 2. PAGASA Warning Points - EXACT same as main map
    if st.session_state.get('active_pagasa', False):
        try:
            pagasa_df = pd.read_csv('overlays/pagasa_warning_points.csv')
            for _, row in pagasa_df.iterrows():
                folium.CircleMarker(
                    location=[row['LAT'], row['LONG']],
                    radius=5,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.7
                ).add_to(m)
        except Exception as e:
            st.error(f"Error loading PAGASA warning points: {e}")
    
    # 3. Evacuation Centers - EXACT same as main map
    if st.session_state.get('active_evac', False):
        try:
            evac_df = pd.read_csv('overlays/evacuation_centers.csv')
            for _, row in evac_df.iterrows():
                folium.CircleMarker(
                    location=[row['LAT'], row['LONG']],
                    radius=6,
                    color='green',
                    fill=True,
                    fill_color='green',
                    fill_opacity=0.8
                ).add_to(m)
        except Exception as e:
            st.error(f"Error loading evacuation centers: {e}")
    
    # 4. Buffer Zones
    if st.session_state.get('active_buffer', False):
        try:
            # Create buffer zones around evacuation centers
            evac_df = pd.read_csv('overlays/evacuation_centers.csv')
            for _, row in evac_df.iterrows():
                folium.Circle(
                    location=[row['LAT'], row['LONG']],
                    radius=5000,  # 5km buffer
                    color='cyan',
                    fill=True,
                    fill_color='cyan',
                    fill_opacity=0.3,
                    weight=2
                ).add_to(m)
            st.info("🔵 Buffer zones added to capture map")
        except Exception as e:
            st.error(f"Error loading buffer zones: {e}")
    
    # 5. Hazard vs Warning Coverage - EXACT same as main map
    if st.session_state.get('active_hazard_vs_warning', False):
        try:
            folium.raster_layers.ImageOverlay(
                image='overlays/hazard_vs_warning_overlay.png',
                bounds=[[9.34962, 119.38184], [13.01001, 126.2428]],  # Updated bounds: [[south, west], [north, east]]
                opacity=st.session_state.get('hazard_vs_warning_opacity', 0.5),
                name='Hazard vs Warning Overlay',
                interactive=False,
                cross_origin=False,
                zindex=10
            ).add_to(m)
        except Exception as e:
            st.error(f"Error loading hazard vs warning overlay PNG: {e}")
    
    # 6. PHIVOLCS Rain-Induced Landslide - EXACT same as main map
    if st.session_state.get('active_phivolcs_hazard', False):
        try:
            # Using rain_induced_landslide.png for the overlay
            # Updated bounds: top left (12.84616, 121.32461), bottom right (9.22908, 126.68594)
            phivolcs_bounds = [[9.22908, 121.32461], [12.84616, 126.68594]]  # [[south, west], [north, east]]
            folium.raster_layers.ImageOverlay(
                image='overlays/rain_induced_landslide.png',
                bounds=phivolcs_bounds,
                opacity=st.session_state.get('phivolcs_hazard_opacity', 0.35),
                name='Rain-Induced Landslide Zones',
                interactive=False,
                cross_origin=False,
                zindex=8  # Place it below hazard_vs_warning_overlay if that's higher
            ).add_to(m)
        except Exception as e:
            st.error(f"Error loading Rain-Induced Landslide PNG: {e}")
    
    # Add patch selection rectangle
    patch_size_degrees = 0.01
    patch_bounds = [
        [center_lat - patch_size_degrees/2, center_lng - patch_size_degrees/2],
        [center_lat + patch_size_degrees/2, center_lng + patch_size_degrees/2]
    ]
    
    folium.Rectangle(
        bounds=patch_bounds,
        color='blue',
        weight=3,
        fillColor='blue',
        fillOpacity=0.0,
        dashArray='10, 10'
    ).add_to(m)
    
    # Create complete HTML with proper styling
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Map Capture</title>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #000;
            }}
            #map {{
                width: 800px;
                height: 600px;
                margin: 0 auto;
            }}
            /* Ensure overlays are visible */
            .leaflet-overlay-pane svg {{
                z-index: 1000 !important;
            }}
            .leaflet-marker-pane {{
                z-index: 1001 !important;
            }}
            /* Make overlays more prominent */
            path[stroke] {{
                stroke-width: 3px !important;
                stroke-opacity: 0.8 !important;
            }}
            circle[stroke] {{
                stroke-width: 3px !important;
                stroke-opacity: 0.8 !important;
            }}
            /* Ensure fill colors are visible */
            path[fill] {{
                fill-opacity: 0.6 !important;
            }}
            circle[fill] {{
                fill-opacity: 0.8 !important;
            }}
        </style>
    </head>
    <body>
        <div id="map">
            {m._repr_html_()}
        </div>

    </body>
    </html>
    """
    
    return html_content

def get_active_overlays():
    """Get list of active overlays"""
    active_overlays = []
    if st.session_state.get('active_hazard', False):
        active_overlays.append("Hazard")
    if st.session_state.get('active_pagasa', False):
        active_overlays.append("PAGASA")
    if st.session_state.get('active_evac', False):
        active_overlays.append("Shelter")
    if st.session_state.get('active_phivolcs_hazard', False):
        active_overlays.append("PHIVOLCS")
    if st.session_state.get('active_hazard_vs_warning', False):
        active_overlays.append("Hazard vs Warning")
    return active_overlays

def fallback_to_tile_method(center_lat, center_lng):
    """Improved tile-based method that captures multiple tiles for accurate 224x224 patches"""
    
    # Calculate zoom level and tile coordinates for ESRI satellite tiles
    zoom_level = 15  # High resolution zoom level
    tile_size = 256  # Standard tile size
    
    # Convert lat/lng to tile coordinates
    lat_rad = math.radians(center_lat)
    n = 2.0 ** zoom_level
    xtile = int((center_lng + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    
    # Calculate pixel offset within the tile
    x_pixel = int((center_lng + 180.0) / 360.0 * n * tile_size) % tile_size
    y_pixel = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * tile_size) % tile_size
    
    try:
        import requests
        from PIL import Image, ImageDraw
        import io
        
        # Calculate which tiles we need to download
        # We need enough tiles to cover a 224x224 pixel area
        tiles_needed_x = 2  # We'll need 2 tiles horizontally
        tiles_needed_y = 2  # We'll need 2 tiles vertically
        
        # Create a larger canvas to stitch tiles together
        canvas_width = tile_size * tiles_needed_x
        canvas_height = tile_size * tiles_needed_y
        canvas = Image.new('RGB', (canvas_width, canvas_height))
        
        # Download and stitch tiles
        for i in range(tiles_needed_x):
            for j in range(tiles_needed_y):
                # Calculate tile coordinates
                current_xtile = xtile + i - 1  # -1 to get tiles to the left
                current_ytile = ytile + j - 1  # -1 to get tiles above
                
                # Construct ESRI satellite tile URL
                tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom_level}/{current_ytile}/{current_xtile}"
                
                try:
                    response = requests.get(tile_url, timeout=10)
                    if response.status_code == 200:
                        # Open the tile image
                        tile_img = Image.open(io.BytesIO(response.content))
                        
                        # Paste tile onto canvas
                        paste_x = i * tile_size
                        paste_y = j * tile_size
                        canvas.paste(tile_img, (paste_x, paste_y))
                    else:
                        st.warning(f"Failed to download tile {current_xtile},{current_ytile}")
                        
                except Exception as e:
                    st.warning(f"Error downloading tile {current_xtile},{current_ytile}: {e}")
        
        # Calculate the center pixel in the canvas
        center_x = tile_size + x_pixel  # Center tile + offset
        center_y = tile_size + y_pixel  # Center tile + offset
        
        # Crop the 224x224 pixel area from the center
        patch_size = 112  # Half of 224 pixels
        start_x = max(0, center_x - patch_size)
        end_x = min(canvas_width, center_x + patch_size)
        start_y = max(0, center_y - patch_size)
        end_y = min(canvas_height, center_y + patch_size)
        
        # Crop the patch
        patch_img = canvas.crop((start_x, start_y, end_x, end_y))
        
        # Resize to exactly 224x224 if needed
        if patch_img.size != (224, 224):
            patch_img = patch_img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Display the patch image
        caption = f"224x224 Pixel Patch at {center_lat:.4f}°, {center_lng:.4f}° (Base satellite only)"
        st.image(patch_img, caption=caption, use_container_width=False)
        
        # Add download button for the patch
        img_buffer = io.BytesIO()
        patch_img.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Patch PNG",
            data=img_data,
            file_name=f"patch_{center_lat:.4f}_{center_lng:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png"
        )
        
        st.warning("⚠️ **Fallback mode:** Base satellite imagery only - overlays not included")
        st.info("💡 **Tip:** This method captures a larger area and crops to 224x224 for better accuracy")
        
    except Exception as e:
        st.error(f"Error in improved fallback method: {e}")
        # Final fallback to placeholder
        st.markdown("""
        <div style="padding: 15px; text-align: center;">
            <div style="width: 224px; height: 224px; background: linear-gradient(45deg, #2a2a2a 25%, transparent 25%), linear-gradient(-45deg, #2a2a2a 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #2a2a2a 75%), linear-gradient(-45deg, transparent 75%, #2a2a2a 75%); background-size: 20px 20px; background-position: 0 0, 0 10px, 10px -10px, -10px 0px; border: 2px dashed #666; display: flex; align-items: center; justify-content: center; margin: 0 auto;">
                <div style="text-align: center; color: #ccc;">
                    <div style="font-size: 24px;">🛰️</div>
                    <div style="font-size: 12px; margin-top: 5px;">224x224 Pixel Patch</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def generate_patch_metadata(center_lat, center_lng, patch_bounds):
    """Generate metadata for the selected patch"""
    
    # Mock hazard analysis based on location
    hazard_score = np.random.uniform(0.1, 0.9)
    elevation = np.random.uniform(10, 500)
    shelter_proximity = np.random.uniform(0.5, 10.0)
    
    # Determine risk class based on location and hazard score
    if center_lat < 11.5:  # Southern Philippines
        if hazard_score > 0.7:
            risk_class = "HighRisk_Coastal"
        elif hazard_score > 0.4:
            risk_class = "ModerateRisk_Upland"
        else:
            risk_class = "SafeZone_UrbanCore"
    else:  # Northern Philippines
        if hazard_score > 0.6:
            risk_class = "WarningGap_Barangay"
        else:
            risk_class = "BufferZone_Proposed"
    
    # Display metadata
    st.subheader("📊 Patch Metadata")
    st.markdown(
        f"""
        <div class='sidebar-card' style='background:#3a3a3a; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <b style='color:#1cc88a;'>Hazard Score:</b> <span style='color:#fff;'>{hazard_score:.2f}</span><br>
            <b style='color:#f6c23e;'>Elevation:</b> <span style='color:#fff;'>{elevation:.0f} m</span><br>
            <b style='color:#36b9cc;'>Coordinates:</b> <span style='color:#fff;'>{center_lat:.4f}°, {center_lng:.4f}°</span><br>
            <b style='color:#e74a3b;'>Shelter Proximity:</b> <span style='color:#fff;'>{shelter_proximity:.1f} km</span><br>
            <b style='color:#858796;'>Timestamp:</b> <span style='color:#fff;'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span><br>
            <b style='color:#6f42c1;'>Risk Class:</b> <span style='color:#fff;'>{risk_class}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Export functionality
    st.subheader("💾 Export Patch Data")
    
    # Create metadata dataframe
    metadata_data = {
        'Center_Latitude': [center_lat],
        'Center_Longitude': [center_lng],
        'North_Bound': [patch_bounds['north']],
        'South_Bound': [patch_bounds['south']],
        'East_Bound': [patch_bounds['east']],
        'West_Bound': [patch_bounds['west']],
        'Risk_Class': [risk_class],
        'Hazard_Score': [hazard_score],
        'Elevation_m': [elevation],
        'Shelter_Proximity_km': [shelter_proximity],
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }
    
    df = pd.DataFrame(metadata_data)
    
    # Generate CSV
    csv = df.to_csv(index=False)
    
    # Download button
    st.download_button(
        label="📥 Download Patch CSV",
        data=csv,
        file_name=f"patch_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download the patch selection data as a CSV file"
    ) 