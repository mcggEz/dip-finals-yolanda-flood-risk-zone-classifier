import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
import numpy as np

def show_patch_selector():
    """Main patch selector UI component"""
    
    # Initialize session state for patch selection
    if 'patch_selection_enabled' not in st.session_state:
        st.session_state['patch_selection_enabled'] = False
   
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload a patch image (224x224 pixels recommended)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image file to classify"
    )
    
    st.markdown("---")
    
    # Map selection section
    map_selection_enabled = st.checkbox(
        "Enable interactive map patch selection",
        value=st.session_state.get('patch_selection_enabled', False),
    )
    
    # Update session state based on checkbox
    st.session_state['patch_selection_enabled'] = map_selection_enabled
    
    if map_selection_enabled:
        st.success("✅ **Patch selection is now active on the main map!**")
    
    selected_patch = None
    
    # Display file info and preview
    if uploaded_file is not None:
        st.write(f"**File size:** {uploaded_file.size} bytes")
        
        # Show preview
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Patch", use_container_width=True)
        
            # Classification button
        if st.button("🔘 Classify Patch", type="primary"):
            display_metadata_and_export(uploaded_file.name, "uploaded_file")

def display_metadata_and_export(source_name, source_type, patch_data=None):
    """Display metadata and export functionality"""
    
    # Use patch data if available, otherwise use placeholder data
    if patch_data:
        hazard_score = patch_data['hazard_score']
        elevation = patch_data['elevation']
        latitude = patch_data['coords'][0]
        longitude = patch_data['coords'][1]
        shelter_proximity = patch_data['shelter_proximity']
        predicted_class = patch_data['class']
    else:
        # Try to use CNN prediction if available
        try:
            from predict_patch import predict_image, load_trained_model
            model = load_trained_model()
            if model and os.path.exists('temp_patch.png'):
                result = predict_image(model, 'temp_patch.png')
                if result:
                    hazard_score = result['confidence']
                    predicted_class = result['class']
                else:
                    hazard_score = 0.75
                    predicted_class = "Unknown"
            else:
                hazard_score = 0.75
                predicted_class = "Unknown"
        except:
            hazard_score = 0.75
            predicted_class = "Unknown"
        
        elevation = 45
        latitude = 11.2345
        longitude = 124.5678
        shelter_proximity = 2.3
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display metadata table
    st.markdown(
        f"""
        <div class='sidebar-card' style='background:#3a3a3a; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <b style='color:#1cc88a;'>Hazard Score:</b> <span style='color:#fff;'>{hazard_score:.2f}</span><br>
            <b style='color:#f6c23e;'>Elevation:</b> <span style='color:#fff;'>{elevation} m</span><br>
            <b style='color:#36b9cc;'>Coordinates:</b> <span style='color:#fff;'>{latitude:.4f}°, {longitude:.4f}°</span><br>
            <b style='color:#e74a3b;'>Shelter Proximity:</b> <span style='color:#fff;'>{shelter_proximity} km</span><br>
            <b style='color:#858796;'>Timestamp:</b> <span style='color:#fff;'>{timestamp}</span><br>
            <b style='color:#6f42c1;'>Predicted Class:</b> <span style='color:#fff;'>{predicted_class}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create metadata dataframe
    metadata_data = {
        'Source_Name': [source_name],
        'Source_Type': [source_type],
        'Predicted_Class': [predicted_class],
        'Hazard_Score': [hazard_score],
        'Elevation_m': [elevation],
        'Latitude': [latitude],
        'Longitude': [longitude],
        'Shelter_Proximity_km': [shelter_proximity],
        'Timestamp': [timestamp]
    }
    
    df = pd.DataFrame(metadata_data)
    
    # Generate CSV
    csv = df.to_csv(index=False)
    
    # Download button
    st.download_button(
                label="📥 Download CSV",
                data=csv,
        file_name=f"patch_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download the metadata as a CSV file"
    )

def create_patch_map():
    """Create an interactive map with patch locations"""
    
    # Center map on Philippines (Tacloban area)
    m = folium.Map(
        location=[11.2444, 125.0098],  # Tacloban coordinates
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Define patch locations with their coordinates and metadata
    patch_locations = [
        {
            'name': 'High Risk Coastal - Tacloban',
            'coords': [11.2444, 125.0098],
            'class': 'HighRisk_Coastal',
            'hazard_score': 0.85,
            'elevation': 5,
            'shelter_proximity': 0.8
        },
        {
            'name': 'Moderate Risk Upland - Ormoc',
            'coords': [11.0047, 124.6075],
            'class': 'ModerateRisk_Upland',
            'hazard_score': 0.65,
            'elevation': 150,
            'shelter_proximity': 2.1
        },
        {
            'name': 'Safe Zone Urban - Cebu City',
            'coords': [10.3157, 123.8854],
            'class': 'SafeZone_UrbanCore',
            'hazard_score': 0.25,
            'elevation': 200,
            'shelter_proximity': 0.3
        },
        {
            'name': 'Evacuation Center - Palo',
            'coords': [11.1577, 124.9908],
            'class': 'EvacCenter_Active',
            'hazard_score': 0.15,
            'elevation': 25,
            'shelter_proximity': 0.1
        },
        {
            'name': 'Warning Gap - Tanauan',
            'coords': [11.1111, 125.0167],
            'class': 'WarningGap_Barangay',
            'hazard_score': 0.75,
            'elevation': 35,
            'shelter_proximity': 5.2
        },
        {
            'name': 'Buffer Zone - Baybay',
            'coords': [10.6785, 124.8016],
            'class': 'BufferZone_Proposed',
            'hazard_score': 0.45,
            'elevation': 80,
            'shelter_proximity': 1.5
        }
    ]
    
    # Add patch markers to map
    for patch in patch_locations:
        # Create red rectangular marker
        folium.Rectangle(
            bounds=[
                [patch['coords'][0] - 0.01, patch['coords'][1] - 0.01],
                [patch['coords'][0] + 0.01, patch['coords'][1] + 0.01]
            ],
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6,
            weight=2,
            popup=f"<b>{patch['name']}</b><br>Click to select this patch"
        ).add_to(m)
        
        # Add clickable marker
        folium.Marker(
            location=patch['coords'],
            popup=f"<b>{patch['name']}</b><br>Class: {patch['class']}<br>Hazard: {patch['hazard_score']:.2f}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    return m

def get_patch_from_location(clicked_location):
    """Get patch information based on clicked location"""
    
    # Define patch locations with their coordinates and metadata
    patch_locations = [
        {
            'name': 'High Risk Coastal - Tacloban',
            'coords': [11.2444, 125.0098],
            'class': 'HighRisk_Coastal',
            'hazard_score': 0.85,
            'elevation': 5,
            'shelter_proximity': 0.8
        },
        {
            'name': 'Moderate Risk Upland - Ormoc',
            'coords': [11.0047, 124.6075],
            'class': 'ModerateRisk_Upland',
            'hazard_score': 0.65,
            'elevation': 150,
            'shelter_proximity': 2.1
        },
        {
            'name': 'Safe Zone Urban - Cebu City',
            'coords': [10.3157, 123.8854],
            'class': 'SafeZone_UrbanCore',
            'hazard_score': 0.25,
            'elevation': 200,
            'shelter_proximity': 0.3
        },
        {
            'name': 'Evacuation Center - Palo',
            'coords': [11.1577, 124.9908],
            'class': 'EvacCenter_Active',
            'hazard_score': 0.15,
            'elevation': 25,
            'shelter_proximity': 0.1
        },
        {
            'name': 'Warning Gap - Tanauan',
            'coords': [11.1111, 125.0167],
            'class': 'WarningGap_Barangay',
            'hazard_score': 0.75,
            'elevation': 35,
            'shelter_proximity': 5.2
        },
        {
            'name': 'Buffer Zone - Baybay',
            'coords': [10.6785, 124.8016],
            'class': 'BufferZone_Proposed',
            'hazard_score': 0.45,
            'elevation': 80,
            'shelter_proximity': 1.5
        }
    ]
    
    # Find the closest patch to the clicked location
    clicked_lat, clicked_lng = clicked_location['lat'], clicked_location['lng']
    min_distance = float('inf')
    closest_patch = None
    
    for patch in patch_locations:
        patch_lat, patch_lng = patch['coords']
        distance = np.sqrt((clicked_lat - patch_lat)**2 + (clicked_lng - patch_lng)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_patch = patch
    
    return closest_patch 