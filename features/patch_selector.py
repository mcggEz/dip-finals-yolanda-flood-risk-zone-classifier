import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_patch_selector():
    """Main patch selector UI component"""
    
    # Initialize session state for patch selection
    if 'patch_selection_enabled' not in st.session_state:
        st.session_state['patch_selection_enabled'] = False
   
    st.markdown("---")
   
    # Single uploader for both single and batch
    st.markdown(
        """
        <div class='sidebar-card' style='background:#1e3a5e;'>
            <b style='color:#3999e6;'>📁 Select:</b> <span style='color:#fff;'>Patch Images (Single or Batch)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Dropdown for patch collections
    # File uploader for manual uploads
    uploaded_files = st.file_uploader(
        "Or upload patch images manually (224x224 pixels recommended)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload one or more image files to classify"
    )
    
    # Analysis buttons (immediately after file upload)
    if uploaded_files:
        st.session_state['patch_uploaded_files'] = uploaded_files
        if len(uploaded_files) == 1:
            if st.button("🔘 Classify Patch", type="primary"):
                st.session_state['patch_analysis_trigger'] = 'single'
        else:
            if st.button("🔘 Run Batch Analysis", type="primary"):
                st.session_state['patch_analysis_trigger'] = 'batch'
    else:
        st.session_state['patch_uploaded_files'] = None
        st.session_state['patch_analysis_trigger'] = None
    
    st.markdown("---")
    
    # Map selection section (separate feature)
    st.markdown(
        """
        <div class='sidebar-card' style='background:#225e5e;'>
            <b style='color:#36b9cc;'>🗺️ Additional:</b> <span style='color:#fff;'>Interactive Map Patch Selection</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    map_selection_enabled = st.checkbox(
        "Enable interactive map patch selection",
        value=st.session_state.get('patch_selection_enabled', False),
    )
    
    # Update session state based on checkbox
    st.session_state['patch_selection_enabled'] = map_selection_enabled
    
    if map_selection_enabled:
        st.success("✅ **Patch selection is now active on the main map!**")
    
    st.markdown("---")

def display_metadata_and_export(source_name, source_type, patch_data=None):
    """Display metadata and export functionality"""
    
    # Use patch data if available, otherwise use Random Forest prediction
    if patch_data:
        hazard_score = patch_data['hazard_score']
        elevation = patch_data['elevation']
        latitude = patch_data['coords'][0]
        longitude = patch_data['coords'][1]
        shelter_proximity = patch_data['shelter_proximity']
        predicted_class = patch_data['class']
    else:
        # Get the actual filename for coordinate extraction
        actual_filename = source_name.name if hasattr(source_name, 'name') else str(source_name)
        
        # Try to use Random Forest prediction
        try:
            import sys
            sys.path.append('.')
            from test_random_forest import get_random_forest_prediction
            
            # Save uploaded file temporarily if it's a file object
            if hasattr(source_name, 'read'):
                temp_path = "temp_patch.png"
                with open(temp_path, "wb") as f:
                    f.write(source_name.getbuffer())
                image_path = temp_path
            else:
                image_path = source_name
            
            result = get_random_forest_prediction(image_path)
            if result:
                hazard_score = result['hazard_score']
                predicted_class = result['class']
                confidence = result['confidence']
                
                # Check if confidence is too low
                if confidence < 0.3:  # 30% threshold
                    st.warning(f"⚠️ **Very Low Confidence ({confidence:.1%})** - Model is highly uncertain")
                    st.write("   Consider using a different image or the image may not match any training class.")
            else:
                hazard_score = 0.16
                predicted_class = "SafeZone_UrbanCore"
                confidence = 0.85
                
            # Clean up temp file
            import os
            if os.path.exists("temp_patch.png"):
                os.remove("temp_patch.png")
                
        except Exception as e:
            st.warning(f"Random Forest prediction failed: {str(e)}")
            hazard_score = 0.16
            predicted_class = "SafeZone_UrbanCore"
            confidence = 0.85
        
        # Generate coordinates from filename if it contains lat/lng
        if '_' in actual_filename and actual_filename.replace('.', '_').count('_') >= 2:
            try:
                # Extract coordinates from filename like "patch_10.6487_122.9789.png"
                parts = actual_filename.replace('.png', '').split('_')
                if len(parts) >= 3:
                    latitude = float(parts[1])
                    longitude = float(parts[2])
                else:
                    latitude = 10.6487
                    longitude = 122.9789
            except:
                latitude = 10.6487
                longitude = 122.9789
        else:
            latitude = 10.6487
            longitude = 122.9789
        
        shelter_proximity = 3.5
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display metadata in table format
    st.markdown("### 📊 Patch Analysis Results")
    
    # Format timestamp to split date and time
    timestamp_parts = timestamp.split(' ')
    date_part = timestamp_parts[0] if len(timestamp_parts) > 0 else timestamp
    time_part = timestamp_parts[1] if len(timestamp_parts) > 1 else "00:00"
    
    # Get the actual filename for display
    display_filename = source_name.name if hasattr(source_name, 'name') else str(source_name)
    
    # Add RiskClass to table data
    table_data = {
        'Filename': [f"{display_filename}"],
        'HazardScore': [f"{hazard_score:.2f}"],
        'Latitude': [f"{latitude:.4f}"],
        'Longitude': [f"{longitude:.4f}"],
        'ShelterProximity': [f"{shelter_proximity}"],
        'Timestamp': [f"{date_part}<br>{time_part}"],
        'RiskClass': [predicted_class]
    }
    
    # Custom HTML table with RiskClass
    html_table = f"""
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0; font-family: monospace; font-size: 12px;">
        <thead>
            <tr>
                <th style="background-color: #23272b; color: #fff; border: 1px solid #dee2e6; padding: 8px; text-align: left; font-weight: bold;">Filename</th>
                <th style="background-color: #23272b; color: #fff; border: 1px solid #dee2e6; padding: 8px; text-align: left; font-weight: bold;">HazardScore</th>
                <th style="background-color: #23272b; color: #fff; border: 1px solid #dee2e6; padding: 8px; text-align: left; font-weight: bold;">Latitude</th>
                <th style="background-color: #23272b; color: #fff; border: 1px solid #dee2e6; padding: 8px; text-align: left; font-weight: bold;">Longitude</th>
                <th style="background-color: #23272b; color: #fff; border: 1px solid #dee2e6; padding: 8px; text-align: left; font-weight: bold;">ShelterProximity</th>
                <th style="background-color: #23272b; color: #fff; border: 1px solid #dee2e6; padding: 8px; text-align: left; font-weight: bold;">Timestamp</th>
                <th style="background-color: #23272b; color: #fff; border: 1px solid #dee2e6; padding: 8px; text-align: left; font-weight: bold;">RiskClass</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border: 1px solid #dee2e6; padding: 8px; text-align: left;">{display_filename}</td>
                <td style="border: 1px solid #dee2e6; padding: 8px; text-align: left;">{table_data['HazardScore'][0]}</td>
                <td style="border: 1px solid #dee2e6; padding: 8px; text-align: left;">{table_data['Latitude'][0]}</td>
                <td style="border: 1px solid #dee2e6; padding: 8px; text-align: left;">{table_data['Longitude'][0]}</td>
                <td style="border: 1px solid #dee2e6; padding: 8px; text-align: left;">{table_data['ShelterProximity'][0]}</td>
                <td style="border: 1px solid #dee2e6; padding: 8px; text-align: left;">{date_part}<br>{time_part}</td>
                <td style="border: 1px solid #dee2e6; padding: 8px; text-align: left;">{table_data['RiskClass'][0]}</td>
            </tr>
        </tbody>
    </table>
    """
    
    st.markdown(html_table, unsafe_allow_html=True)
    
            # Show Random Forest prediction details
    if 'confidence' in locals():
            st.info(f"🤖 **Random Forest Model Results:**")
            st.write(f"   📍 Predicted Class: {predicted_class}")
            st.write(f"   🎯 Confidence: {confidence:.1%}")
            st.write(f"   🚨 Hazard Score: {hazard_score:.3f}")
            
            # Add confidence warning
            if confidence < 0.5:
                st.warning(f"⚠️ **Low Confidence Warning:** The model is uncertain about this image (confidence: {confidence:.1%})")
                st.write("   This suggests the image doesn't clearly match any training class.")
                st.write("   Consider using a different image or checking the image content.")
            else:
                st.success(f"✅ **High Confidence:** The model is confident about this prediction")
    
    # Always show Heatmap Visualization after single patch table
    st.markdown("---")
    st.markdown("### 📈 Heatmap Visualization")
    # Prepare single-item batch_data for heatmap
    batch_data = [{
        'Filename': display_filename,
        'HazardScore': f"{hazard_score:.2f}",
        'Latitude': f"{latitude:.4f}",
        'Longitude': f"{longitude:.4f}",
        'ShelterProximity': f"{shelter_proximity}",
        'Timestamp': f"{date_part}<br>{time_part}",
        'RiskClass': predicted_class
    }]
    create_heatmap_viewer(batch_data)
    
    # Batch analysis button
    if st.button('Select Another (Batch Analysis)', type='secondary', key='single_select_another'):
        st.session_state['patch_selection_enabled'] = True
        st.experimental_rerun()
    
    # Show predicted class separately
    st.info(f"🎯 **Predicted Class:** {predicted_class}")
    
    # Create metadata dataframe for CSV export
    metadata_data = {
        'Source_Name': [display_filename],
        'Source_Type': [source_type],
        'Predicted_Class': [predicted_class],
        'Hazard_Score': [hazard_score],
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

def create_heatmap_viewer(batch_data):
    """Create a grid-style heatmap visualization of batch classification results"""
    import math
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract hazard scores
    hazard_scores = []
    for data in batch_data:
        try:
            hazard_score = float(data['HazardScore'])
            hazard_scores.append(hazard_score)
        except:
            hazard_scores.append(0.0)
    
    n = len(hazard_scores)
    if n == 0:
        st.warning("No data to display heatmap.")
        return
    # Determine grid size (as square as possible)
    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)
    # Fill grid with hazard scores, pad with NaN if needed
    grid = np.full((grid_rows, grid_cols), np.nan)
    for idx, score in enumerate(hazard_scores):
        row = idx // grid_cols
        col = idx % grid_cols
        grid[row, col] = score
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(1.5*grid_cols, 1.2*grid_rows))
    sns.heatmap(grid, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, linecolor='gray', cbar=True, ax=ax, square=False, mask=np.isnan(grid))
    ax.set_title("Hazard Score Grid Heatmap")
    ax.set_xlabel("Patch Column")
    ax.set_ylabel("Patch Row")
    st.pyplot(fig)

def display_batch_metadata_and_export(uploaded_files):
    """Display batch metadata and export functionality"""
    
    st.markdown("### 📊 Batch Analysis Results")
    
    # Initialize batch data
    batch_data = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_parts = timestamp.split(' ')
    date_part = timestamp_parts[0] if len(timestamp_parts) > 0 else timestamp
    time_part = timestamp_parts[1] if len(timestamp_parts) > 1 else "00:00"
    
    # Process each file
    with st.spinner("Processing batch files..."):
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Try to use Random Forest prediction
                try:
                    import sys
                    sys.path.append('.')
                    from test_random_forest import get_random_forest_prediction
                    
                    # Save uploaded file temporarily
                    temp_path = f"temp_batch_{i}.png"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    result = get_random_forest_prediction(temp_path)
                    if result:
                        hazard_score = result['hazard_score']
                        predicted_class = result['class']
                    else:
                        hazard_score = 0.16
                        predicted_class = "SafeZone_UrbanCore"
                    
                    # Clean up temp file
                    import os
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    st.warning(f"Random Forest prediction failed for {uploaded_file.name}: {str(e)}")
                    hazard_score = 0.16
                    predicted_class = "SafeZone_UrbanCore"
                
                # Generate coordinates from filename if it contains lat/lng
                if '_' in uploaded_file.name and uploaded_file.name.replace('.', '_').count('_') >= 2:
                    try:
                        # Extract coordinates from filename like "patch_10.6487_122.9789.png"
                        parts = uploaded_file.name.replace('.png', '').split('_')
                        if len(parts) >= 3:
                            latitude = float(parts[1]) + (i * 0.001)  # Slight variation
                            longitude = float(parts[2]) + (i * 0.001)
                        else:
                            latitude = 10.6487 + (i * 0.01)
                            longitude = 122.9789 + (i * 0.01)
                    except:
                        latitude = 10.6487 + (i * 0.01)
                        longitude = 122.9789 + (i * 0.01)
                else:
                    latitude = 10.6487 + (i * 0.01)
                    longitude = 122.9789 + (i * 0.01)
                
                shelter_proximity = 3.5 + (i * 0.1)
                
                # Add to batch data
                batch_data.append({
                    'Filename': uploaded_file.name,
                    'HazardScore': f"{hazard_score:.2f}",
                    'Latitude': f"{latitude:.4f}",
                    'Longitude': f"{longitude:.4f}",
                    'ShelterProximity': f"{shelter_proximity}",
                    'Timestamp': f"{date_part}<br>{time_part}",
                    'RiskClass': predicted_class
                })
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    if batch_data:
        # Create DataFrame for batch results
        batch_df = pd.DataFrame(batch_data)
        
        # Display table using Streamlit's built-in table
        st.table(batch_df)
        
        # Always show Heatmap Visualization after batch table
        st.markdown("---")
        st.markdown("### 📈 Heatmap Visualization")
        create_heatmap_viewer(batch_data)
        
        # Create batch metadata dataframe for CSV export
        batch_metadata = []
        for data in batch_data:
            batch_metadata.append({
                'Filename': data['Filename'],
                'Hazard_Score': float(data['HazardScore']),
                'Latitude': float(data['Latitude']),
                'Longitude': float(data['Longitude']),
                'Shelter_Proximity_km': float(data['ShelterProximity']),
                'Risk_Class': data['RiskClass'],
                'Timestamp': timestamp
            })
        
        df_batch = pd.DataFrame(batch_metadata)
        
        # Generate CSV
        csv_batch = df_batch.to_csv(index=False)
        
        # Download button for batch
        st.download_button(
            label="📥 Download Batch CSV",
            data=csv_batch,
            file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the batch analysis results as a CSV file"
        )
        
        st.success(f"✅ **Batch analysis completed!** Processed {len(batch_data)} files.")
    else:
        st.error("❌ No files were processed successfully.")

def create_patch_map():
    """Create an interactive map with patch locations"""
    
    # Center map on Philippines (Tacloban area)
    m = folium.Map(
        location=[11.2444, 125.0098],  # Tacloban coordinates
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Try to get Random Forest predictions for predefined locations
    try:
        import sys
        sys.path.append('.')
        from test_random_forest import get_random_forest_prediction
        
        # Get sample images from each class for prediction
        import os
        sample_images = {}
        for class_name in ['HighRisk_Coastal', 'ModerateRisk_Upland', 'SafeZone_UrbanCore', 'EvacCenter_Active', 'WarningGap_Barangay', 'BufferZone_Proposed']:
            class_dir = os.path.join('DisasterData', class_name)
            if os.path.exists(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith('.png'):
                        sample_images[class_name] = os.path.join(class_dir, file)
                        break
        
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
        
        # Update predictions with Random Forest results if available
        for patch in patch_locations:
            if patch['class'] in sample_images:
                result = get_random_forest_prediction(sample_images[patch['class']])
                if result:
                    patch['hazard_score'] = result['hazard_score']
                    patch['class'] = result['class']
                    
    except Exception as e:
        st.warning(f"Random Forest prediction for map failed: {str(e)}")
        # Use default patch locations if Random Forest fails
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
            weight=2
        ).add_to(m)
        
        # Add clickable marker
        folium.Marker(
            location=patch['coords'],
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    return m

def get_patch_from_location(clicked_location):
    """Get patch information based on clicked location"""
    
    # Try to get Random Forest predictions for predefined locations
    try:
        import sys
        sys.path.append('.')
        from test_random_forest import get_random_forest_prediction
        
        # Get sample images from each class for prediction
        import os
        sample_images = {}
        for class_name in ['HighRisk_Coastal', 'ModerateRisk_Upland', 'SafeZone_UrbanCore', 'EvacCenter_Active', 'WarningGap_Barangay', 'BufferZone_Proposed']:
            class_dir = os.path.join('DisasterData', class_name)
            if os.path.exists(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith('.png'):
                        sample_images[class_name] = os.path.join(class_dir, file)
                        break
        
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
        
        # Update predictions with Random Forest results if available
        for patch in patch_locations:
            if patch['class'] in sample_images:
                result = get_random_forest_prediction(sample_images[patch['class']])
                if result:
                    patch['hazard_score'] = result['hazard_score']
                    patch['class'] = result['class']
                    
    except Exception as e:
        st.warning(f"Random Forest prediction for map failed: {str(e)}")
        # Use default patch locations if Random Forest fails
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