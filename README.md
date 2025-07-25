# Yolanda Flood Risk Zone Classifier

An interactive dashboard for visualizing, analyzing, and classifying disaster risk zones affected by Typhoon Yolanda using satellite imagery, hazard overlays, and deep learning.

This project is for partial requirement for the Digital Image Processing course in Computer Engineering at Pamantasan ng Lungsod ng Maynila.

## Features
- **Map Viewer:** Visualize Himawari satellite imagery with hazard overlays, shelter markers, and buffer zones.
- **Patch Selector:** Select or upload image patches for risk classification.
- **Metadata Viewer:** View hazard score, elevation, coordinates, shelter proximity, and timestamp for each patch.
- **Batch Analysis:** Upload multiple patches, view classification heatmaps, and export results to CSV.

## Project Structure
```
DIP_FINALS/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── features/               # Modular feature components
│   ├── overlays.py
│   ├── patch_selector.py
│   ├── metadata_viewer.py
│   └── batch_analysis.py
└── data/                   # (Optional) Data files, imagery, shapefiles, etc.
```

**Members:**
- Mc Giberri M. Ginez
- Carlos San Gabriel
- Kurth Angelo Espiritu
- Mary Angelique Terre


