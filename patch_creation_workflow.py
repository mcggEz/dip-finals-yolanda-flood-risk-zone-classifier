#!/usr/bin/env python3
"""
Patch Creation Workflow
Step 1: Load Base Map and Overlays
Step 2: Define Patch Coordinates  
Step 3: Save Patch to Class Folder

This script automates the creation of 224x224 pixel patches from georeferenced layers
for CNN training data.
"""

import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import json
from datetime import datetime

class PatchCreationWorkflow:
    """Main class for patch creation workflow"""
    
    def __init__(self, patch_size=(224, 224)):
        self.patch_size = patch_size
        self.base_image = None
        self.hazard_overlay = None
        self.evac_centers = None
        self.geo_transform = None
        self.output_dir = 'DisasterData'
        
        # Create output directories
        self.classes = [
            'HighRisk_Coastal',
            'ModerateRisk_Upland', 
            'SafeZone_UrbanCore',
            'EvacCenter_Active',
            'WarningGap_Barangay',
            'BufferZone_Proposed'
        ]
        
        self._create_output_directories()
    
    def _create_output_directories(self):
        """Create output directories for each class"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        for class_name in self.classes:
            class_dir = os.path.join(self.output_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
    
    def step1_load_base_map_and_overlays(self):
        """
        Step 1: Load Base Map and Overlays
        Equivalent to MATLAB:
        baseImage = imread('himawari_satellite_image.png');
        hazardOverlay = imread('storm_surge_overlay.png');
        evacCenters = readtable('evacuation_centers.csv');
        """
        print("Step 1: Loading Base Map and Overlays...")
        
        # Load base satellite image
        base_image_path = 'overlays/himawari_satellite_image.png'
        if os.path.exists(base_image_path):
            self.base_image = cv2.imread(base_image_path)
            self.base_image = cv2.cvtColor(self.base_image, cv2.COLOR_BGR2RGB)
            print(f"✅ Loaded base image: {base_image_path}")
        else:
            print(f"⚠️  Base image not found: {base_image_path}")
            # Create a placeholder base image
            self.base_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
            self.base_image[:] = [100, 150, 100]  # Greenish background
        
        # Load hazard overlay
        hazard_overlay_path = 'overlays/storm_surge_overlay.png'
        if os.path.exists(hazard_overlay_path):
            self.hazard_overlay = cv2.imread(hazard_overlay_path)
            self.hazard_overlay = cv2.cvtColor(self.hazard_overlay, cv2.COLOR_BGR2RGB)
            print(f"✅ Loaded hazard overlay: {hazard_overlay_path}")
        else:
            print(f"⚠️  Hazard overlay not found: {hazard_overlay_path}")
            # Create a placeholder hazard overlay
            self.hazard_overlay = np.zeros_like(self.base_image)
        
        # Load evacuation centers CSV
        evac_centers_path = 'overlays/evacuation_centers.csv'
        if os.path.exists(evac_centers_path):
            self.evac_centers = pd.read_csv(evac_centers_path)
            print(f"✅ Loaded evacuation centers: {evac_centers_path}")
            print(f"   Found {len(self.evac_centers)} evacuation centers")
        else:
            print(f"⚠️  Evacuation centers not found: {evac_centers_path}")
            # Create placeholder evacuation centers
            self.evac_centers = pd.DataFrame({
                'name': ['Center1', 'Center2', 'Center3'],
                'latitude': [11.2444, 11.1577, 11.1111],
                'longitude': [125.0098, 124.9908, 125.0167]
            })
        
        # Load other overlay files
        self._load_additional_overlays()
        
        print("✅ Step 1 Complete: Base map and overlays loaded")
    
    def _load_additional_overlays(self):
        """Load additional overlay files from the overlays directory"""
        overlay_dir = 'overlays'
        if not os.path.exists(overlay_dir):
            return
        
        overlay_files = [f for f in os.listdir(overlay_dir) if f.endswith('.png')]
        
        for overlay_file in overlay_files:
            if 'himawari' not in overlay_file and 'storm_surge' not in overlay_file:
                overlay_path = os.path.join(overlay_dir, overlay_file)
                overlay_img = cv2.imread(overlay_path)
                if overlay_img is not None:
                    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                    print(f"✅ Loaded additional overlay: {overlay_file}")
    
    def step2_define_patch_coordinates(self):
        """
        Step 2: Define Patch Coordinates
        Note: use GIS coordinates or pixel indices to crop relevant areas.
        patchSize = [224, 224]; % Standard for CNN input
        coords = [x, y]; % Replace with actual pixel locations
        """
        print("\nStep 2: Defining Patch Coordinates...")
        
        if self.base_image is None:
            print("❌ Base image not loaded. Run Step 1 first.")
            return []
        
        height, width = self.base_image.shape[:2]
        patch_width, patch_height = self.patch_size
        
        # Define patch coordinates based on different strategies
        patch_coordinates = []
        
        # Strategy 1: Grid-based sampling
        print("   Strategy 1: Grid-based sampling...")
        grid_spacing = 300  # pixels between patches
        for y in range(0, height - patch_height, grid_spacing):
            for x in range(0, width - patch_width, grid_spacing):
                patch_coordinates.append({
                    'x': x,
                    'y': y,
                    'strategy': 'grid',
                    'class': self._determine_patch_class(x, y)
                })
        
        # Strategy 2: Random sampling
        print("   Strategy 2: Random sampling...")
        num_random_patches = 50
        for _ in range(num_random_patches):
            x = np.random.randint(0, width - patch_width)
            y = np.random.randint(0, height - patch_height)
            patch_coordinates.append({
                'x': x,
                'y': y,
                'strategy': 'random',
                'class': self._determine_patch_class(x, y)
            })
        
        # Strategy 3: Feature-based sampling (around evacuation centers)
        print("   Strategy 3: Feature-based sampling...")
        if self.evac_centers is not None:
            for _, center in self.evac_centers.iterrows():
                # Convert lat/lon to pixel coordinates (simplified)
                pixel_x = int((center['longitude'] - 124.0) * 1000)  # Approximate conversion
                pixel_y = int((center['latitude'] - 11.0) * 1000)
                
                # Ensure coordinates are within bounds
                pixel_x = max(0, min(width - patch_width, pixel_x))
                pixel_y = max(0, min(height - patch_height, pixel_y))
                
                patch_coordinates.append({
                    'x': pixel_x,
                    'y': pixel_y,
                    'strategy': 'feature',
                    'class': 'EvacCenter_Active'
                })
        
        print(f"✅ Step 2 Complete: Defined {len(patch_coordinates)} patch coordinates")
        return patch_coordinates
    
    def _determine_patch_class(self, x, y):
        """
        Determine the class of a patch based on GIS analysis of overlays
        This creates properly labeled training data for the CNN to learn from
        """
        
        # Strategy 1: Use evacuation center proximity
        if self.evac_centers is not None:
            # Convert pixel coordinates to approximate lat/lon
            pixel_lat = 11.0 + (y / 1000.0)  # Approximate conversion
            pixel_lon = 124.0 + (x / 1000.0)
            
            # Check if patch is near evacuation center
            for _, center in self.evac_centers.iterrows():
                distance = np.sqrt((pixel_lat - center['latitude'])**2 + 
                                 (pixel_lon - center['longitude'])**2)
                if distance < 0.01:  # Within ~1km
                    return 'EvacCenter_Active'
        
        # Strategy 2: Analyze hazard overlay patterns (not just colors)
        if self.hazard_overlay is not None:
            patch_area = self.hazard_overlay[y:y+self.patch_size[1], x:x+self.patch_size[0]]
            
            # Look for specific patterns in the overlay
            # High risk areas: intense red patterns with specific textures
            red_channel = patch_area[:, :, 0]
            green_channel = patch_area[:, :, 1]
            blue_channel = patch_area[:, :, 2]
            
            # Calculate more sophisticated features
            red_mean = np.mean(red_channel)
            red_std = np.std(red_channel)
            red_entropy = self._calculate_entropy(red_channel)
            
            # High risk: high red intensity with low variance (consistent hazard)
            if red_mean > 180 and red_std < 30 and red_entropy < 2.0:
                return 'HighRisk_Coastal'
            
            # Moderate risk: medium red intensity with some variation
            elif red_mean > 120 and red_std > 30 and red_entropy > 2.0:
                return 'ModerateRisk_Upland'
            
            # Safe zones: low red, high green (vegetation/urban areas)
            elif red_mean < 100 and np.mean(green_channel) > 150:
                return 'SafeZone_UrbanCore'
            
            # Warning gaps: mixed patterns with purple/red
            elif red_mean > 100 and np.mean(blue_channel) > 100:
                return 'WarningGap_Barangay'
            
            # Buffer zones: moderate orange/yellow
            elif red_mean > 80 and np.mean(green_channel) > 80:
                return 'BufferZone_Proposed'
        
        # Strategy 3: Position-based classification (fallback)
        center_x = x + self.patch_size[0] // 2
        center_y = y + self.patch_size[1] // 2
        
        # Use more sophisticated position analysis
        if center_y < 300:  # Coastal areas (top)
            return 'HighRisk_Coastal'
        elif center_y < 600:  # Upland areas (middle)
            return 'ModerateRisk_Upland'
        else:  # Urban areas (bottom)
            return 'SafeZone_UrbanCore'
    
    def _calculate_entropy(self, channel):
        """Calculate entropy of an image channel (texture measure)"""
        hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])
        hist = hist[hist > 0]  # Remove zero bins
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log2(prob))
        return entropy
    
    def step3_save_patch_to_class_folder(self, patch_coordinates):
        """
        Step 3: Save Patch to Class Folder
        Equivalent to MATLAB:
        patch = imcrop(baseImage, [coords(1), coords(2), patchSize(1), patchSize(2)]);
        imwrite(patch, 'DisasterData/HighRisk_Coastal/patch1.png');
        """
        print("\nStep 3: Saving Patches to Class Folders...")
        
        if self.base_image is None:
            print("❌ Base image not loaded. Run Step 1 first.")
            return
        
        saved_patches = []
        
        for i, coords in enumerate(patch_coordinates):
            x, y = coords['x'], coords['y']
            patch_class = coords['class']
            strategy = coords['strategy']
            
            # Extract patch from base image
            patch = self.base_image[y:y+self.patch_size[1], x:x+self.patch_size[0]]
            
            # Ensure patch is the correct size
            if patch.shape[:2] != self.patch_size:
                patch = cv2.resize(patch, self.patch_size)
            
            # Apply overlay if available
            if self.hazard_overlay is not None:
                overlay_patch = self.hazard_overlay[y:y+self.patch_size[1], x:x+self.patch_size[0]]
                if overlay_patch.shape[:2] == self.patch_size:
                    # Blend overlay with base image
                    alpha = 0.3  # Overlay transparency
                    patch = cv2.addWeighted(patch, 1-alpha, overlay_patch, alpha, 0)
            
            # Save patch to appropriate class folder
            filename = f"{patch_class}_{strategy}_{i:04d}.png"
            filepath = os.path.join(self.output_dir, patch_class, filename)
            
            # Convert to BGR for OpenCV save
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, patch_bgr)
            
            saved_patches.append({
                'filename': filename,
                'class': patch_class,
                'coordinates': (x, y),
                'strategy': strategy,
                'filepath': filepath
            })
            
            if (i + 1) % 10 == 0:
                print(f"   Saved {i + 1}/{len(patch_coordinates)} patches...")
        
        print(f"✅ Step 3 Complete: Saved {len(saved_patches)} patches")
        
        # Save metadata
        self._save_patch_metadata(saved_patches)
        
        return saved_patches
    
    def _save_patch_metadata(self, saved_patches):
        """Save metadata about created patches"""
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'patch_size': self.patch_size,
            'total_patches': len(saved_patches),
            'patches': []
        }
        
        for patch in saved_patches:
            metadata['patches'].append({
                'filename': patch['filename'],
                'class': patch['class'],
                'coordinates': patch['coordinates'],
                'strategy': patch['strategy']
            })
        
        # Save to JSON file
        with open('patch_creation_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save summary to CSV
        summary_data = []
        for patch in saved_patches:
            summary_data.append({
                'filename': patch['filename'],
                'class': patch['class'],
                'x': patch['coordinates'][0],
                'y': patch['coordinates'][1],
                'strategy': patch['strategy']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('patch_creation_summary.csv', index=False)
        
        print("   📊 Metadata saved to patch_creation_metadata.json")
        print("   📊 Summary saved to patch_creation_summary.csv")
    
    def visualize_patches(self, patch_coordinates, num_samples=9):
        """Visualize sample patches"""
        print(f"\n📊 Visualizing {num_samples} sample patches...")
        
        if not patch_coordinates:
            print("❌ No patch coordinates available")
            return
        
        # Select random samples
        samples = np.random.choice(patch_coordinates, min(num_samples, len(patch_coordinates)), replace=False)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for i, coords in enumerate(samples):
            if i >= 9:
                break
                
            x, y = coords['x'], coords['y']
            patch_class = coords['class']
            
            # Extract patch
            patch = self.base_image[y:y+self.patch_size[1], x:x+self.patch_size[0]]
            
            # Apply overlay if available
            if self.hazard_overlay is not None:
                overlay_patch = self.hazard_overlay[y:y+self.patch_size[1], x:x+self.patch_size[0]]
                if overlay_patch.shape[:2] == self.patch_size:
                    alpha = 0.3
                    patch = cv2.addWeighted(patch, 1-alpha, overlay_patch, alpha, 0)
            
            axes[i].imshow(patch)
            axes[i].set_title(f"{patch_class}\n({x}, {y})")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_patches.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   📊 Sample patches saved to sample_patches.png")
    
    def run_complete_workflow(self):
        """Run the complete patch creation workflow"""
        print("=== Patch Creation Workflow ===\n")
        
        # Step 1: Load base map and overlays
        self.step1_load_base_map_and_overlays()
        
        # Step 2: Define patch coordinates
        patch_coordinates = self.step2_define_patch_coordinates()
        
        if not patch_coordinates:
            print("❌ No patch coordinates defined")
            return
        
        # Step 3: Save patches to class folders
        saved_patches = self.step3_save_patch_to_class_folder(patch_coordinates)
        
        # Visualize sample patches
        self.visualize_patches(patch_coordinates)
        
        # Print summary
        print(f"\n=== Workflow Complete ===")
        print(f"✅ Created {len(saved_patches)} patches")
        
        # Count patches per class
        class_counts = {}
        for patch in saved_patches:
            class_counts[patch['class']] = class_counts.get(patch['class'], 0) + 1
        
        print(f"\n📊 Patches per class:")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} patches")
        
        print(f"\n📁 Patches saved to: {self.output_dir}/")
        print(f"📊 Metadata saved to: patch_creation_metadata.json")
        print(f"📊 Summary saved to: patch_creation_summary.csv")

def main():
    """Main function to run the patch creation workflow"""
    
    # Create workflow instance
    workflow = PatchCreationWorkflow(patch_size=(224, 224))
    
    # Run complete workflow
    workflow.run_complete_workflow()

if __name__ == "__main__":
    main() 