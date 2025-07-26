#!/usr/bin/env python3
"""
Generate Training Data for CNN
This script creates sample images for each disaster class to train the CNN
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import glob

def create_sample_images():
    """Create sample images for each disaster class"""
    
    # Define class directories
    classes = [
        'HighRisk_Coastal',
        'ModerateRisk_Upland', 
        'SafeZone_UrbanCore',
        'EvacCenter_Active',
        'WarningGap_Barangay',
        'BufferZone_Proposed'
    ]
    
    # Create base directory
    base_dir = 'DisasterData'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create class directories
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    # Generate sample images for each class
    samples_per_class = 50  # Generate 50 samples per class
    
    for class_name in classes:
        print(f"Generating {samples_per_class} samples for {class_name}...")
        
        for i in range(samples_per_class):
            # Create a 224x224 image
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Generate different patterns based on class
            if class_name == 'HighRisk_Coastal':
                # Red/orange colors for high risk
                img = create_high_risk_pattern()
            elif class_name == 'ModerateRisk_Upland':
                # Yellow/orange colors for moderate risk
                img = create_moderate_risk_pattern()
            elif class_name == 'SafeZone_UrbanCore':
                # Green colors for safe zones
                img = create_safe_zone_pattern()
            elif class_name == 'EvacCenter_Active':
                # Blue colors for evacuation centers
                img = create_evac_center_pattern()
            elif class_name == 'WarningGap_Barangay':
                # Purple/red colors for warning gaps
                img = create_warning_gap_pattern()
            elif class_name == 'BufferZone_Proposed':
                # Orange/yellow colors for buffer zones
                img = create_buffer_zone_pattern()
            
            # Add some noise and variations
            img = add_variations(img)
            
            # Save the image
            filename = f"{class_name}_sample_{i:03d}.png"
            filepath = os.path.join(base_dir, class_name, filename)
            cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"Generated {samples_per_class * len(classes)} training images!")

def create_high_risk_pattern():
    """Create high risk coastal pattern (red/orange)"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Base red color
    base_color = [random.randint(150, 255), random.randint(0, 100), random.randint(0, 100)]
    
    # Create wave-like patterns
    for y in range(224):
        for x in range(224):
            # Add wave effect
            wave = int(20 * np.sin(x * 0.1) + 20 * np.cos(y * 0.1))
            intensity = random.randint(-30, 30)
            
            r = min(255, max(0, base_color[0] + wave + intensity))
            g = min(255, max(0, base_color[1] + wave + intensity))
            b = min(255, max(0, base_color[2] + wave + intensity))
            
            img[y, x] = [r, g, b]
    
    return img

def create_moderate_risk_pattern():
    """Create moderate risk upland pattern (yellow/orange)"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Base orange color
    base_color = [random.randint(200, 255), random.randint(100, 200), random.randint(0, 100)]
    
    # Create terrain-like patterns
    for y in range(224):
        for x in range(224):
            # Add terrain effect
            terrain = int(15 * np.sin(x * 0.05) * np.cos(y * 0.05))
            intensity = random.randint(-20, 20)
            
            r = min(255, max(0, base_color[0] + terrain + intensity))
            g = min(255, max(0, base_color[1] + terrain + intensity))
            b = min(255, max(0, base_color[2] + terrain + intensity))
            
            img[y, x] = [r, g, b]
    
    return img

def create_safe_zone_pattern():
    """Create safe zone urban pattern (green)"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Base green color
    base_color = [random.randint(0, 100), random.randint(150, 255), random.randint(0, 100)]
    
    # Create urban grid pattern
    for y in range(224):
        for x in range(224):
            # Add grid effect
            grid = int(10 * (np.sin(x * 0.2) + np.cos(y * 0.2)))
            intensity = random.randint(-15, 15)
            
            r = min(255, max(0, base_color[0] + grid + intensity))
            g = min(255, max(0, base_color[1] + grid + intensity))
            b = min(255, max(0, base_color[2] + grid + intensity))
            
            img[y, x] = [r, g, b]
    
    return img

def create_evac_center_pattern():
    """Create evacuation center pattern (blue)"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Base blue color
    base_color = [random.randint(0, 100), random.randint(0, 100), random.randint(150, 255)]
    
    # Create building-like pattern
    for y in range(224):
        for x in range(224):
            # Add building effect
            building = int(25 * np.sin(x * 0.1) + 25 * np.cos(y * 0.1))
            intensity = random.randint(-25, 25)
            
            r = min(255, max(0, base_color[0] + building + intensity))
            g = min(255, max(0, base_color[1] + building + intensity))
            b = min(255, max(0, base_color[2] + building + intensity))
            
            img[y, x] = [r, g, b]
    
    return img

def create_warning_gap_pattern():
    """Create warning gap pattern (purple/red)"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Base purple color
    base_color = [random.randint(100, 200), random.randint(0, 100), random.randint(150, 255)]
    
    # Create gap-like pattern
    for y in range(224):
        for x in range(224):
            # Add gap effect
            gap = int(30 * np.sin(x * 0.15) - 30 * np.cos(y * 0.15))
            intensity = random.randint(-35, 35)
            
            r = min(255, max(0, base_color[0] + gap + intensity))
            g = min(255, max(0, base_color[1] + gap + intensity))
            b = min(255, max(0, base_color[2] + gap + intensity))
            
            img[y, x] = [r, g, b]
    
    return img

def create_buffer_zone_pattern():
    """Create buffer zone pattern (orange/yellow)"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Base orange-yellow color
    base_color = [random.randint(180, 255), random.randint(120, 200), random.randint(0, 80)]
    
    # Create buffer-like pattern
    for y in range(224):
        for x in range(224):
            # Add buffer effect
            buffer = int(20 * np.sin(x * 0.08) + 20 * np.cos(y * 0.08))
            intensity = random.randint(-25, 25)
            
            r = min(255, max(0, base_color[0] + buffer + intensity))
            g = min(255, max(0, base_color[1] + buffer + intensity))
            b = min(255, max(0, base_color[2] + buffer + intensity))
            
            img[y, x] = [r, g, b]
    
    return img

def add_variations(img):
    """Add random variations to make images more realistic"""
    
    # Add some noise
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add some blur randomly
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Add some brightness variation
    brightness = random.uniform(0.8, 1.2)
    img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    
    return img

def create_from_overlays():
    """Create training data from existing overlay images"""
    
    overlay_dir = 'overlays'
    if not os.path.exists(overlay_dir):
        print("Overlays directory not found!")
        return
    
    # Look for overlay images
    overlay_files = glob.glob(os.path.join(overlay_dir, "*.png"))
    
    if not overlay_files:
        print("No overlay images found!")
        return
    
    print(f"Found {len(overlay_files)} overlay images")
    
    # Create base directory
    base_dir = 'DisasterData'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Define class directories
    classes = [
        'HighRisk_Coastal',
        'ModerateRisk_Upland', 
        'SafeZone_UrbanCore',
        'EvacCenter_Active',
        'WarningGap_Barangay',
        'BufferZone_Proposed'
    ]
    
    # Create class directories
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    # Process each overlay file
    for overlay_file in overlay_files:
        print(f"Processing {overlay_file}...")
        
        # Load overlay image
        overlay = cv2.imread(overlay_file)
        if overlay is None:
            continue
            
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Create patches from overlay
        height, width = overlay.shape[:2]
        patch_size = 224
        
        # Generate random patches
        for class_name in classes:
            for i in range(10):  # 10 patches per class per overlay
                # Random position
                x = random.randint(0, max(0, width - patch_size))
                y = random.randint(0, max(0, height - patch_size))
                
                # Extract patch
                patch = overlay[y:y+patch_size, x:x+patch_size]
                
                # Resize if needed
                if patch.shape[:2] != (patch_size, patch_size):
                    patch = cv2.resize(patch, (patch_size, patch_size))
                
                # Add variations
                patch = add_variations(patch)
                
                # Save patch
                filename = f"{class_name}_overlay_{os.path.basename(overlay_file)}_{i:03d}.png"
                filepath = os.path.join(base_dir, class_name, filename)
                cv2.imwrite(filepath, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

def main():
    """Main function"""
    print("=== Generating Training Data for CNN ===\n")
    
    # Option 1: Create synthetic samples
    print("1. Creating synthetic training samples...")
    create_sample_images()
    
    # Option 2: Create from overlays (if available)
    print("\n2. Creating samples from overlay images...")
    create_from_overlays()
    
    print("\n=== Training Data Generation Complete ===")
    print("Check the DisasterData directory for generated images.")

if __name__ == "__main__":
    main() 