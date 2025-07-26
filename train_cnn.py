#!/usr/bin/env python3
"""
Simple CNN Training Script
Run this after adding PNG files to DisasterData folders
"""

import os
import sys

def check_data():
    """Check if training data exists"""
    data_dir = 'DisasterData'
    if not os.path.exists(data_dir):
        print("❌ DisasterData directory not found!")
        return False
    
    classes = [
        'HighRisk_Coastal',
        'ModerateRisk_Upland', 
        'SafeZone_UrbanCore',
        'EvacCenter_Active',
        'WarningGap_Barangay',
        'BufferZone_Proposed'
    ]
    
    total_images = 0
    print("📁 Checking training data...")
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"❌ {class_name} directory not found!")
            continue
        
        # Count PNG files
        png_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        jpg_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        jpeg_files = [f for f in os.listdir(class_dir) if f.endswith('.jpeg')]
        
        total_files = len(png_files) + len(jpg_files) + len(jpeg_files)
        total_images += total_files
        
        if total_files > 0:
            print(f"✅ {class_name}: {total_files} images")
        else:
            print(f"⚠️  {class_name}: No images found")
    
    print(f"\n📊 Total images: {total_images}")
    
    if total_images == 0:
        print("\n❌ No training images found!")
        print("Please add PNG/JPG images to the DisasterData subfolders first.")
        return False
    
    return True

def main():
    """Main training function"""
    print("=== Disaster Zone CNN Training ===\n")
    
    # Check if data exists
    if not check_data():
        return
    
    print("\n🚀 Starting CNN training...")
    
    try:
        # Import and run the CNN training
        from cnn_disaster_classification import main as train_cnn
        train_cnn()
        
        print("\n✅ Training completed successfully!")
        print("📁 Check the following files:")
        print("   - disaster_classifier_weights.npz (trained model)")
        print("   - classification_results.csv (results)")
        print("   - training_summary.txt (summary)")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("Please check your data and try again.")

if __name__ == "__main__":
    main() 