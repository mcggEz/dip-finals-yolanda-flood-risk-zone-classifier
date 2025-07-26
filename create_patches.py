#!/usr/bin/env python3
"""
Simple Patch Creation Script
Run this to create training patches from your overlays
"""

import os
import sys

def check_overlays():
    """Check if overlay files exist"""
    overlay_dir = 'overlays'
    if not os.path.exists(overlay_dir):
        print("❌ Overlays directory not found!")
        return False
    
    overlay_files = [f for f in os.listdir(overlay_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not overlay_files:
        print("❌ No overlay images found in overlays/ directory!")
        return False
    
    print("📁 Found overlay files:")
    for file in overlay_files:
        print(f"   ✅ {file}")
    
    return True

def main():
    """Main function to run patch creation"""
    print("=== Patch Creation Workflow ===\n")
    
    # Check if overlays exist
    if not check_overlays():
        print("\n❌ Please ensure you have overlay images in the 'overlays/' directory")
        print("   Required files:")
        print("   - himawari_satellite_image.png (base map)")
        print("   - storm_surge_overlay.png (hazard overlay)")
        print("   - evacuation_centers.csv (evacuation center data)")
        print("   - Other overlay PNG files")
        return
    
    print("\n🚀 Starting patch creation workflow...")
    
    try:
        # Import and run the patch creation workflow
        from patch_creation_workflow import PatchCreationWorkflow
        
        # Create workflow instance
        workflow = PatchCreationWorkflow(patch_size=(224, 224))
        
        # Run complete workflow
        workflow.run_complete_workflow()
        
        print("\n✅ Patch creation completed successfully!")
        print("📁 Check the following files:")
        print("   - DisasterData/ (training patches)")
        print("   - patch_creation_metadata.json (metadata)")
        print("   - patch_creation_summary.csv (summary)")
        print("   - sample_patches.png (visualization)")
        
        print("\n🎯 Next steps:")
        print("   1. Review the created patches in DisasterData/")
        print("   2. Train the CNN: python train_cnn.py")
        print("   3. Test predictions: python predict_patch.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Patch creation error: {e}")
        print("Please check your overlay files and try again.")

if __name__ == "__main__":
    main() 