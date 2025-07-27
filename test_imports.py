#!/usr/bin/env python3
"""
Test script to debug import issues for Streamlit Cloud deployment
"""
import sys
import traceback

def test_imports():
    print("Testing imports...")
    
    try:
        print("1. Testing basic imports...")
        import streamlit as st
        print("✅ streamlit imported successfully")
        
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
    except Exception as e:
        print(f"❌ Basic import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("2. Testing features.patch_selector imports...")
        from features.patch_selector import show_patch_selector
        print("✅ show_patch_selector imported successfully")
        
        from features.patch_selector import display_metadata_and_export
        print("✅ display_metadata_and_export imported successfully")
        
        from features.patch_selector import display_batch_metadata_and_export
        print("✅ display_batch_metadata_and_export imported successfully")
        
        from features.patch_selector import create_heatmap_viewer
        print("✅ create_heatmap_viewer imported successfully")
        
    except Exception as e:
        print(f"❌ features.patch_selector import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("3. Testing features.overlays imports...")
        from features.overlays import show_overlays, render_overlay_main_content
        print("✅ features.overlays imported successfully")
        
    except Exception as e:
        print(f"❌ features.overlays import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("4. Testing test_random_forest imports...")
        from test_random_forest import get_random_forest_prediction
        print("✅ test_random_forest imported successfully")
        
    except Exception as e:
        print(f"❌ test_random_forest import failed: {e}")
        traceback.print_exc()
        return False
    
    print("✅ All imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("🎉 All tests passed! Ready for deployment.")
    else:
        print("❌ Some tests failed. Check the errors above.") 