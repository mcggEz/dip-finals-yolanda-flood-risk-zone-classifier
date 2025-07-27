import os
import json
from datetime import datetime
from PIL import Image, PngImagePlugin
import shutil

def add_metadata_to_png(image_path, metadata_dict):
    """Add metadata to a PNG file"""
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Create PngInfo object
        pnginfo = PngImagePlugin.PngInfo()
        
        # Add metadata as text chunks
        for key, value in metadata_dict.items():
            pnginfo.add_text(key, str(value))
        
        # Create output filename with timestamp
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base_name}_with_metadata_{timestamp}.png"
        
        # Save with metadata
        img.save(output_path, "PNG", pnginfo=pnginfo)
        
        return output_path, True
    except Exception as e:
        return str(e), False

def save_to_class_folder(image_path, class_name):
    """Save image to appropriate class folder"""
    try:
        # Create class folder if it doesn't exist
        class_folder = f"DisasterData/{class_name}"
        os.makedirs(class_folder, exist_ok=True)
        
        # Generate filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{class_name.lower()}_{timestamp}.png"
        output_path = os.path.join(class_folder, filename)
        
        # Copy file to class folder
        shutil.copy2(image_path, output_path)
        
        return output_path, True
    except Exception as e:
        return str(e), False

def main():
    # Metadata for each file (m1 to m15)
    metadata_list = [
        # m1
        {
            "center_lat": 11.042762,
            "center_lon": 122.469979,
            "north_lat": 11.047762,
            "south_lat": 11.037762,
            "east_lon": 122.474979,
            "west_lon": 122.464979,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:08:09",
            "center_coordinates": "11.042762°, 122.469979°"
        },
        # m2
        {
            "center_lat": 11.057462,
            "center_lon": 122.493398,
            "north_lat": 11.062462,
            "south_lat": 11.052462,
            "east_lon": 122.498398,
            "west_lon": 122.488398,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:11:51",
            "center_coordinates": "11.057462°, 122.493398°"
        },
        # m3
        {
            "center_lat": 11.078521,
            "center_lon": 122.540825,
            "north_lat": 11.083521,
            "south_lat": 11.073521,
            "east_lon": 122.545825,
            "west_lon": 122.535825,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:12:35",
            "center_coordinates": "11.078521°, 122.540825°"
        },
        # m4
        {
            "center_lat": 11.093945,
            "center_lon": 122.599322,
            "north_lat": 11.098945,
            "south_lat": 11.088945,
            "east_lon": 122.604322,
            "west_lon": 122.594322,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:13:30",
            "center_coordinates": "11.093945°, 122.599322°"
        },
        # m5
        {
            "center_lat": 11.132087,
            "center_lon": 122.684209,
            "north_lat": 11.137087,
            "south_lat": 11.127087,
            "east_lon": 122.689209,
            "west_lon": 122.679209,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:14:11",
            "center_coordinates": "11.132087°, 122.684209°"
        },
        # m6
        {
            "center_lat": 11.287455,
            "center_lon": 122.929073,
            "north_lat": 11.292455,
            "south_lat": 11.282455,
            "east_lon": 122.934073,
            "west_lon": 122.924073,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:14:53",
            "center_coordinates": "11.287455°, 122.929073°"
        },
        # m7
        {
            "center_lat": 11.287455,
            "center_lon": 122.929073,
            "north_lat": 11.292455,
            "south_lat": 11.282455,
            "east_lon": 122.934073,
            "west_lon": 122.924073,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:14:53",
            "center_coordinates": "11.287455°, 122.929073°"
        },
        # m8
        {
            "center_lat": 11.295198,
            "center_lon": 122.951497,
            "north_lat": 11.300198,
            "south_lat": 11.290198,
            "east_lon": 122.956497,
            "west_lon": 122.946497,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:16:44",
            "center_coordinates": "11.295198°, 122.951497°"
        },
        # m9
        {
            "center_lat": 11.056114,
            "center_lon": 124.401746,
            "north_lat": 11.061114,
            "south_lat": 11.051114,
            "east_lon": 124.406746,
            "west_lon": 124.396746,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:17:27",
            "center_coordinates": "11.056114°, 124.401746°"
        },
        # m10
        {
            "center_lat": 11.025724,
            "center_lon": 124.418729,
            "north_lat": 11.030724,
            "south_lat": 11.020724,
            "east_lon": 124.423729,
            "west_lon": 124.413729,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:18:04",
            "center_coordinates": "11.025724°, 124.418729°"
        },
        # m11
        {
            "center_lat": 11.170403,
            "center_lon": 124.458020,
            "north_lat": 11.175403,
            "south_lat": 11.165403,
            "east_lon": 124.463020,
            "west_lon": 124.453020,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:18:44",
            "center_coordinates": "11.170403°, 124.458020°"
        },
        # m12
        {
            "center_lat": 11.205324,
            "center_lon": 124.499915,
            "north_lat": 11.210324,
            "south_lat": 11.200324,
            "east_lon": 124.504915,
            "west_lon": 124.494915,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:19:58",
            "center_coordinates": "11.205324°, 124.499915°"
        },
        # m13
        {
            "center_lat": 11.140056,
            "center_lon": 124.810901,
            "north_lat": 11.145056,
            "south_lat": 11.135056,
            "east_lon": 124.815901,
            "west_lon": 124.805901,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:21:37",
            "center_coordinates": "11.140056°, 124.810901°"
        },
        # m14
        {
            "center_lat": 11.239505,
            "center_lon": 124.767505,
            "north_lat": 11.244505,
            "south_lat": 11.234505,
            "east_lon": 124.772505,
            "west_lon": 124.762505,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:22:20",
            "center_coordinates": "11.239505°, 124.767505°"
        },
        # m15
        {
            "center_lat": 11.159477,
            "center_lon": 124.575028,
            "north_lat": 11.164477,
            "south_lat": 11.154477,
            "east_lon": 124.580028,
            "west_lon": 124.570028,
            "risk_class": "ModerateRisk_Upland",
            "resolution": "224x224",
            "timestamp": "2025-07-27 16:20:41",
            "center_coordinates": "11.159477°, 124.575028°"
        }
    ]
    
    # Process each file
    moderate_folder = "moderate"
    results = []
    
    print("🚀 Starting batch metadata processing...")
    print(f"📁 Processing files from: {moderate_folder}")
    print("=" * 50)
    
    for i, metadata in enumerate(metadata_list, 1):
        filename = f"m{i}.png"
        file_path = os.path.join(moderate_folder, filename)
        
        print(f"📄 Processing {filename}...")
        
        if os.path.exists(file_path):
            # Add metadata to PNG
            output_path, success = add_metadata_to_png(file_path, metadata)
            
            if success:
                print(f"  ✅ Metadata added: {output_path}")
                
                # Save to class folder
                class_path, class_success = save_to_class_folder(output_path, "ModerateRisk_Upland")
                
                if class_success:
                    print(f"  ✅ Saved to class folder: {class_path}")
                    results.append({
                        "file": filename,
                        "status": "Success",
                        "output": class_path,
                        "coordinates": metadata["center_coordinates"],
                        "timestamp": metadata["timestamp"]
                    })
                else:
                    print(f"  ❌ Error saving to class folder: {class_path}")
                    results.append({
                        "file": filename,
                        "status": "Partial Success",
                        "output": output_path,
                        "coordinates": metadata["center_coordinates"],
                        "timestamp": metadata["timestamp"]
                    })
            else:
                print(f"  ❌ Error adding metadata: {output_path}")
                results.append({
                    "file": filename,
                    "status": "Failed",
                    "output": str(output_path),
                    "coordinates": metadata["center_coordinates"],
                    "timestamp": metadata["timestamp"]
                })
        else:
            print(f"  ❌ File not found: {file_path}")
            results.append({
                "file": filename,
                "status": "File Not Found",
                "output": "N/A",
                "coordinates": metadata["center_coordinates"],
                "timestamp": metadata["timestamp"]
            })
        
        print()
    
    # Summary
    print("=" * 50)
    print("📊 Processing Summary:")
    print("=" * 50)
    
    success_count = sum(1 for r in results if r["status"] == "Success")
    partial_count = sum(1 for r in results if r["status"] == "Partial Success")
    failed_count = sum(1 for r in results if r["status"] in ["Failed", "File Not Found"])
    
    print(f"✅ Successful: {success_count}")
    print(f"⚠️  Partial Success: {partial_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 Total Processed: {len(results)}")
    
    print("\n📋 Detailed Results:")
    for result in results:
        status_emoji = "✅" if result["status"] == "Success" else "⚠️" if result["status"] == "Partial Success" else "❌"
        print(f"{status_emoji} {result['file']}: {result['status']}")
        if result["status"] != "File Not Found":
            print(f"   📍 {result['coordinates']}")
            print(f"   🕒 {result['timestamp']}")
    
    print("\n🎉 Batch processing complete!")

if __name__ == "__main__":
    main() 