import os
from datetime import datetime
from PIL import Image, PngImagePlugin
import shutil

def add_metadata_to_png(image_path, metadata_dict):
    try:
        img = Image.open(image_path)
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in metadata_dict.items():
            pnginfo.add_text(key, str(value))
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base_name}_with_metadata_{timestamp}.png"
        img.save(output_path, "PNG", pnginfo=pnginfo)
        return output_path, True
    except Exception as e:
        return str(e), False

def save_to_class_folder(image_path, class_name):
    try:
        class_folder = f"DisasterData/{class_name}"
        os.makedirs(class_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{class_name.lower()}_{timestamp}.png"
        output_path = os.path.join(class_folder, filename)
        shutil.copy2(image_path, output_path)
        return output_path, True
    except Exception as e:
        return str(e), False

def main():
    metadata_list = [
        # e1
        {"center_lat": 10.789224, "center_lon": 122.018549, "north_lat": 10.794224, "south_lat": 10.784224, "east_lon": 122.023549, "west_lon": 122.013549, "risk_class": "EvacCenter_Active", "resolution": "224x224", "timestamp": "2025-07-27 16:59:45", "center_coordinates": "10.789224°, 122.018549°"},
        # e2
        {"center_lat": 10.699237, "center_lon": 122.560077, "north_lat": 10.704237, "south_lat": 10.694237, "east_lon": 122.565077, "west_lon": 122.555077, "risk_class": "EvacCenter_Active", "resolution": "224x224", "timestamp": "2025-07-27 17:01:02", "center_coordinates": "10.699237°, 122.560077°"},
        # e3
        {"center_lat": 11.487269, "center_lon": 122.997432, "north_lat": 11.492269, "south_lat": 11.482269, "east_lon": 123.002432, "west_lon": 122.992432, "risk_class": "EvacCenter_Active", "resolution": "224x224", "timestamp": "2025-07-27 17:01:11", "center_coordinates": "11.487269°, 122.997432°"},
        # e4
        {"center_lat": 10.590347, "center_lon": 123.478711, "north_lat": 10.595347, "south_lat": 10.585347, "east_lon": 123.483711, "west_lon": 123.473711, "risk_class": "EvacCenter_Active", "resolution": "224x224", "timestamp": "2025-07-27 17:02:28", "center_coordinates": "10.590347°, 123.478711°"},
        # e5
        {"center_lat": 11.160740, "center_lon": 125.519250, "north_lat": 11.165740, "south_lat": 11.155740, "east_lon": 125.524250, "west_lon": 125.514250, "risk_class": "EvacCenter_Active", "resolution": "224x224", "timestamp": "2025-07-27 17:02:51", "center_coordinates": "11.160740°, 125.519250°"},
        # e6
        {"center_lat": 10.380003, "center_lon": 123.960141, "north_lat": 10.385003, "south_lat": 10.375003, "east_lon": 123.965141, "west_lon": 123.955141, "risk_class": "EvacCenter_Active", "resolution": "224x224", "timestamp": "2025-07-27 17:04:15", "center_coordinates": "10.380003°, 123.960141°"},
        # e7
        {"center_lat": 10.539796, "center_lon": 122.839460, "north_lat": 10.544796, "south_lat": 10.534796, "east_lon": 122.844460, "west_lon": 122.834460, "risk_class": "EvacCenter_Active", "resolution": "224x224", "timestamp": "2025-07-27 17:04:36", "center_coordinates": "10.539796°, 122.839460°"},
    ]
    evac_folder = "DisasterData/EvacCenter_Active"
    results = []
    print("🚀 Starting batch metadata processing for EvacCenter_Active...")
    print(f"📁 Processing files from: {evac_folder}")
    print("=" * 60)
    for i, metadata in enumerate(metadata_list, 1):
        filename = f"e{i}.png"
        file_path = os.path.join(evac_folder, filename)
        print(f"📄 Processing {filename}...")
        if os.path.exists(file_path):
            output_path, success = add_metadata_to_png(file_path, metadata)
            if success:
                print(f"  ✅ Metadata added: {output_path}")
                class_path, class_success = save_to_class_folder(output_path, "EvacCenter_Active")
                if class_success:
                    print(f"  ✅ Saved to class folder: {class_path}")
                    results.append({"file": filename, "status": "Success", "output": class_path, "coordinates": metadata["center_coordinates"], "timestamp": metadata["timestamp"]})
                else:
                    print(f"  ❌ Error saving to class folder: {class_path}")
                    results.append({"file": filename, "status": "Partial Success", "output": output_path, "coordinates": metadata["center_coordinates"], "timestamp": metadata["timestamp"]})
            else:
                print(f"  ❌ Error adding metadata: {output_path}")
                results.append({"file": filename, "status": "Failed", "output": str(output_path), "coordinates": metadata["center_coordinates"], "timestamp": metadata["timestamp"]})
        else:
            print(f"  ❌ File not found: {file_path}")
            results.append({"file": filename, "status": "File Not Found", "output": "N/A", "coordinates": metadata["center_coordinates"], "timestamp": metadata["timestamp"]})
        print()
    print("=" * 60)
    print("📊 Processing Summary:")
    print("=" * 60)
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