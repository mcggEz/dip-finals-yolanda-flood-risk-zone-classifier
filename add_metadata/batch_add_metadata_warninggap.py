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
        # w1
        {"center_lat": 11.584811, "center_lon": 122.755441, "north_lat": 11.589811, "south_lat": 11.579811, "east_lon": 122.760441, "west_lon": 122.750441, "risk_class": "WarningGap_Barangay", "resolution": "224x224", "timestamp": "2025-07-27 17:11:56", "center_coordinates": "11.584811°, 122.755441°"},
        # w2
        {"center_lat": 11.581237, "center_lon": 122.788055, "north_lat": 11.586237, "south_lat": 11.576237, "east_lon": 122.793055, "west_lon": 122.783055, "risk_class": "WarningGap_Barangay", "resolution": "224x224", "timestamp": "2025-07-27 17:12:26", "center_coordinates": "11.581237°, 122.788055°"},
        # w3
        {"center_lat": 11.472959, "center_lon": 123.086702, "north_lat": 11.477959, "south_lat": 11.467959, "east_lon": 123.091702, "west_lon": 123.081702, "risk_class": "WarningGap_Barangay", "resolution": "224x224", "timestamp": "2025-07-27 17:13:11", "center_coordinates": "11.472959°, 123.086702°"},
        # w4
        {"center_lat": 11.706031, "center_lon": 122.371997, "north_lat": 11.711031, "south_lat": 11.701031, "east_lon": 122.376997, "west_lon": 122.366997, "risk_class": "WarningGap_Barangay", "resolution": "224x224", "timestamp": "2025-07-27 17:14:01", "center_coordinates": "11.706031°, 122.371997°"},
        # w5
        {"center_lat": 12.355932, "center_lon": 121.067282, "north_lat": 12.360932, "south_lat": 12.350932, "east_lon": 121.072282, "west_lon": 121.062282, "risk_class": "WarningGap_Barangay", "resolution": "224x224", "timestamp": "2025-07-27 17:15:24", "center_coordinates": "12.355932°, 121.067282°"},
        # w6
        {"center_lat": 10.951399, "center_lon": 125.028665, "north_lat": 10.956399, "south_lat": 10.946399, "east_lon": 125.033665, "west_lon": 125.023665, "risk_class": "WarningGap_Barangay", "resolution": "224x224", "timestamp": "2025-07-27 17:15:34", "center_coordinates": "10.951399°, 125.028665°"},
        # w7
        {"center_lat": 10.891405, "center_lon": 123.416614, "north_lat": 10.896405, "south_lat": 10.886405, "east_lon": 123.421614, "west_lon": 123.411614, "risk_class": "WarningGap_Barangay", "resolution": "224x224", "timestamp": "2025-07-27 17:16:50", "center_coordinates": "10.891405°, 123.416614°"},
        # w8
        {"center_lat": 11.063401, "center_lon": 125.034717, "north_lat": 11.068401, "south_lat": 11.058401, "east_lon": 125.039717, "west_lon": 125.029717, "risk_class": "WarningGap_Barangay", "resolution": "224x224", "timestamp": "2025-07-27 17:19:45", "center_coordinates": "11.063401°, 125.034717°"},
    ]
    warninggap_folder = "DisasterData/WarningGap_Barangay"
    results = []
    print("🚀 Starting batch metadata processing for WarningGap_Barangay...")
    print(f"📁 Processing files from: {warninggap_folder}")
    print("=" * 60)
    for i, metadata in enumerate(metadata_list, 1):
        filename = f"w{i}.png"
        file_path = os.path.join(warninggap_folder, filename)
        print(f"📄 Processing {filename}...")
        if os.path.exists(file_path):
            output_path, success = add_metadata_to_png(file_path, metadata)
            if success:
                print(f"  ✅ Metadata added: {output_path}")
                class_path, class_success = save_to_class_folder(output_path, "WarningGap_Barangay")
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