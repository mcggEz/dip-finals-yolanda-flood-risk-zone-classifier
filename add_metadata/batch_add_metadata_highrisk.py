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
        # h1
        {"center_lat": 11.830212, "center_lon": 122.092320, "north_lat": 11.835212, "south_lat": 11.825212, "east_lon": 122.097320, "west_lon": 122.087320, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:14:05", "center_coordinates": "11.830212°, 122.092320°"},
        # h2
        {"center_lat": 11.293420, "center_lon": 124.638007, "north_lat": 11.298420, "south_lat": 11.288420, "east_lon": 124.643007, "west_lon": 124.633007, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:14:30", "center_coordinates": "11.293420°, 124.638007°"},
        # h3
        {"center_lat": 10.700714, "center_lon": 122.587785, "north_lat": 10.705714, "south_lat": 10.695714, "east_lon": 122.592785, "west_lon": 122.582785, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:17:25", "center_coordinates": "10.700714°, 122.587785°"},
        # h4
        {"center_lat": 11.547202, "center_lon": 124.330120, "north_lat": 11.552202, "south_lat": 11.542202, "east_lon": 124.335120, "west_lon": 124.325120, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:16:47", "center_coordinates": "11.547202°, 124.330120°"},
        # h5
        {"center_lat": 11.605452, "center_lon": 122.739795, "north_lat": 11.610452, "south_lat": 11.600452, "east_lon": 122.744795, "west_lon": 122.734795, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:19:14", "center_coordinates": "11.605452°, 122.739795°"},
        # h6
        {"center_lat": 12.217603, "center_lon": 123.507229, "north_lat": 12.222603, "south_lat": 12.212603, "east_lon": 123.512229, "west_lon": 123.502229, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:19:44", "center_coordinates": "12.217603°, 123.507229°"},
        # h7
        {"center_lat": 11.593114, "center_lon": 122.832959, "north_lat": 11.598114, "south_lat": 11.588114, "east_lon": 122.837959, "west_lon": 122.827959, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:20:30", "center_coordinates": "11.593114°, 122.832959°"},
        # h8
        {"center_lat": 12.286407, "center_lon": 123.721886, "north_lat": 12.291407, "south_lat": 12.281407, "east_lon": 123.726886, "west_lon": 123.716886, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:21:25", "center_coordinates": "12.286407°, 123.721886°"},
        # h9
        {"center_lat": 11.735550, "center_lon": 122.325895, "north_lat": 11.740550, "south_lat": 11.730550, "east_lon": 122.330895, "west_lon": 122.320895, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:22:18", "center_coordinates": "11.735550°, 122.325895°"},
        # h10
        {"center_lat": 12.507392, "center_lon": 124.633640, "north_lat": 12.512392, "south_lat": 12.502392, "east_lon": 124.638640, "west_lon": 124.628640, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:22:59", "center_coordinates": "12.507392°, 124.633640°"},
        # h11
        {"center_lat": 11.867167, "center_lon": 123.916367, "north_lat": 11.872167, "south_lat": 11.862167, "east_lon": 123.921367, "west_lon": 123.911367, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:23:46", "center_coordinates": "11.867167°, 123.916367°"},
        # h12
        {"center_lat": 12.523501, "center_lon": 125.218161, "north_lat": 12.528501, "south_lat": 12.518501, "east_lon": 125.223161, "west_lon": 125.213161, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:24:39", "center_coordinates": "12.523501°, 125.218161°"},
        # h13
        {"center_lat": 10.529833, "center_lon": 125.163139, "north_lat": 10.534833, "south_lat": 10.524833, "east_lon": 125.168139, "west_lon": 125.158139, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:25:48", "center_coordinates": "10.529833°, 125.163139°"},
        # h14
        {"center_lat": 12.523501, "center_lon": 125.218161, "north_lat": 12.528501, "south_lat": 12.518501, "east_lon": 125.223161, "west_lon": 125.213161, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:24:39", "center_coordinates": "12.523501°, 125.218161°"},
        # h15
        {"center_lat": 12.523501, "center_lon": 125.218161, "north_lat": 12.528501, "south_lat": 12.518501, "east_lon": 125.223161, "west_lon": 125.213161, "risk_class": "HighRisk_Coastal", "resolution": "224x224", "timestamp": "2025-07-27 18:24:39", "center_coordinates": "12.523501°, 125.218161°"},
    ]
    highrisk_folder = "DisasterData/HighRisk_Coastal"
    results = []
    print("🚀 Starting batch metadata processing for HighRisk_Coastal...")
    print(f"📁 Processing files from: {highrisk_folder}")
    print("=" * 60)
    for i, metadata in enumerate(metadata_list, 1):
        filename = f"h{i}.png"
        file_path = os.path.join(highrisk_folder, filename)
        print(f"📄 Processing {filename}...")
        if os.path.exists(file_path):
            output_path, success = add_metadata_to_png(file_path, metadata)
            if success:
                print(f"  ✅ Metadata added: {output_path}")
                class_path, class_success = save_to_class_folder(output_path, "HighRisk_Coastal")
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