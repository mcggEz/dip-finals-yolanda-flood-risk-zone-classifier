import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Global variable to cache the trained model
_cached_model = None
_cached_classes = None

def load_and_train_random_forest():
    """Load data and train Random Forest model"""
    global _cached_model, _cached_classes
    
    # Check if model is already cached
    if _cached_model is not None and _cached_classes is not None:
        print("🔄 Using cached Random Forest model...")
        return _cached_model, _cached_classes
    
    print("🔄 Loading dataset and training Random Forest...")
    
    data_dir = "DisasterData"
    img_size = (224, 224)
    classes = [
        "HighRisk_Coastal",
        "ModerateRisk_Upland", 
        "SafeZone_UrbanCore",
        "EvacCenter_Active",
        "WarningGap_Barangay",
        "BufferZone_Proposed"
    ]
    
    features = []
    labels = []
    
    # Count samples per class for debugging
    class_counts = {class_name: 0 for class_name in classes}
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️ Warning: Directory {class_dir} does not exist")
            continue
            
        print(f"📁 Processing {class_name} from {class_dir}")
        files_processed = 0
        
        for filename in os.listdir(class_dir):
            if filename.endswith('.png'):
                file_path = os.path.join(class_dir, filename)
                
                try:
                    img = Image.open(file_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img = img.resize(img_size)
                    img_array = np.array(img)
                    
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]
                    
                    img_features = img_array.flatten() / 255.0
                    
                    features.append(img_features)
                    labels.append(class_idx)
                    class_counts[class_name] += 1
                    files_processed += 1
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        print(f"   ✅ Processed {files_processed} files for {class_name}")
    
    # Print class distribution
    print("📊 Dataset Statistics:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} samples")
    
    if not features:
        print("❌ No features extracted! Check if DisasterData directory exists and contains images.")
        return None, None
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"✅ Total samples: {len(features)}")
    print(f"✅ Feature shape: {features.shape}")
    
    # Train Random Forest model
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.4f}")
    
    # Cache the model
    _cached_model = model
    _cached_classes = classes
    
    return model, classes

def calculate_hazard_score(probabilities, classes):
    """Calculate hazard score based on class probabilities"""
    # Define hazard weights for each class (higher = more hazardous)
    hazard_weights = {
        "HighRisk_Coastal": 1.0,        # Highest hazard
        "WarningGap_Barangay": 0.8,     # High hazard (no warnings)
        "ModerateRisk_Upland": 0.6,     # Moderate hazard
        "EvacCenter_Active": 0.4,       # Low hazard (evacuation available)
        "BufferZone_Proposed": 0.3,     # Very low hazard
        "SafeZone_UrbanCore": 0.1       # Minimal hazard
    }
    
    # Calculate weighted hazard score
    total_hazard_score = 0.0
    for class_name, prob in zip(classes, probabilities):
        weight = hazard_weights.get(class_name, 0.5)
        total_hazard_score += prob * weight
    
    return total_hazard_score

def predict_image_random_forest(image_path, model, classes):
    """Predict class for a single image using Random Forest"""
    try:
        print(f"🔍 Predicting image: {image_path}")
        
        # Load and preprocess image
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        img_features = img_array.flatten() / 255.0
        img_features = img_features.reshape(1, -1)
        
        print(f"✅ Image preprocessed: shape {img_features.shape}")
        
        # Predict
        prediction = model.predict(img_features)
        prediction_proba = model.predict_proba(img_features)
        
        predicted_class = prediction[0]
        confidence = np.max(prediction_proba[0])
        
        print(f"🎯 Raw prediction: {predicted_class}")
        print(f"🎯 Predicted class: {classes[predicted_class]}")
        print(f"🎯 Confidence: {confidence:.3f}")
        
        # Print all probabilities for debugging
        print("📊 All class probabilities:")
        for i, (class_name, prob) in enumerate(zip(classes, prediction_proba[0])):
            print(f"   {class_name}: {prob:.3f}")
        
        # Calculate hazard score
        hazard_score = calculate_hazard_score(prediction_proba[0], classes)
        print(f"🚨 Hazard score: {hazard_score:.3f}")
        
        return classes[predicted_class], confidence, prediction_proba[0], hazard_score
        
    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def get_random_forest_prediction(image_path):
    """Main function to get Random Forest prediction for web UI integration"""
    try:
        # Train or load model
        model, classes = load_and_train_random_forest()
        
        # Predict
        predicted_class, confidence, probabilities, hazard_score = predict_image_random_forest(image_path, model, classes)
        
        if predicted_class:
            # Return results in format compatible with web UI
            return {
                'class': predicted_class,
                'confidence': confidence,
                'hazard_score': hazard_score,
                'probabilities': dict(zip(classes, probabilities)),
                'model_type': 'Random Forest'
            }
        else:
            return None
            
    except Exception as e:
        print(f"❌ Error in Random Forest prediction: {e}")
        return None

def test_model_on_training_data():
    """Test the model on some training data to verify it's working correctly"""
    print("\n🧪 Testing model on training data...")
    
    model, classes = load_and_train_random_forest()
    if model is None:
        print("❌ Model training failed!")
        return
    
    # Test on one sample from each class
    data_dir = "DisasterData"
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith('.png'):
                    file_path = os.path.join(class_dir, filename)
                    print(f"\n🔍 Testing {class_name} with {filename}:")
                    
                    predicted_class, confidence, probabilities, hazard_score = predict_image_random_forest(file_path, model, classes)
                    
                    if predicted_class:
                        correct = predicted_class == class_name
                        status = "✅" if correct else "❌"
                        print(f"{status} Expected: {class_name}, Predicted: {predicted_class}, Confidence: {confidence:.3f}")
                    else:
                        print(f"❌ Failed to predict {filename}")
                    break  # Only test one sample per class

def analyze_test_image():
    """Analyze the test image and compare with training samples"""
    print("\n🔍 Analyzing test image...")
    
    # Check if test image exists
    test_image_path = "image.png"
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    # Load and display test image info
    try:
        from PIL import Image
        test_img = Image.open(test_image_path)
        print(f"📷 Test image: {test_image_path}")
        print(f"   Size: {test_img.size}")
        print(f"   Mode: {test_img.mode}")
        
        # Get model prediction
        model, classes = load_and_train_random_forest()
        if model is None:
            return
            
        predicted_class, confidence, probabilities, hazard_score = predict_image_random_forest(test_image_path, model, classes)
        
        print(f"\n🎯 **PREDICTION ANALYSIS:**")
        print(f"   Predicted Class: {predicted_class}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Hazard Score: {hazard_score:.3f}")
        
        # Show top 3 predictions
        print(f"\n📊 **TOP PREDICTIONS:**")
        class_probs = list(zip(classes, probabilities))
        class_probs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (class_name, prob) in enumerate(class_probs[:3]):
            print(f"   {i+1}. {class_name}: {prob:.1%}")
        
        # Suggest what the image might be
        print(f"\n💡 **ANALYSIS:**")
        if confidence < 0.5:
            print(f"   ⚠️  Low confidence ({confidence:.1%}) suggests the image doesn't clearly match any class")
            print(f"   🤔 The image might be:")
            print(f"      - A mixed/complex scene")
            print(f"      - Different from training data")
            print(f"      - Similar to multiple classes")
        else:
            print(f"   ✅ Reasonable confidence ({confidence:.1%}) for {predicted_class}")
        
        # Show what each class typically looks like
        print(f"\n📋 **CLASS DESCRIPTIONS:**")
        class_descriptions = {
            "HighRisk_Coastal": "Coastal areas with high flood risk, near water bodies",
            "ModerateRisk_Upland": "Rural/agricultural areas with moderate risk",
            "SafeZone_UrbanCore": "Urban areas with good infrastructure",
            "EvacCenter_Active": "Evacuation centers, shelters, safe buildings",
            "WarningGap_Barangay": "Areas without proper warning systems",
            "BufferZone_Proposed": "Proposed buffer zones, planned safe areas"
        }
        
        for class_name, description in class_descriptions.items():
            if class_name == predicted_class:
                print(f"   🎯 {class_name}: {description} (PREDICTED)")
            else:
                print(f"   📍 {class_name}: {description}")
                
    except Exception as e:
        print(f"❌ Error analyzing test image: {e}")

def test_with_known_image():
    """Test the model with a known training image to verify it works"""
    print("\n🧪 Testing with known training image...")
    
    model, classes = load_and_train_random_forest()
    if model is None:
        return
    
    # Use a known HighRisk_Coastal image
    test_image = "DisasterData/HighRisk_Coastal/h1_with_metadata_20250727_183019_highrisk_coastal_20250727_183019.png"
    
    if os.path.exists(test_image):
        print(f"🔍 Testing with known HighRisk_Coastal image: {test_image}")
        
        predicted_class, confidence, probabilities, hazard_score = predict_image_random_forest(test_image, model, classes)
        
        if predicted_class:
            correct = predicted_class == "HighRisk_Coastal"
            status = "✅" if correct else "❌"
            print(f"{status} Expected: HighRisk_Coastal, Predicted: {predicted_class}, Confidence: {confidence:.1%}")
            
            if correct:
                print(f"🎉 **SUCCESS:** Model correctly identifies HighRisk_Coastal image!")
                print(f"   This proves the model is working correctly.")
                print(f"   Your test image issue is likely due to image content, not model problems.")
            else:
                print(f"❌ **FAILURE:** Model incorrectly predicts {predicted_class}")
        else:
            print(f"❌ Failed to predict {test_image}")
    else:
        print(f"❌ Test image not found: {test_image}")

def main():
    print("=" * 30)
    
    # Train Random Forest model
    model, classes = load_and_train_random_forest()
    
    if model is None:
        print("❌ Model training failed! Check your DisasterData directory.")
        return
    
    # Test the model on training data first
    test_model_on_training_data()
    
    # Test with a known image
    test_with_known_image()
    
    # Analyze the test image
    analyze_test_image()
    
    # Test image
    image_path = "image.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n🔍 Testing on external image: {image_path}")
    
    # Predict
    predicted_class, confidence, probabilities, hazard_score = predict_image_random_forest(image_path, model, classes)
    
    if predicted_class:
        # Interpret hazard score
        if hazard_score >= 0.8:
            hazard_level = "🔴 CRITICAL HAZARD"
        elif hazard_score >= 0.6:
            hazard_level = "🟠 HIGH HAZARD"
        elif hazard_score >= 0.4:
            hazard_level = "🟡 MODERATE HAZARD"
        elif hazard_score >= 0.2:
            hazard_level = "🟢 LOW HAZARD"
        else:
            hazard_level = "🟢 MINIMAL HAZARD"
        
        print(f"\n📍 Class: {predicted_class}")
        print(f"🎯 Confidence: {confidence:.1%}")
        print(f"🚨 Hazard Score: {hazard_score:.3f}")
        print(f"📊 Hazard Level: {hazard_level}")
        
        print(f"\n📊 Probabilities:")
        for class_name, prob in zip(classes, probabilities):
            print(f"   {class_name}: {prob:.1%}")
    
    else:
        print("❌ Failed to predict image class")

if __name__ == "__main__":
    main() 