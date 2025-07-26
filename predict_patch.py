#!/usr/bin/env python3
"""
Patch Prediction Script
Use this to classify new patch images with the trained CNN
"""

import os
import sys
import numpy as np
import cv2
from cnn_disaster_classification import SimpleCNN

def load_trained_model():
    """Load the trained model weights"""
    weights_file = 'disaster_classifier_weights.npz'
    
    if not os.path.exists(weights_file):
        print(f"❌ Trained model not found: {weights_file}")
        print("Please train the model first using: python train_cnn.py")
        return None
    
    try:
        # Load weights
        weights_data = np.load(weights_file)
        
        # Create model
        model = SimpleCNN(input_shape=(224, 224, 3), num_classes=6)
        
        # Load weights into model
        model.weights['conv1'] = weights_data['conv1_weights']
        model.biases['conv1'] = weights_data['conv1_bias']
        model.weights['conv2'] = weights_data['conv2_weights']
        model.biases['conv2'] = weights_data['conv2_bias']
        model.weights['fc'] = weights_data['fc_weights']
        model.biases['fc'] = weights_data['fc_bias']
        
        print("✅ Trained model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_image(model, image_path):
    """Predict class for a single image"""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = model.forward(img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Class names
        class_names = [
            'HighRisk_Coastal',
            'ModerateRisk_Upland', 
            'SafeZone_UrbanCore',
            'EvacCenter_Active',
            'WarningGap_Barangay',
            'BufferZone_Proposed'
        ]
        
        predicted_class = class_names[predicted_class_idx]
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': predictions[0],
            'class_names': class_names
        }
        
    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        return None

def main():
    """Main prediction function"""
    print("=== Disaster Zone Patch Prediction ===\n")
    
    # Load trained model
    model = load_trained_model()
    if model is None:
        return
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find temp_patch.png
        if os.path.exists('temp_patch.png'):
            image_path = 'temp_patch.png'
        else:
            print("❌ No image specified!")
            print("Usage: python predict_patch.py <image_path>")
            print("Or place an image named 'temp_patch.png' in the current directory")
            return
    
    print(f"🔍 Predicting: {image_path}")
    
    # Make prediction
    result = predict_image(model, image_path)
    
    if result:
        print(f"\n📊 Prediction Results:")
        print(f"   Class: {result['class']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        print(f"\n📈 All Probabilities:")
        for i, (class_name, prob) in enumerate(zip(result['class_names'], result['probabilities'])):
            marker = "🎯" if i == np.argmax(result['probabilities']) else "  "
            print(f"   {marker} {class_name}: {prob:.2%}")
        
        # Generate metadata (similar to patch_selector.py)
        print(f"\n📋 Generated Metadata:")
        print(f"   Hazard Score: {result['confidence']:.2f}")
        print(f"   Predicted Class: {result['class']}")
        
        # Save prediction to file
        import json
        prediction_data = {
            'image_path': image_path,
            'predicted_class': result['class'],
            'confidence': float(result['confidence']),
            'probabilities': {name: float(prob) for name, prob in zip(result['class_names'], result['probabilities'])}
        }
        
        with open('prediction_result.json', 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        print(f"\n💾 Prediction saved to: prediction_result.json")
        
    else:
        print("❌ Prediction failed!")

if __name__ == "__main__":
    main() 