#!/usr/bin/env python3
"""
🤖 Training Script for Non-Living Object Classifier

This script trains a machine learning model to classify non-living objects
using the enhanced feature extraction from the classifier.
"""

import cv2
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from typing import List, Tuple
import random

# Import our classifier
from vision.classify_non_living import NonLivingClassifier

def generate_synthetic_data(num_samples: int = 1000) -> Tuple[List[List[float]], List[int]]:
    """
    Generate synthetic training data for non-living vs living objects.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Tuple[List[List[float]], List[int]]: Features and labels
    """
    features = []
    labels = []
    
    # Non-living objects (label 1)
    for _ in range(num_samples // 2):
        # Generate features typical of non-living objects
        color_score = random.uniform(0.3, 1.0)  # High color scores
        edge_score = random.uniform(0.4, 1.0)   # Sharp edges
        texture_score = random.uniform(0.5, 1.0) # Uniform texture
        shape_score = random.uniform(0.4, 1.0)   # Regular shapes
        symmetry_score = random.uniform(0.6, 1.0) # High symmetry
        regularity_score = random.uniform(0.5, 1.0) # Regular patterns
        brightness_score = random.uniform(0.3, 1.0) # Consistent brightness
        contrast_score = random.uniform(0.4, 1.0)   # High contrast
        saturation_score = random.uniform(0.4, 1.0) # High saturation
        
        feature_vector = [
            color_score, edge_score, texture_score, shape_score,
            symmetry_score, regularity_score, brightness_score,
            contrast_score, saturation_score
        ]
        
        features.append(feature_vector)
        labels.append(1)  # Non-living
    
    # Living objects (label 0)
    for _ in range(num_samples // 2):
        # Generate features typical of living objects
        color_score = random.uniform(0.0, 0.4)  # Lower color scores
        edge_score = random.uniform(0.0, 0.3)   # Softer edges
        texture_score = random.uniform(0.0, 0.4) # Irregular texture
        shape_score = random.uniform(0.0, 0.3)   # Irregular shapes
        symmetry_score = random.uniform(0.0, 0.4) # Lower symmetry
        regularity_score = random.uniform(0.0, 0.4) # Irregular patterns
        brightness_score = random.uniform(0.2, 0.6) # Variable brightness
        contrast_score = random.uniform(0.1, 0.4)   # Lower contrast
        saturation_score = random.uniform(0.1, 0.4) # Lower saturation
        
        feature_vector = [
            color_score, edge_score, texture_score, shape_score,
            symmetry_score, regularity_score, brightness_score,
            contrast_score, saturation_score
        ]
        
        features.append(feature_vector)
        labels.append(0)  # Living
    
    return features, labels

def train_classifier(features: List[List[float]], labels: List[int], 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Train the Random Forest classifier.
    
    Args:
        features: Feature vectors
        labels: Target labels
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Tuple[RandomForestClassifier, StandardScaler]: Trained model and scaler
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    
    print("📊 Model Performance:")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=['Living', 'Non-living']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_names = [
        'color_score', 'edge_score', 'texture_score', 'shape_score',
        'symmetry_score', 'regularity_score', 'brightness_score',
        'contrast_score', 'saturation_score'
    ]
    
    print("\nFeature Importance:")
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"  {name}: {importance:.3f}")
    
    return model, scaler

def save_model(model: RandomForestClassifier, scaler: StandardScaler, 
               model_path: str = "model/non_living_classifier.pkl"):
    """
    Save the trained model and scaler.
    
    Args:
        model: Trained Random Forest model
        scaler: Fitted StandardScaler
        model_path: Path to save the model
    """
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': [
            'color_score', 'edge_score', 'texture_score', 'shape_score',
            'symmetry_score', 'regularity_score', 'brightness_score',
            'contrast_score', 'saturation_score'
        ]
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved to {model_path}")

def test_classifier_on_images(classifier: NonLivingClassifier, test_images: List[str]):
    """
    Test the classifier on real images.
    
    Args:
        classifier: NonLivingClassifier instance
        test_images: List of image paths to test
    """
    print("\n🧪 Testing classifier on images:")
    print("=" * 50)
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            continue
        
        # Create a dummy detection (entire image)
        h, w = image.shape[:2]
        detection = (0, 0, w, h, 1.0)
        
        # Classify
        is_non_living = classifier.classify_object(image, detection)
        confidence = classifier.get_classification_confidence(image, detection)
        
        result = "Non-living" if is_non_living else "Living"
        print(f"📸 {os.path.basename(image_path)}: {result} (confidence: {confidence:.3f})")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train non-living object classifier")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--model-path", type=str, default="model/non_living_classifier.pkl", 
                       help="Path to save model")
    parser.add_argument("--test-images", nargs="*", help="Images to test classifier on")
    
    args = parser.parse_args()
    
    print("🤖 Training Non-Living Object Classifier")
    print("=" * 50)
    
    # Generate synthetic data
    print(f"📊 Generating {args.samples} synthetic samples...")
    features, labels = generate_synthetic_data(args.samples)
    
    # Train classifier
    print("🎯 Training Random Forest classifier...")
    model, scaler = train_classifier(features, labels, args.test_size, args.random_state)
    
    # Save model
    print("💾 Saving model...")
    save_model(model, scaler, args.model_path)
    
    # Test on real images if provided
    if args.test_images:
        classifier = NonLivingClassifier()
        test_classifier_on_images(classifier, args.test_images)
    
    print("\n✅ Training complete!")
    print("💡 The model will be automatically loaded by the classifier next time you run the system.")

if __name__ == "__main__":
    main() 