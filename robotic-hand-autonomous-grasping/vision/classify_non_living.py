import cv2
import numpy as np
from typing import Tuple, List, Optional
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

class NonLivingClassifier:
    """Enhanced classifier for non-living objects using advanced computer vision techniques."""
    
    def __init__(self):
        """Initialize the enhanced non-living object classifier."""
        # Define color ranges for common non-living objects (HSV)
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([110, 100, 100], [130, 255, 255]),
            'green': ([50, 100, 100], [70, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'orange': ([10, 100, 100], [20, 255, 255]),
            'purple': ([130, 100, 100], [150, 255, 255]),
            'pink': ([150, 100, 100], [170, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'black': ([0, 0, 0], [180, 255, 30])
        }
        
        # Edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        
        # Texture analysis parameters
        self.texture_threshold = 0.3
        self.lbp_radius = 3
        self.lbp_neighbors = 8
        
        # Shape analysis parameters
        self.min_contour_area = 100
        
        # Machine learning model
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'color_score', 'edge_score', 'texture_score', 'shape_score',
            'symmetry_score', 'regularity_score', 'brightness_score',
            'contrast_score', 'saturation_score'
        ]
        
        # Load pre-trained model if available
        self._load_model()
        
    def _load_model(self):
        """Load pre-trained machine learning model."""
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'non_living_classifier.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("✅ Loaded pre-trained non-living classifier model")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
    
    def classify_object(self, frame: np.ndarray, detection: Tuple[int, int, int, int, float]) -> bool:
        """
        Enhanced classification of non-living objects using multiple features.
        
        Args:
            frame (np.ndarray): Input frame
            detection (Tuple): Detection tuple (x, y, w, h, confidence)
            
        Returns:
            bool: True if object is classified as non-living
        """
        if frame is None:
            return False
            
        x, y, w, h, _ = detection
        
        # Extract object region
        object_region = frame[y:y+h, x:x+w]
        
        if object_region.size == 0:
            return False
        
        # Extract features
        features = self._extract_features(object_region)
        
        # Use machine learning model if available
        if self.model is not None:
            try:
                # Normalize features
                features_normalized = self.scaler.transform([features])
                prediction = self.model.predict(features_normalized)[0]
                probability = self.model.predict_proba(features_normalized)[0][1]
                return prediction == 1 and probability > 0.6
            except Exception as e:
                print(f"⚠️  ML model prediction failed: {e}")
        
        # Fallback to rule-based classification
        return self._rule_based_classification(features)
    
    def _extract_features(self, region: np.ndarray) -> List[float]:
        """
        Extract comprehensive features from object region.
        
        Args:
            region (np.ndarray): Object region
            
        Returns:
            List[float]: Feature vector
        """
        features = []
        
        # Color analysis
        color_score = self._analyze_color_advanced(region)
        features.append(color_score)
        
        # Edge analysis
        edge_score = self._analyze_edges_advanced(region)
        features.append(edge_score)
        
        # Texture analysis
        texture_score = self._analyze_texture_advanced(region)
        features.append(texture_score)
        
        # Shape analysis
        shape_score = self._analyze_shape_advanced(region)
        features.append(shape_score)
        
        # Symmetry analysis
        symmetry_score = self._analyze_symmetry(region)
        features.append(symmetry_score)
        
        # Regularity analysis
        regularity_score = self._analyze_regularity(region)
        features.append(regularity_score)
        
        # Brightness analysis
        brightness_score = self._analyze_brightness(region)
        features.append(brightness_score)
        
        # Contrast analysis
        contrast_score = self._analyze_contrast(region)
        features.append(contrast_score)
        
        # Saturation analysis
        saturation_score = self._analyze_saturation(region)
        features.append(saturation_score)
        
        return features
    
    def _analyze_color_advanced(self, region: np.ndarray) -> float:
        """
        Advanced color analysis for non-living objects.
        
        Args:
            region (np.ndarray): Object region
            
        Returns:
            float: Color analysis score (0-1)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Check for artificial/manufactured colors
        color_scores = []
        
        for color_name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            
            mask = cv2.inRange(hsv, lower, upper)
            color_ratio = np.sum(mask > 0) / mask.size
            color_scores.append(color_ratio)
        
        # Calculate color diversity (non-living objects often have uniform colors)
        hsv_std = np.std(hsv, axis=(0, 1))
        color_diversity = np.mean(hsv_std) / 255.0
        
        # Combine scores
        max_color_score = max(color_scores) if color_scores else 0.0
        return max_color_score * (1 - color_diversity)  # Prefer uniform colors
    
    def _analyze_edges_advanced(self, region: np.ndarray) -> float:
        """
        Advanced edge analysis for non-living objects.
        
        Args:
            region (np.ndarray): Object region
            
        Returns:
            float: Edge analysis score (0-1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges with multiple methods
        edges_canny = cv2.Canny(blurred, self.canny_low, self.canny_high)
        edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
        
        # Calculate edge density
        edge_density_canny = np.sum(edges_canny > 0) / edges_canny.size
        edge_density_sobel = np.sum(edges_sobel > np.mean(edges_sobel)) / edges_sobel.size
        
        # Calculate edge regularity (non-living objects have more regular edges)
        edge_regularity = self._calculate_edge_regularity(edges_canny)
        
        # Combine scores
        edge_score = (edge_density_canny * 0.4 + edge_density_sobel * 0.3 + edge_regularity * 0.3)
        return min(edge_score * 5, 1.0)  # Scale up and cap at 1.0
    
    def _calculate_edge_regularity(self, edges: np.ndarray) -> float:
        """Calculate edge regularity score."""
        if np.sum(edges > 0) == 0:
            return 0.0
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Calculate perimeter regularity
        total_perimeter = 0
        total_area = 0
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            total_perimeter += perimeter
            total_area += area
        
        if total_area == 0:
            return 0.0
        
        # Circularity (closer to 1 = more regular)
        circularity = 4 * np.pi * total_area / (total_perimeter * total_perimeter)
        return min(circularity, 1.0)
    
    def _analyze_texture_advanced(self, region: np.ndarray) -> float:
        """
        Advanced texture analysis using Local Binary Patterns.
        
        Args:
            region (np.ndarray): Object region
            
        Returns:
            float: Texture analysis score (0-1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Calculate Local Binary Pattern
        lbp = self._calculate_lbp(gray)
        
        # Calculate texture features
        texture_uniformity = self._calculate_texture_uniformity(lbp)
        texture_contrast = self._calculate_texture_contrast(gray)
        
        # Combine texture features
        texture_score = (texture_uniformity * 0.6 + texture_contrast * 0.4)
        return min(texture_score, 1.0)
    
    def _calculate_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        height, width = gray.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = gray[i, j]
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j-1], gray[i, j+1],
                    gray[i+1, j-1], gray[i+1, j], gray[i+1, j+1]
                ]
                
                # Calculate LBP
                lbp_value = 0
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        lbp_value += 2**k
                
                lbp[i, j] = lbp_value
        
        return lbp
    
    def _calculate_texture_uniformity(self, lbp: np.ndarray) -> float:
        """Calculate texture uniformity."""
        # Count unique LBP patterns
        unique_patterns = len(np.unique(lbp))
        max_patterns = 256  # 2^8 for 8 neighbors
        
        # Lower number of unique patterns = more uniform texture
        uniformity = 1 - (unique_patterns / max_patterns)
        return uniformity
    
    def _calculate_texture_contrast(self, gray: np.ndarray) -> float:
        """Calculate texture contrast."""
        # Calculate local standard deviation
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        variance = cv2.filter2D((gray.astype(np.float32) - mean)**2, -1, kernel)
        std_dev = np.sqrt(variance)
        
        # Normalize contrast
        contrast = np.mean(std_dev) / 255.0
        return min(contrast * 2, 1.0)
    
    def _analyze_shape_advanced(self, region: np.ndarray) -> float:
        """
        Advanced shape analysis for non-living objects.
        
        Args:
            region (np.ndarray): Object region
            
        Returns:
            float: Shape analysis score (0-1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1
        
        # Convexity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Combine shape features
        shape_score = (circularity * 0.4 + (1 / aspect_ratio) * 0.3 + solidity * 0.3)
        return min(shape_score, 1.0)
    
    def _analyze_symmetry(self, region: np.ndarray) -> float:
        """Analyze object symmetry."""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        height, width = gray.shape
        
        # Calculate horizontal symmetry
        mid_x = width // 2
        left_half = gray[:, :mid_x]
        right_half = gray[:, mid_x:2*mid_x] if 2*mid_x <= width else gray[:, mid_x:]
        
        if left_half.shape == right_half.shape:
            horizontal_symmetry = 1 - np.mean(np.abs(left_half - np.fliplr(right_half))) / 255.0
        else:
            horizontal_symmetry = 0.0
        
        # Calculate vertical symmetry
        mid_y = height // 2
        top_half = gray[:mid_y, :]
        bottom_half = gray[mid_y:2*mid_y, :] if 2*mid_y <= height else gray[mid_y:, :]
        
        if top_half.shape == bottom_half.shape:
            vertical_symmetry = 1 - np.mean(np.abs(top_half - np.flipud(bottom_half))) / 255.0
        else:
            vertical_symmetry = 0.0
        
        # Return average symmetry
        return (horizontal_symmetry + vertical_symmetry) / 2
    
    def _analyze_regularity(self, region: np.ndarray) -> float:
        """Analyze object regularity."""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate gradient uniformity
        gradient_std = np.std(gradient_magnitude)
        gradient_mean = np.mean(gradient_magnitude)
        
        if gradient_mean == 0:
            return 0.0
        
        # Coefficient of variation (lower = more regular)
        cv_gradient = gradient_std / gradient_mean
        regularity = max(0, 1 - cv_gradient)
        
        return min(regularity, 1.0)
    
    def _analyze_brightness(self, region: np.ndarray) -> float:
        """Analyze object brightness."""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean brightness
        brightness = np.mean(gray) / 255.0
        
        # Non-living objects often have consistent brightness
        brightness_std = np.std(gray) / 255.0
        brightness_consistency = max(0, 1 - brightness_std)
        
        return brightness * brightness_consistency
    
    def _analyze_contrast(self, region: np.ndarray) -> float:
        """Analyze object contrast."""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        if max_val == min_val:
            return 0.0
        
        contrast = (max_val - min_val) / 255.0
        return contrast
    
    def _analyze_saturation(self, region: np.ndarray) -> float:
        """Analyze object saturation."""
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Extract saturation channel
        saturation = hsv[:, :, 1]
        
        # Calculate mean saturation
        mean_saturation = np.mean(saturation) / 255.0
        
        return mean_saturation
    
    def _rule_based_classification(self, features: List[float]) -> bool:
        """
        Rule-based classification as fallback.
        
        Args:
            features (List[float]): Feature vector
            
        Returns:
            bool: True if classified as non-living
        """
        if len(features) != len(self.feature_names):
            return False
        
        # Unpack features
        color_score, edge_score, texture_score, shape_score, \
        symmetry_score, regularity_score, brightness_score, \
        contrast_score, saturation_score = features
        
        # Rule-based scoring
        score = 0.0
        
        # Color analysis (non-living objects often have artificial colors)
        if color_score > 0.3:
            score += 0.2
        
        # Edge analysis (non-living objects have sharp, regular edges)
        if edge_score > 0.4:
            score += 0.2
        
        # Texture analysis (non-living objects have uniform textures)
        if texture_score > 0.5:
            score += 0.15
        
        # Shape analysis (non-living objects have regular shapes)
        if shape_score > 0.4:
            score += 0.15
        
        # Symmetry analysis (non-living objects are often symmetric)
        if symmetry_score > 0.6:
            score += 0.1
        
        # Regularity analysis (non-living objects are regular)
        if regularity_score > 0.5:
            score += 0.1
        
        # Brightness analysis (non-living objects have consistent brightness)
        if brightness_score > 0.3:
            score += 0.05
        
        # Contrast analysis (non-living objects often have high contrast)
        if contrast_score > 0.4:
            score += 0.05
        
        # Saturation analysis (non-living objects often have high saturation)
        if saturation_score > 0.4:
            score += 0.05
        
        return score > 0.6
    
    def get_classification_confidence(self, frame: np.ndarray, detection: Tuple[int, int, int, int, float]) -> float:
        """
        Get confidence score for non-living classification.
        
        Args:
            frame (np.ndarray): Input frame
            detection (Tuple): Detection tuple
            
        Returns:
            float: Confidence score (0-1)
        """
        if frame is None:
            return 0.0
            
        x, y, w, h, _ = detection
        object_region = frame[y:y+h, x:x+w]
        
        if object_region.size == 0:
            return 0.0
        
        # Extract features
        features = self._extract_features(object_region)
        
        # Use machine learning model if available
        if self.model is not None:
            try:
                features_normalized = self.scaler.transform([features])
                probability = self.model.predict_proba(features_normalized)[0][1]
                return probability
            except Exception as e:
                print(f"⚠️  ML model confidence failed: {e}")
        
        # Fallback to rule-based confidence
        return sum(features) / len(features) 