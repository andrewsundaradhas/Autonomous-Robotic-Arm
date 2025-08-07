import cv2
import numpy as np
from typing import List, Tuple, Optional
import time
from collections import deque

class ObjectDetector:
    """Enhanced object detection class using multiple algorithms for robotic hand vision."""
    
    def __init__(self):
        """Initialize the enhanced object detector."""
        # Detection parameters
        self.min_area = 800  # Minimum object area to detect
        self.max_area = 60000  # Maximum object area to detect
        self.detection_threshold = 0.4
        self.confidence_threshold = 0.6
        
        # Background subtraction with multiple methods
        self.bg_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=300, 
            varThreshold=40, 
            detectShadows=False
        )
        
        # KNN background subtractor for better performance
        self.bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
            history=500,
            dist2Threshold=400.0,
            detectShadows=False
        )
        
        # Motion detection parameters
        self.motion_threshold = 25
        self.prev_frame = None
        
        # Object tracking with Kalman filter
        self.tracked_objects = {}
        self.object_id_counter = 0
        self.tracking_history = deque(maxlen=30)
        
        # Edge detection parameters
        self.canny_low = 30
        self.canny_high = 100
        
        # Morphological operations
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Color-based detection
        self.color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'yellow': ([15, 50, 50], [35, 255, 255]),
            'orange': ([5, 50, 50], [15, 255, 255])
        }
        
    def detect_objects(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Enhanced object detection using multiple algorithms.
        
        Args:
            frame (np.ndarray): Input frame from camera
            
        Returns:
            List[Tuple[int, int, int, int, float]]: List of (x, y, w, h, confidence) for detected objects
        """
        if frame is None:
            return []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multiple detection methods
        detections_motion = self._detect_motion_based(gray)
        detections_background = self._detect_background_subtraction(gray)
        detections_edge = self._detect_edge_based(gray)
        detections_color = self._detect_color_based(frame)
        
        # Combine detections using voting system
        all_detections = detections_motion + detections_background + detections_edge + detections_color
        combined_detections = self._combine_detections(all_detections)
        
        # Filter and validate detections
        valid_detections = self._filter_detections(combined_detections, frame.shape)
        
        # Update tracking
        self._update_tracking(valid_detections)
        
        return valid_detections
    
    def _detect_motion_based(self, gray: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect objects using frame differencing."""
        detections = []
        
        if self.prev_frame is not None:
            # Calculate frame difference
            frame_diff = cv2.absdiff(self.prev_frame, gray)
            
            # Apply threshold
            _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel_open)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(area / self.max_area, 1.0)
                    detections.append((x, y, w, h, confidence * 0.8))
        
        self.prev_frame = gray.copy()
        return detections
    
    def _detect_background_subtraction(self, gray: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect objects using background subtraction."""
        detections = []
        
        # Apply MOG2 background subtraction
        fg_mask_mog2 = self.bg_subtractor_mog2.apply(gray)
        
        # Apply KNN background subtraction
        fg_mask_knn = self.bg_subtractor_knn.apply(gray)
        
        # Combine masks
        fg_mask = cv2.bitwise_or(fg_mask_mog2, fg_mask_knn)
        
        # Apply morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(area / self.max_area, 1.0)
                detections.append((x, y, w, h, confidence * 0.9))
        
        return detections
    
    def _detect_edge_based(self, gray: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect objects using edge detection."""
        detections = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Apply morphological operations
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate edge density as confidence
                roi = edges[y:y+h, x:x+w]
                edge_density = np.sum(roi > 0) / roi.size if roi.size > 0 else 0
                confidence = min(edge_density * 2, 1.0)
                
                detections.append((x, y, w, h, confidence * 0.7))
        
        return detections
    
    def _detect_color_based(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect objects using color-based detection."""
        detections = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for color_name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            
            # Create mask for color
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphological operations
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(area / self.max_area, 1.0)
                    detections.append((x, y, w, h, confidence * 0.6))
        
        return detections
    
    def _combine_detections(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, float]]:
        """Combine multiple detection results using non-maximum suppression."""
        if not detections:
            return []
        
        # Convert to numpy array for easier processing
        boxes = np.array([(x, y, x + w, y + h, conf) for x, y, w, h, conf in detections])
        
        if len(boxes) == 0:
            return []
        
        # Sort by confidence
        indices = np.argsort(boxes[:, 4])[::-1]
        
        keep = []
        while len(indices) > 0:
            # Pick the highest confidence detection
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[indices[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[indices[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[indices[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[indices[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = w * h
            
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (boxes[indices[1:], 2] - boxes[indices[1:], 0]) * (boxes[indices[1:], 3] - boxes[indices[1:], 1])
            union = area1 + area2 - overlap
            
            iou = overlap / union
            indices = indices[np.where(iou < 0.5)[0] + 1]
        
        # Convert back to original format
        combined = []
        for i in keep:
            x1, y1, x2, y2, conf = boxes[i]
            combined.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(conf)))
        
        return combined
    
    def _filter_detections(self, detections: List[Tuple[int, int, int, int, float]], frame_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int, float]]:
        """Filter detections based on various criteria."""
        filtered = []
        height, width = frame_shape[:2]
        
        for x, y, w, h, confidence in detections:
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Check area constraints
            area = w * h
            if area < self.min_area or area > self.max_area:
                continue
            
            # Check aspect ratio (avoid very thin or very wide objects)
            aspect_ratio = w / h if h > 0 else 1
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            # Check if detection is within frame bounds
            if x < 0 or y < 0 or x + w > width or y + h > height:
                continue
            
            # Check minimum size
            if w < 20 or h < 20:
                continue
            
            filtered.append((x, y, w, h, confidence))
        
        return filtered
    
    def _update_tracking(self, detections: List[Tuple[int, int, int, int, float]]):
        """Update object tracking."""
        current_time = time.time()
        
        # Simple tracking based on proximity
        for detection in detections:
            x, y, w, h, conf = detection
            center = (x + w // 2, y + h // 2)
            
            # Find closest tracked object
            min_distance = float('inf')
            best_match = None
            
            for obj_id, obj_data in self.tracked_objects.items():
                last_center = obj_data['center']
                distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                
                if distance < min_distance and distance < 100:  # Max tracking distance
                    min_distance = distance
                    best_match = obj_id
            
            if best_match is not None:
                # Update existing track
                self.tracked_objects[best_match].update({
                    'center': center,
                    'bbox': (x, y, w, h),
                    'confidence': conf,
                    'last_seen': current_time
                })
            else:
                # Create new track
                self.object_id_counter += 1
                self.tracked_objects[self.object_id_counter] = {
                    'center': center,
                    'bbox': (x, y, w, h),
                    'confidence': conf,
                    'last_seen': current_time
                }
        
        # Remove old tracks
        current_time = time.time()
        to_remove = []
        for obj_id, obj_data in self.tracked_objects.items():
            if current_time - obj_data['last_seen'] > 2.0:  # Remove after 2 seconds
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Draw detection boxes on the frame with enhanced visualization.
        
        Args:
            frame (np.ndarray): Input frame
            detections (List[Tuple]): List of detected objects
            
        Returns:
            np.ndarray: Frame with detection boxes drawn
        """
        result_frame = frame.copy()
        
        for i, (x, y, w, h, confidence) in enumerate(detections):
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence text
            text = f"{confidence:.2f}"
            cv2.putText(result_frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw object ID if tracked
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data['bbox'] == (x, y, w, h):
                    cv2.putText(result_frame, f"ID:{obj_id}", (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    break
        
        return result_frame
    
    def get_object_center(self, detection: Tuple[int, int, int, int, float]) -> Tuple[int, int]:
        """
        Get the center point of a detected object.
        
        Args:
            detection (Tuple): Detection tuple (x, y, w, h, confidence)
            
        Returns:
            Tuple[int, int]: Center point (x, y)
        """
        x, y, w, h, _ = detection
        center_x = x + w // 2
        center_y = y + h // 2
        return (center_x, center_y)
    
    def is_object_in_grasp_zone(self, detection: Tuple[int, int, int, int, float], 
                                frame_size: Tuple[int, int]) -> bool:
        """
        Check if object is in the grasp zone (center area of frame).
        
        Args:
            detection (Tuple): Detection tuple
            frame_size (Tuple[int, int]): Frame dimensions (width, height)
            
        Returns:
            bool: True if object is in grasp zone
        """
        x, y, w, h, _ = detection
        frame_width, frame_height = frame_size
        
        # Define grasp zone (center 25% of frame)
        grasp_zone_x = frame_width * 0.375
        grasp_zone_y = frame_height * 0.375
        grasp_zone_w = frame_width * 0.25
        grasp_zone_h = frame_height * 0.25
        
        # Check if object center is in grasp zone
        center_x, center_y = self.get_object_center(detection)
        
        return (grasp_zone_x <= center_x <= grasp_zone_x + grasp_zone_w and
                grasp_zone_y <= center_y <= grasp_zone_y + grasp_zone_h)
    
    def get_best_grasp_candidate(self, detections: List[Tuple[int, int, int, int, float]], 
                                 frame_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Get the best candidate for grasping based on multiple criteria.
        
        Args:
            detections: List of detections
            frame_size: Frame dimensions
            
        Returns:
            Optional[Tuple]: Best grasp candidate or None
        """
        if not detections:
            return None
        
        candidates = []
        for detection in detections:
            x, y, w, h, confidence = detection
            
            # Check if in grasp zone
            if not self.is_object_in_grasp_zone(detection, frame_size):
                continue
            
            # Calculate score based on multiple factors
            area = w * h
            area_score = min(area / (frame_size[0] * frame_size[1] * 0.1), 1.0)  # Normalize area
            
            # Distance from center
            center_x, center_y = self.get_object_center(detection)
            frame_center_x, frame_center_y = frame_size[0] // 2, frame_size[1] // 2
            distance = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
            distance_score = max(0, 1 - distance / (frame_size[0] * 0.5))
            
            # Combined score
            total_score = (confidence * 0.4 + area_score * 0.3 + distance_score * 0.3)
            candidates.append((detection, total_score))
        
        if not candidates:
            return None
        
        # Return the best candidate
        best_candidate = max(candidates, key=lambda x: x[1])
        return best_candidate[0] 