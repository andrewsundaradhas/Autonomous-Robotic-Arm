import cv2
import numpy as np
from typing import Optional, Tuple

class Camera:
    """Camera utility class for webcam handling and frame processing."""
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize camera with specified index.
        
        Args:
            camera_index (int): Index of the camera device (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize the camera capture.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera at index {self.camera_index}")
                return False
            self.is_initialized = True
            print(f"Camera initialized successfully at index {self.camera_index}")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the camera.
        
        Returns:
            Optional[np.ndarray]: Frame as numpy array, or None if failed
        """
        if not self.is_initialized or self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            return None
            
        return frame
    
    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """
        Get the current frame size.
        
        Returns:
            Optional[Tuple[int, int]]: (width, height) or None if not initialized
        """
        if not self.is_initialized or self.cap is None:
            return None
            
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def release(self):
        """Release the camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_initialized = False
            print("Camera released")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release() 