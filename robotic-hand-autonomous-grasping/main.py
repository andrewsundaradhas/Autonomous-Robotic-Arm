#!/usr/bin/env python3
"""
🤖 Autonomous Robotic Hand - Enhanced Main Application

This script orchestrates the complete autonomous robotic hand system,
combining advanced computer vision for object detection and Arduino control for grasping.
"""

import cv2
import time
import sys
import signal
import numpy as np
from typing import Optional, Tuple, List
from collections import deque
import threading

# Import our custom modules
from utils.camera import Camera
from vision.object_detection import ObjectDetector
from vision.classify_non_living import NonLivingClassifier
from control.serial_comm import ArduinoController

class AutonomousRoboticHand:
    """Enhanced main class for autonomous robotic hand system."""
    
    def __init__(self, arduino_port: str = "COM3", camera_index: int = 0):
        """
        Initialize the enhanced autonomous robotic hand system.
        
        Args:
            arduino_port (str): Serial port for Arduino connection
            camera_index (int): Camera device index
        """
        self.arduino_port = arduino_port
        self.camera_index = camera_index
        
        # Initialize components
        self.camera = Camera(camera_index)
        self.object_detector = ObjectDetector()
        self.classifier = NonLivingClassifier()
        self.arduino = ArduinoController(port=arduino_port)
        
        # System state
        self.running = False
        self.grasp_mode = False
        self.last_grasp_time = 0
        self.grasp_cooldown = 2.0  # Reduced cooldown for better responsiveness
        
        # Enhanced grasp decision making
        self.grasp_candidates = deque(maxlen=10)  # Track recent candidates
        self.grasp_confidence_threshold = 0.7
        self.min_grasp_duration = 1.0  # Minimum time object must be in grasp zone
        
        # Display settings
        self.show_detections = True
        self.show_grasp_zone = True
        self.show_debug_info = True
        
        # Statistics and monitoring
        self.frames_processed = 0
        self.objects_detected = 0
        self.grasp_attempts = 0
        self.successful_grasps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Performance monitoring
        self.processing_times = deque(maxlen=30)
        self.detection_history = deque(maxlen=50)
        
        # Error handling
        self.error_count = 0
        self.max_errors = 10
        self.last_error_time = 0
        
    def initialize_system(self) -> bool:
        """
        Initialize all system components with enhanced error handling.
        
        Returns:
            bool: True if all components initialized successfully
        """
        print("🤖 Initializing Enhanced Autonomous Robotic Hand System...")
        
        try:
            # Initialize camera with retry mechanism
            camera_initialized = False
            for attempt in range(3):
                if self.camera.initialize():
                    camera_initialized = True
                    break
                print(f"⚠️  Camera initialization attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
            
            if not camera_initialized:
                print("❌ Failed to initialize camera after 3 attempts")
                return False
            
            # Initialize Arduino connection with enhanced error handling
            arduino_initialized = False
            for attempt in range(3):
                if self.arduino.connect():
                    arduino_initialized = True
                    break
                print(f"⚠️  Arduino connection attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
            
            if not arduino_initialized:
                print("❌ Failed to connect to Arduino after 3 attempts")
                print("💡 Make sure Arduino is connected and gripper_control.ino is uploaded")
                return False
            
            # Calibrate gripper with timeout
            print("🔧 Calibrating gripper...")
            calibration_start = time.time()
            calibration_success = False
            
            while time.time() - calibration_start < 10:  # 10 second timeout
                if self.arduino.calibrate_gripper():
                    calibration_success = True
                    break
                time.sleep(0.5)
            
            if not calibration_success:
                print("⚠️  Gripper calibration failed, continuing anyway...")
            
            print("✅ System initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ System initialization error: {e}")
            return False
    
    def process_frame(self, frame) -> Tuple[List[Tuple], Optional[Tuple], bool]:
        """
        Enhanced frame processing with better object detection and classification.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Tuple[List[Tuple], Optional[Tuple], bool]: (detections, best_candidate, should_grasp)
        """
        start_time = time.time()
        
        try:
            # Detect objects using enhanced detector
            detections = self.object_detector.detect_objects(frame)
            
            if not detections:
                self._update_performance_metrics(start_time, [])
                return [], None, False
            
            # Classify detections and find grasp candidates
            grasp_candidates = []
            frame_size = (frame.shape[1], frame.shape[0])
            
            for detection in detections:
                # Classify as non-living
                is_non_living = self.classifier.classify_object(frame, detection)
                classification_confidence = self.classifier.get_classification_confidence(frame, detection)
                
                if is_non_living and classification_confidence > self.grasp_confidence_threshold:
                    # Check if in grasp zone
                    if self.object_detector.is_object_in_grasp_zone(detection, frame_size):
                        grasp_candidates.append((detection, classification_confidence))
            
            # Get best grasp candidate
            best_candidate = None
            should_grasp = False
            
            if grasp_candidates:
                # Sort by confidence and select best
                grasp_candidates.sort(key=lambda x: x[1], reverse=True)
                best_candidate = grasp_candidates[0][0]
                
                # Enhanced grasp decision logic
                should_grasp = self._should_grasp_object(best_candidate, frame_size)
            
            # Update performance metrics
            self._update_performance_metrics(start_time, detections)
            
            return detections, best_candidate, should_grasp
            
        except Exception as e:
            print(f"❌ Error in frame processing: {e}")
            self.error_count += 1
            return [], None, False
    
    def _should_grasp_object(self, candidate: Tuple, frame_size: Tuple[int, int]) -> bool:
        """
        Enhanced grasp decision making with multiple criteria.
        
        Args:
            candidate: Detection candidate
            frame_size: Frame dimensions
            
        Returns:
            bool: True if should grasp
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_grasp_time < self.grasp_cooldown:
            return False
        
        # Get object center and calculate distance from frame center
        center_x, center_y = self.object_detector.get_object_center(candidate)
        frame_center_x, frame_center_y = frame_size[0] // 2, frame_size[1] // 2
        
        distance_from_center = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
        max_distance = min(frame_size[0], frame_size[1]) * 0.3  # 30% of smaller dimension
        
        # Check if object is close enough to center
        if distance_from_center > max_distance:
            return False
        
        # Check object size (avoid very small or very large objects)
        x, y, w, h, confidence = candidate
        object_area = w * h
        frame_area = frame_size[0] * frame_size[1]
        area_ratio = object_area / frame_area
        
        if area_ratio < 0.001 or area_ratio > 0.1:  # Between 0.1% and 10% of frame
            return False
        
        # Check if object has been stable in grasp zone
        self.grasp_candidates.append((candidate, current_time))
        
        # Count how long this object has been in grasp zone
        stable_time = 0
        for past_candidate, past_time in self.grasp_candidates:
            if self._is_same_object(candidate, past_candidate):
                stable_time = current_time - past_time
                break
        
        # Only grasp if object has been stable for minimum duration
        return stable_time >= self.min_grasp_duration
    
    def _is_same_object(self, obj1: Tuple, obj2: Tuple) -> bool:
        """Check if two detections are likely the same object."""
        x1, y1, w1, h1, _ = obj1
        x2, y2, w2, h2, _ = obj2
        
        # Calculate IoU
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = w1 * h1 + w2 * h2 - intersection_area
        
        if union_area == 0:
            return False
        
        iou = intersection_area / union_area
        return iou > 0.5  # 50% overlap threshold
    
    def _update_performance_metrics(self, start_time: float, detections: List[Tuple]):
        """Update performance monitoring metrics."""
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Update FPS
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Update FPS every 30 frames
            elapsed_time = time.time() - self.fps_start_time
            self.current_fps = 30 / elapsed_time
            self.fps_start_time = time.time()
        
        # Update detection history
        self.detection_history.append(len(detections))
    
    def execute_grasp(self, candidate: Tuple) -> bool:
        """
        Enhanced grasping sequence with better error handling.
        
        Args:
            candidate: Detection candidate to grasp
            
        Returns:
            bool: True if grasp was successful
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_grasp_time < self.grasp_cooldown:
            return False
        
        print("🦾 Executing enhanced grasp sequence...")
        
        try:
            # Calculate optimal grasp force based on object size
            x, y, w, h, confidence = candidate
            object_area = w * h
            
            # Adjust force based on object size
            if object_area < 5000:
                force = 40  # Light force for small objects
            elif object_area < 15000:
                force = 60  # Medium force for medium objects
            else:
                force = 80  # Strong force for large objects
            
            # Attempt to grasp
            if self.arduino.grasp_object(force=force):
                self.last_grasp_time = current_time
                self.grasp_attempts += 1
                print(f"✅ Grasp command sent successfully with force {force}")
                
                # Wait for grasp to complete
                time.sleep(2)
                
                # Check gripper status
                status = self.arduino.get_gripper_status()
                if status:
                    print(f"📊 Gripper status: {status}")
                    if "OK" in status:
                        self.successful_grasps += 1
                        print("🎉 Grasp appears successful!")
                
                return True
            else:
                print("❌ Grasp command failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during grasp execution: {e}")
            self.error_count += 1
            return False
    
    def release_grasp(self):
        """Release the gripper with enhanced error handling."""
        print("🔄 Releasing gripper...")
        try:
            if self.arduino.release_object():
                print("✅ Release command sent successfully")
            else:
                print("❌ Release command failed")
        except Exception as e:
            print(f"❌ Error during release: {e}")
    
    def draw_interface(self, frame, detections: List[Tuple], best_candidate: Optional[Tuple], grasp_ready: bool):
        """
        Enhanced user interface with more information and better visualization.
        
        Args:
            frame: Input frame
            detections: List of detected objects
            best_candidate: Best grasp candidate
            grasp_ready: Whether grasp is ready
        """
        # Draw detections with enhanced visualization
        if self.show_detections:
            frame = self.object_detector.draw_detections(frame, detections)
        
        # Draw grasp zone with enhanced visualization
        if self.show_grasp_zone:
            height, width = frame.shape[:2]
            zone_x = int(width * 0.375)
            zone_y = int(height * 0.375)
            zone_w = int(width * 0.25)
            zone_h = int(height * 0.25)
            
            # Draw grasp zone with different colors based on state
            if grasp_ready:
                color = (0, 255, 0)  # Green when ready
                thickness = 3
            else:
                color = (0, 0, 255)  # Red when not ready
                thickness = 2
            
            cv2.rectangle(frame, (zone_x, zone_y), (zone_x + zone_w, zone_y + zone_h), color, thickness)
            cv2.putText(frame, "GRASP ZONE", (zone_x, zone_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw center point
            center_x, center_y = width // 2, height // 2
            cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
        
        # Draw enhanced statistics
        stats_text = [
            f"FPS: {self.current_fps:.1f}",
            f"Frames: {self.frames_processed}",
            f"Objects: {self.objects_detected}",
            f"Grasps: {self.grasp_attempts}",
            f"Success: {self.successful_grasps}",
            f"Status: {'GRASP READY' if grasp_ready else 'SCANNING'}",
            f"Errors: {self.error_count}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw debug information
        if self.show_debug_info and best_candidate:
            x, y, w, h, conf = best_candidate
            debug_text = [
                f"Candidate: ({x},{y}) {w}x{h}",
                f"Confidence: {conf:.2f}",
                f"Area: {w*h}",
                f"Distance: {self._get_distance_from_center(best_candidate, frame.shape[1], frame.shape[0]):.1f}"
            ]
            
            for i, text in enumerate(debug_text):
                cv2.putText(frame, text, (width - 300, 30 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw instructions
        instructions = [
            "Press 'q' to quit",
            "Press 'r' to release gripper",
            "Press 'c' to calibrate",
            "Press 'd' to toggle detections",
            "Press 'z' to toggle grasp zone",
            "Press 'i' to toggle debug info"
        ]
        
        for i, text in enumerate(instructions):
            y_pos = height - 140 + i * 20
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _get_distance_from_center(self, detection: Tuple, width: int, height: int) -> float:
        """Calculate distance from frame center."""
        center_x, center_y = self.object_detector.get_object_center(detection)
        frame_center_x, frame_center_y = width // 2, height // 2
        return np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
    
    def handle_keyboard(self, key: int) -> bool:
        """
        Enhanced keyboard input handling.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            bool: True if should continue, False if should quit
        """
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.release_grasp()
        elif key == ord('c'):
            print("🔧 Recalibrating gripper...")
            self.arduino.calibrate_gripper()
        elif key == ord('d'):
            self.show_detections = not self.show_detections
            print(f"📊 Detections {'ON' if self.show_detections else 'OFF'}")
        elif key == ord('z'):
            self.show_grasp_zone = not self.show_grasp_zone
            print(f"🎯 Grasp zone {'ON' if self.show_grasp_zone else 'OFF'}")
        elif key == ord('i'):
            self.show_debug_info = not self.show_debug_info
            print(f"🔍 Debug info {'ON' if self.show_debug_info else 'OFF'}")
        
        return True
    
    def run(self):
        """Enhanced main system loop with better error handling."""
        print("🚀 Starting enhanced autonomous robotic hand system...")
        print("📋 Controls:")
        print("   q - Quit")
        print("   r - Release gripper")
        print("   c - Calibrate gripper")
        print("   d - Toggle detections")
        print("   z - Toggle grasp zone")
        print("   i - Toggle debug info")
        
        self.running = True
        
        try:
            while self.running:
                # Check for too many errors
                if self.error_count > self.max_errors:
                    print(f"❌ Too many errors ({self.error_count}), restarting system...")
                    self.error_count = 0
                    if not self.initialize_system():
                        break
                
                # Read frame from camera
                frame = self.camera.read_frame()
                if frame is None:
                    print("❌ Failed to read frame from camera")
                    self.error_count += 1
                    time.sleep(0.1)
                    continue
                
                # Process frame
                detections, best_candidate, grasp_ready = self.process_frame(frame)
                
                # Update statistics
                self.frames_processed += 1
                if detections:
                    self.objects_detected += len(detections)
                
                # Execute grasp if ready
                if grasp_ready and best_candidate:
                    self.execute_grasp(best_candidate)
                
                # Draw interface
                self.draw_interface(frame, detections, best_candidate, grasp_ready)
                
                # Display frame
                cv2.imshow("Enhanced Autonomous Robotic Hand", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard(key):
                    break
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            self.error_count += 1
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Enhanced cleanup with better resource management."""
        print("🧹 Cleaning up enhanced system...")
        
        try:
            # Release gripper
            self.release_grasp()
            
            # Close camera
            self.camera.release()
            
            # Disconnect Arduino
            self.arduino.disconnect()
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            # Print final statistics
            print(f"📊 Final Statistics:")
            print(f"   Frames processed: {self.frames_processed}")
            print(f"   Objects detected: {self.objects_detected}")
            print(f"   Grasp attempts: {self.grasp_attempts}")
            print(f"   Successful grasps: {self.successful_grasps}")
            print(f"   Success rate: {(self.successful_grasps/self.grasp_attempts*100):.1f}%" if self.grasp_attempts > 0 else "   Success rate: N/A")
            
            print("✅ Enhanced cleanup complete")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    print("\n⚠️  Received shutdown signal")
    sys.exit(0)

def main():
    """Enhanced main entry point."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    arduino_port = "COM3"  # Default for Windows
    camera_index = 0
    
    if len(sys.argv) > 1:
        arduino_port = sys.argv[1]
    if len(sys.argv) > 2:
        camera_index = int(sys.argv[2])
    
    # Create and run enhanced system
    robotic_hand = AutonomousRoboticHand(arduino_port, camera_index)
    
    if robotic_hand.initialize_system():
        robotic_hand.run()
    else:
        print("❌ Failed to initialize enhanced system. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main() 