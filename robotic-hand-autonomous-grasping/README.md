#  Enhanced Autonomous Robotic Hand

This is an advanced Python + Arduino-based project that allows a robotic hand to automatically detect non-living objects and grasp them using sophisticated computer vision algorithms.

## 📌 Enhanced Features
- **Multi-Algorithm Object Detection**: Combines motion detection, background subtraction, edge detection, and color-based detection
- **Advanced Non-Living Classification**: Uses machine learning with 9 comprehensive features including symmetry, regularity, and texture analysis
- **Smart Grasp Decision Making**: Intelligent grasp zone detection with object stability tracking
- **Adaptive Force Control**: Automatically adjusts grasp force based on object size
- **Real-time Performance Monitoring**: FPS tracking, error handling, and success rate statistics
- **Enhanced User Interface**: Live debug information, confidence scores, and object tracking
- **Robust Error Handling**: Automatic retry mechanisms and graceful error recovery
- **Arduino-Controlled Robotic Gripper**: Smooth servo control with status feedback

##  Advanced Algorithms

### **Object Detection**
- **Motion-Based Detection**: Frame differencing with morphological operations
- **Background Subtraction**: Dual MOG2 and KNN background subtractors
- **Edge-Based Detection**: Canny and Sobel edge detection with density analysis
- **Color-Based Detection**: HSV color space analysis for artificial objects
- **Non-Maximum Suppression**: Intelligent detection combination and filtering

### **Non-Living Classification**
- **Color Analysis**: HSV color space analysis with artificial color detection
- **Edge Analysis**: Multi-method edge detection with regularity scoring
- **Texture Analysis**: Local Binary Pattern (LBP) texture uniformity
- **Shape Analysis**: Circularity, aspect ratio, and convexity analysis
- **Symmetry Analysis**: Horizontal and vertical symmetry detection
- **Regularity Analysis**: Gradient uniformity and pattern consistency
- **Brightness Analysis**: Consistent brightness detection
- **Contrast Analysis**: High contrast artificial object detection
- **Saturation Analysis**: High saturation artificial color detection

### **Grasp Decision Making**
- **Multi-Criteria Selection**: Confidence, area, distance, and stability scoring
- **Object Tracking**: IoU-based object tracking across frames
- **Stability Detection**: Minimum duration requirements for grasp decisions
- **Adaptive Force Control**: Object size-based force adjustment
- **Grasp Zone Optimization**: 25% center zone with distance constraints

## 📁 Enhanced Folder Structure

```
robotic-hand-autonomous-grasping/
├── README.md
├── requirements.txt
├── train_classifier.py          # NEW: ML model training script
├── vision/
│   ├── object_detection.py     # ENHANCED: Multi-algorithm detection
│   └── classify_non_living.py  # ENHANCED: ML-based classification
├── control/
│   └── serial_comm.py          # ENHANCED: Robust Arduino communication
├── utils/
│   └── camera.py               # ENHANCED: Better camera handling
├── main.py                     # ENHANCED: Advanced main application
├── model/
│   ├── object_classifier.pt    # Placeholder for future work
│   └── non_living_classifier.pkl  # NEW: Trained ML model
└── arduino/
    └── gripper_control.ino     # ENHANCED: Improved servo control
```

##  Installation & Setup

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Train the ML Model (Optional but Recommended)**
```bash
python train_classifier.py --samples 2000
```

### **3. Upload Arduino Sketch**
- Open `arduino/gripper_control.ino` in Arduino IDE
- Connect servo to pin 9
- Upload to your Arduino board

### **4. Connect Hardware**
- Connect Arduino via USB
- Connect servo motor to pin 9
- Ensure webcam is connected and working

##  Usage

### **Basic Usage**
```bash
python main.py
```

### **Advanced Usage**
```bash
# Specify custom port and camera
python main.py COM3 0

# Run with different parameters
python main.py COM4 1
```

### **Training Custom Model**
```bash
# Train with custom parameters
python train_classifier.py --samples 5000 --test-size 0.3

# Test on specific images
python train_classifier.py --test-images image1.jpg image2.jpg
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `r` | Release gripper |
| `c` | Calibrate gripper |
| `d` | Toggle detections |
| `z` | Toggle grasp zone |
| `i` | Toggle debug info |

##  Performance Features

### **Real-time Monitoring**
- **FPS Counter**: Live frame rate monitoring
- **Processing Times**: Average processing time tracking
- **Detection History**: Object detection statistics
- **Error Tracking**: Automatic error counting and recovery
- **Success Rate**: Grasp success rate calculation

### **Enhanced Statistics**
- Frames processed
- Objects detected
- Grasp attempts
- Successful grasps
- Error count
- Processing performance

##  Hardware Requirements

### **Essential Components**
- Arduino board (Uno, Nano, or similar)
- Servo motor (SG90 or similar)
- Webcam (USB or built-in)
- USB connection for Arduino

### **Recommended Setup**
- **Servo**: TowerPro SG90 or MG996R
- **Camera**: 720p or higher resolution
- **Arduino**: Uno R3 or Nano
- **Power**: 5V power supply for servo

##  Testing & Validation

### **Object Detection Testing**
- Place various objects in camera view
- Monitor detection confidence scores
- Verify grasp zone detection
- Test with different lighting conditions

### **Classification Testing**
- Test with living vs non-living objects
- Monitor classification confidence
- Verify feature extraction accuracy
- Test with synthetic data

### **Grasp Testing**
- Test with different object sizes
- Verify force adjustment
- Monitor grasp success rate
- Test stability requirements

##  Technical Details

### **Algorithm Parameters**
- **Detection Threshold**: 0.4 (adjustable)
- **Confidence Threshold**: 0.6 (adjustable)
- **Grasp Cooldown**: 2.0 seconds
- **Min Grasp Duration**: 1.0 seconds
- **Max Tracking Distance**: 100 pixels

### **Performance Optimization**
- **Frame Processing**: Optimized for real-time performance
- **Memory Management**: Efficient object tracking
- **Error Recovery**: Automatic retry mechanisms
- **Resource Cleanup**: Proper resource management

##  Troubleshooting

### **Common Issues**
1. **Camera not detected**: Check camera index in main.py
2. **Arduino not connecting**: Verify COM port and upload sketch
3. **Poor detection**: Adjust lighting and camera position
4. **Grasp failures**: Check servo connections and calibration

### **Debug Mode**
Enable debug information with `i` key to see:
- Object detection details
- Classification confidence
- Grasp decision criteria
- Performance metrics

##  License

This project is open source and available under the MIT License.

##  Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Algorithm improvements
- Hardware compatibility
- Performance optimizations
- Documentation updates

##  Future Enhancements

- **Deep Learning Integration**: YOLO or SSD object detection
- **3D Vision**: Depth camera integration
- **Multi-Object Tracking**: Advanced tracking algorithms
- **Learning from Demonstration**: User-guided training
- **Cloud Integration**: Remote monitoring and control 
