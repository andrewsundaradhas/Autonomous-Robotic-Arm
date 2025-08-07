import serial
import time
from typing import Optional, Dict, Any
import threading

class ArduinoController:
    """Serial communication controller for Arduino robotic gripper."""
    
    def __init__(self, port: str = "COM3", baudrate: int = 9600, timeout: float = 1.0):
        """
        Initialize Arduino controller.
        
        Args:
            port (str): Serial port (e.g., "COM3" on Windows, "/dev/ttyUSB0" on Linux)
            baudrate (int): Baud rate for serial communication
            timeout (float): Serial timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.is_connected = False
        self.lock = threading.Lock()
        
        # Gripper states
        self.GRIPPER_OPEN = 0
        self.GRIPPER_CLOSED = 1
        self.GRIPPER_PARTIAL = 2
        
        # Movement commands
        self.COMMANDS = {
            'grasp': 'G',
            'release': 'R',
            'partial': 'P',
            'status': 'S',
            'calibrate': 'C'
        }
        
    def connect(self) -> bool:
        """
        Connect to Arduino via serial port.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            with self.lock:
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout
                )
                
                # Wait for Arduino to initialize
                time.sleep(2)
                
                # Test connection
                self.send_command('status')
                response = self.read_response()
                
                if response and 'OK' in response:
                    self.is_connected = True
                    print(f"Successfully connected to Arduino on {self.port}")
                    return True
                else:
                    print(f"Failed to establish connection with Arduino on {self.port}")
                    self.disconnect()
                    return False
                    
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino."""
        with self.lock:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            self.is_connected = False
            print("Disconnected from Arduino")
    
    def send_command(self, command: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send command to Arduino.
        
        Args:
            command (str): Command to send
            parameters (Optional[Dict]): Optional parameters for the command
            
        Returns:
            bool: True if command sent successfully
        """
        if not self.is_connected or not self.serial_conn:
            print("Not connected to Arduino")
            return False
        
        try:
            with self.lock:
                # Build command string
                cmd_str = self.COMMANDS.get(command, command)
                
                if parameters:
                    # Add parameters to command
                    param_str = ','.join([f"{k}={v}" for k, v in parameters.items()])
                    cmd_str += f":{param_str}"
                
                # Add newline for Arduino to recognize end of command
                cmd_str += '\n'
                
                # Send command
                self.serial_conn.write(cmd_str.encode('utf-8'))
                self.serial_conn.flush()
                
                return True
                
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def read_response(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Read response from Arduino.
        
        Args:
            timeout (Optional[float]): Custom timeout for reading
            
        Returns:
            Optional[str]: Response from Arduino, or None if failed
        """
        if not self.is_connected or not self.serial_conn:
            return None
        
        try:
            with self.lock:
                # Set timeout if provided
                if timeout is not None:
                    original_timeout = self.serial_conn.timeout
                    self.serial_conn.timeout = timeout
                
                # Read response
                response = self.serial_conn.readline().decode('utf-8').strip()
                
                # Restore original timeout
                if timeout is not None:
                    self.serial_conn.timeout = original_timeout
                
                return response if response else None
                
        except Exception as e:
            print(f"Error reading response: {e}")
            return None
    
    def grasp_object(self, force: int = 50) -> bool:
        """
        Command gripper to grasp an object.
        
        Args:
            force (int): Grasp force (0-100)
            
        Returns:
            bool: True if grasp command sent successfully
        """
        return self.send_command('grasp', {'force': force})
    
    def release_object(self) -> bool:
        """
        Command gripper to release object.
        
        Returns:
            bool: True if release command sent successfully
        """
        return self.send_command('release')
    
    def partial_grasp(self, position: int = 50) -> bool:
        """
        Command gripper to partial grasp.
        
        Args:
            position (int): Partial grasp position (0-100)
            
        Returns:
            bool: True if partial grasp command sent successfully
        """
        return self.send_command('partial', {'position': position})
    
    def get_gripper_status(self) -> Optional[str]:
        """
        Get current gripper status.
        
        Returns:
            Optional[str]: Gripper status or None if failed
        """
        if self.send_command('status'):
            return self.read_response()
        return None
    
    def calibrate_gripper(self) -> bool:
        """
        Calibrate the gripper.
        
        Returns:
            bool: True if calibration command sent successfully
        """
        return self.send_command('calibrate')
    
    def is_connected(self) -> bool:
        """
        Check if connected to Arduino.
        
        Returns:
            bool: True if connected
        """
        return self.is_connected and self.serial_conn and self.serial_conn.is_open
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect() 