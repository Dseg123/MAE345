"""
Keyboard-controlled Crazyflie drone with data logging for imitation learning.
Captures: camera images, drone state estimates, and actions at each timestep.
Created for MAE345
"""

import logging
import time
import cv2
import numpy as np
import json
import shutil
from datetime import datetime
from pathlib import Path
from threading import Thread
from pynput import keyboard

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander

# Configuration
group_number = 80
CROP_SIZE = 80
URI = f'radio://0/{group_number}/2M'
camera_index = 1  # Adjust if needed
CONTROL_RATE_HZ = 20  # Adjust if needed
VELOCITY = 0.2

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


class ImitationLearningDataCollector:
    """Collects state/action pairs with camera images for imitation learning."""
    
    def __init__(self, output_dir='imitation_data', trial_number=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session_dir = None
        self.data_log = []
        self.frame_count = 0
        self.start_time = None
        self.trial_number = trial_number
        
        # Current state and action - these get updated asynchronously
        self.current_state = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
        self.current_action = [0.0, 0.0, 0.0, 0.0]  # [vx, vy, vz, yaw_rate]
        
        # For synchronized collection
        self.collecting = False
        self.last_capture_time = 0
        self.capture_period = 1.0 / CONTROL_RATE_HZ
        
        # For collection thread control
        self.thread_running = True  # separate flag for thread lifecycle
        
        # Camera setup
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Data collector initialized")
        print(f"Collection rate: {CONTROL_RATE_HZ} Hz")
    
    def start_session(self):
        """Start a new data collection session."""
        if self.trial_number is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.output_dir / f"session_{timestamp}"
        else:
            self.session_dir = self.output_dir / f"trial_{self.trial_number:02d}"
        
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.session_dir / "images").mkdir(exist_ok=True)
        
        self.start_time = time.time()
        self.last_capture_time = self.start_time
        self.frame_count = 0
        self.data_log = []
        self.collecting = True
        
        print(f"Started data collection session: {self.session_dir}")
        print("Recording at 10 Hz (synchronized state-action-observation)")
    
    def update_state(self, state_dict):
        """Update the current state estimate from drone sensors."""
        self.current_state = state_dict.copy()
    
    def update_action(self, vx, vy, vz, yaw_rate):
        """Update the current action being sent to the drone."""
        self.current_action = [vx, vy, vz, yaw_rate]
    
    def try_capture_timestep(self):
        """
        Try to capture a timestep if enough time has elapsed.
        This ensures fixed-rate synchronized capture of state-action-observation.
        Returns True if captured, False otherwise.
        """
        if not self.collecting or self.session_dir is None:
            return False
        
        current_time = time.time()
        
        # Only capture if enough time has elapsed since last capture
        if current_time - self.last_capture_time < self.capture_period:
            return False
        
        # Capture camera frame
        ret, frame = self.cap.read()
        if not ret:
            print("Warning: Failed to capture frame")
            return False
        
        # Crop top portion to remove camera overlay (battery/frequency display)
        crop_top = CROP_SIZE  
        if frame.shape[0] > crop_top:
            frame = frame[crop_top:, :, :]
        
        # Save image
        image_filename = f"frame_{self.frame_count:06d}.jpg"
        image_path = self.session_dir / "images" / image_filename
        cv2.imwrite(str(image_path), frame)
        
        # Create data entry with SYNCHRONIZED state, action, and observation
        data_entry = {
            'timestep': self.frame_count,
            'timestamp': current_time - self.start_time,
            'image_path': f"images/{image_filename}",
            'state': self.current_state.copy(),
            'action': self.current_action.copy(),
        }
        
        self.data_log.append(data_entry)
        
        # Print progress every 50 frames
        if self.frame_count % 50 == 0 and self.frame_count > 0:
            elapsed = current_time - self.start_time
            print(f"Collected {self.frame_count} timesteps ({elapsed:.1f}s, {self.frame_count/elapsed:.1f} Hz)")
        
        self.frame_count += 1
        self.last_capture_time = current_time
        
        return True
    
    def draw_overlay(self, frame):
        """Draw state and action information on the frame."""
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1
        
        # Display trial number
        if self.trial_number is not None:
            cv2.putText(frame, f"Trial: {self.trial_number}", (10, y_offset), 
                        font, font_scale, color, thickness)
            y_offset += 20
        
        # Display frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, y_offset), 
                    font, font_scale, color, thickness)
        y_offset += 20
        
        # Display action
        action_str = f"Action: vx={self.current_action[0]:.2f}, vy={self.current_action[1]:.2f}, vz={self.current_action[2]:.2f}, yaw={self.current_action[3]:.2f}"
        cv2.putText(frame, action_str, (10, y_offset), 
                    font, font_scale, color, thickness)
        y_offset += 20
        
        # Display key state info if available
        if 'x' in self.current_state:
            pos_str = f"Pos: x={self.current_state['x']:.2f}, y={self.current_state['y']:.2f}, z={self.current_state['z']:.2f}"
            cv2.putText(frame, pos_str, (10, y_offset), 
                        font, font_scale, color, thickness)
    
    def save_session(self):
        """Save the collected data to JSON file."""
        self.collecting = False
        
        if self.session_dir is None or len(self.data_log) == 0:
            print("No data to save")
            return
        
        # Save main data log
        log_path = self.session_dir / "data_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.data_log, f, indent=2)
        
        # Save metadata
        metadata = {
            'session_name': self.session_dir.name,
            'trial_number': self.trial_number,
            'total_timesteps': self.frame_count,
            'duration': time.time() - self.start_time,
            'collection_rate_hz': 1.0 / self.capture_period,
            'camera_resolution': [640, 480],
            'state_variables': list(self.current_state.keys()),
            'action_format': ['vx', 'vy', 'vz', 'yaw_rate'],
            'data_format': 'Each entry contains synchronized state, action, and observation (image) at a fixed timestep'
        }
        
        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nData saved to {self.session_dir}")
        print(f"Total timesteps collected: {self.frame_count}")
        print(f"Session duration: {metadata['duration']:.2f} seconds")
        print(f"Average collection rate: {self.frame_count / metadata['duration']:.1f} Hz")
    
    def stop_thread(self):
        """Signal the collection thread to stop."""
        self.thread_running = False
    
    def cleanup(self):
        """Release camera resources."""
        self.cap.release()


class KeyboardDroneWithLogging:
    """Keyboard control with integrated data logging."""
    
    def __init__(self, motion_commander, data_collector):
        self.mc = motion_commander
        self.data_collector = data_collector
        self.is_flying = False
        self.is_recording = False
        
        # Movement parameters (m/s)
        self.velocity = VELOCITY
        
        # Current velocities
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_rate = 0.0
        
        # Keys currently pressed
        self.pressed_keys = set()
        
        # Control thread
        self.running = False
        self.control_thread = None
        
        self.print_controls()
    
    def print_controls(self):
        print("\n=== Keyboard Controls ===")
        print("u: Take off")
        print("r: START recording")
        print("x: STOP recording")
        print("w/s: Forward/Backward")
        print("a/d: Left/Right")
        print("space: Up")
        print("c: Down")
        print("q/e: Turn left/Right")
        print("l: Land and exit")
        print("========================")
    
    def on_press(self, key):
        """Handle key press events."""
        try:
            # Handle character keys
            if hasattr(key, 'char'):
                if key.char == 'u' and not self.is_flying:
                    self.takeoff()
                elif key.char == 'r' and self.is_flying:
                    self.start_recording()
                elif key.char == 'x' and self.is_flying:
                    self.stop_recording()
                elif key.char == 'l' and self.is_flying:
                    self.land()
                    return False  # Stop listener
                elif key.char in ['w', 's', 'a', 'd', 'q', 'e', 'c']:
                    self.pressed_keys.add(key.char)
            
            # Handle special keys
            elif key == keyboard.Key.space:
                self.pressed_keys.add('space')
                
        except AttributeError:
            pass
    
    def on_release(self, key):
        """Handle key release events."""
        try:
            if hasattr(key, 'char') and key.char in self.pressed_keys:
                self.pressed_keys.discard(key.char)
            elif key == keyboard.Key.space:
                self.pressed_keys.discard('space')
        except AttributeError:
            pass
    
    def takeoff(self):
        """Take off and wait for user to start recording."""
        print("Taking off...")
        self.mc.take_off(0.5, velocity=VELOCITY)
        self.is_flying = True
        
        # Wait for PID to stabilize
        print("Hovering for 3 seconds to stabilize PID...")
        time.sleep(3)
        
        print("PID stabilized - Ready to fly!")
        print("Press 'r' to START recording, 'x' to STOP recording")
        
        # Start control loop
        self.running = True
        self.control_thread = Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
    
    def start_recording(self):
        """Start data collection."""
        if not self.is_flying:
            print("Cannot start recording - not flying!")
            return
        
        if self.is_recording:
            print("Already recording!")
            return
        
        print("▶ Recording STARTED")
        self.data_collector.start_session()
        self.is_recording = True
    
    def stop_recording(self):
        """Stop data collection."""
        if not self.is_recording:
            print("Not currently recording!")
            return
        
        print("⏹ Recording STOPPED")
        self.data_collector.save_session()
        self.is_recording = False
    
    def land(self):
        """Land and stop recording if active."""
        print("Landing...")
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        
        self.mc.land()
        self.is_flying = False
        
        # Save data if recording
        if self.is_recording:
            print("Saving data...")
            self.data_collector.save_session()
            self.is_recording = False
    
    def update_velocity_from_keys(self):
        """Update velocity commands based on currently pressed keys."""
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_rate = 0.0
        
        if 'w' in self.pressed_keys:
            self.vx = self.velocity
        if 's' in self.pressed_keys:
            self.vx = -self.velocity
        if 'a' in self.pressed_keys:
            self.vy = self.velocity
        if 'd' in self.pressed_keys:
            self.vy = -self.velocity
        if 'space' in self.pressed_keys:
            self.vz = self.velocity
        if 'c' in self.pressed_keys:
            self.vz = -self.velocity
        if 'q' in self.pressed_keys:
            self.yaw_rate = 45  # degrees/sec
        if 'e' in self.pressed_keys:
            self.yaw_rate = -45
    
    def control_loop(self):
        """Main control loop - runs in separate thread."""
        rate = CONTROL_RATE_HZ
        dt = 1.0 / rate
        
        while self.running and self.is_flying:
            # Update velocities based on pressed keys
            self.update_velocity_from_keys()
            
            # Send command to drone
            self.mc.start_linear_motion(self.vx, self.vy, self.vz, self.yaw_rate)
            
            # Update action in data collector
            if self.is_recording:
                self.data_collector.update_action(self.vx, self.vy, self.vz, self.yaw_rate)
            
            time.sleep(dt)


def state_callback(timestamp, data, logconf, data_collector):
    """Callback to update state from drone logs."""
    # Update state variables as they come in
    for key, value in data.items():
        # Remove group prefix (e.g., 'stateEstimate.x' -> 'x')
        simple_key = key.split('.')[-1]
        data_collector.current_state[simple_key] = value


def collection_loop(data_collector):
    """
    Background thread that tries to capture data at fixed intervals.
    This ensures synchronized state-action-observation pairs.
    """
    print("Collection loop started")
    while data_collector.thread_running:  # FIXED: Use thread_running flag
        # Try to capture a timestep (will only succeed if collecting=True)
        data_collector.try_capture_timestep()
        
        # Small sleep to prevent busy waiting
        time.sleep(0.01)  # 10ms sleep, allows up to 100Hz checking
    
    print("Collection loop stopped")


def get_existing_trials(output_dir='imitation_data'):
    """Get list of existing trial numbers."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    
    trials = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("trial_"):
            try:
                trial_num = int(item.name.split("_")[1])
                trials.append(trial_num)
            except (IndexError, ValueError):
                continue
    
    return sorted(trials)


def prompt_trial_number(output_dir='imitation_data', max_trials=15):
    """Prompt user for trial number with validation and overwrite checking."""
    existing_trials = get_existing_trials(output_dir)
    
    print("\n" + "="*50)
    print("TRIAL DATA COLLECTION")
    print("="*50)
    
    if existing_trials:
        print(f"\nExisting trials: {existing_trials}")
        print(f"Progress: {len(existing_trials)}/{max_trials} trials completed")
        
        # Show which trials are missing
        missing_trials = [i for i in range(1, max_trials + 1) if i not in existing_trials]
        if missing_trials:
            print(f"Missing trials: {missing_trials}")
    else:
        print(f"\nNo existing trials found.")
        print(f"Progress: 0/{max_trials} trials completed")
    
    # Show visual progress bar
    completed = len(existing_trials)
    bar_length = 30
    filled = int(bar_length * completed / max_trials)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\n[{bar}] {completed}/{max_trials}")
    
    while True:
        try:
            trial_input = input(f"\nEnter trial number (1-{max_trials}): ").strip()
            trial_number = int(trial_input)
            
            if trial_number < 1 or trial_number > max_trials:
                print(f"Error: Trial number must be between 1 and {max_trials}")
                continue
            
            # Check if trial already exists
            trial_path = Path(output_dir) / f"trial_{trial_number:02d}"
            if trial_path.exists():
                print(f"\n⚠️  WARNING: Trial {trial_number} already exists!")
                print(f"Location: {trial_path}")
                
                # Check if it has data
                data_file = trial_path / "data_log.json"
                if data_file.exists():
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    print(f"Existing data: {len(data)} frames")
                
                overwrite = input("Do you want to OVERWRITE this trial? (yes/no): ").strip().lower()
                
                if overwrite in ['yes', 'y']:
                    # Remove existing data
                    shutil.rmtree(trial_path)
                    print(f"✓ Trial {trial_number} will be overwritten")
                    return trial_number
                else:
                    print("Please choose a different trial number.")
                    continue
            else:
                print(f"✓ Trial {trial_number} selected (new trial)")
                return trial_number
                
        except ValueError:
            print("Error: Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user")
            exit(0)


def main():
    # Prompt for trial number
    trial_number = prompt_trial_number(max_trials=15)
    
    print("\nInitializing...")
    
    # Initialize data collector with trial number
    data_collector = ImitationLearningDataCollector(trial_number=trial_number)
    
    # Initialize Crazyflie drivers
    cflib.crtp.init_drivers(enable_debug_driver=False)
    
    print("Connecting to Crazyflie...")
    
    # Start collection loop in background thread
    from threading import Thread
    collection_thread = Thread(target=collection_loop, args=(data_collector,), daemon=True)
    collection_thread.start()
    
    try:
        with SyncCrazyflie(URI) as scf:
            print("Connected!")
            print("Setting up logging...")
            
            # Set up logging configuration for state estimation
            # Split into two configs due to Crazyflie log size limits

            log_period_ms = int(1000 / CONTROL_RATE_HZ)  # Convert Hz to milliseconds
            
            # Position and velocity
            log_conf_pos = LogConfig(name='Position', period_in_ms=log_period_ms) 
            log_conf_pos.add_variable('stateEstimate.x', 'float')
            log_conf_pos.add_variable('stateEstimate.y', 'float')
            log_conf_pos.add_variable('stateEstimate.z', 'float')
            log_conf_pos.add_variable('stateEstimate.vx', 'float')
            log_conf_pos.add_variable('stateEstimate.vy', 'float')
            log_conf_pos.add_variable('stateEstimate.vz', 'float')
            
            # Orientation
            log_conf_att = LogConfig(name='Attitude', period_in_ms=log_period_ms)  
            log_conf_att.add_variable('stabilizer.roll', 'float')
            log_conf_att.add_variable('stabilizer.pitch', 'float')
            log_conf_att.add_variable('stabilizer.yaw', 'float')
            
            try:
                scf.cf.log.add_config(log_conf_pos)
                scf.cf.log.add_config(log_conf_att)
                print("Log configs added successfully")
            except AttributeError as e:
                print(f"Error adding log config: {e}")
                print("This might be due to invalid variable names or configuration size")
                raise
            
            # Set up callbacks to update state
            log_conf_pos.data_received_cb.add_callback(
                lambda ts, data, logconf: state_callback(ts, data, logconf, data_collector)
            )
            log_conf_att.data_received_cb.add_callback(
                lambda ts, data, logconf: state_callback(ts, data, logconf, data_collector)
            )
            
            log_conf_pos.start()
            log_conf_att.start()
            print("Logging started")
            
            # Create motion commander
            mc = MotionCommander(scf)
            
            # Create keyboard controller
            drone = KeyboardDroneWithLogging(mc, data_collector)
            
            print("\n" + "="*50)
            print("Ready to fly! Press 'u' to take off.")
            print("After takeoff, press 'r' to START recording")
            print("and 'x' to STOP recording.")
            print("Data is collected at 10 Hz with synchronized")
            print("state-action-observation pairs.")
            print("="*50)
            
            # Start keyboard listener
            with keyboard.Listener(on_press=drone.on_press, on_release=drone.on_release) as listener:
                listener.join()
            
            # Stop logging
            log_conf_pos.stop()
            log_conf_att.stop()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop the collection thread
        data_collector.stop_thread()
        collection_thread.join(timeout=2.0)
        
        # Cleanup
        data_collector.cleanup()
        print("Done!")


if __name__ == '__main__':
    main()
