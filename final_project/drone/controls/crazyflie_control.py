import numpy as np
import cv2

from cflib.positioning.motion_commander import MotionCommander

class CrazyflieControl:
    def __init__(self, 
                 mc: MotionCommander,
                 camera_index: int = 0,
                 ):
        """
        Scans for crazyflies and connects to the first one found.
        Args:
            group_number (int): The group number to connect to
        """
        self.mc = mc

        # Camera setup
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    def hover(self, height: float = 0.5):
        self.mc.take_off(height)
    

    def send_control_commands(self, actions: np.ndarray):
        '''
        Sends control commands to the crazyflie
        Args:
            actions (np.ndarray): Array of control commands [vx, vy, vz, yaw_rate]
        '''
        vx, vy, vz, yaw_rate = actions
        self.mc.start_linear_motion(vx, vy, vz, yaw_rate)
        

    def get_image(self,
                  crop_top: int = 80,):
        '''
        Gets the current observation from the crazyflie
        Returns:
            image (np.ndarray): The current observation image
        '''
        # Placeholder for image retrieval logic
        # Capture camera frame
        ret, frame = self.cap.read()

        # Crop top portion to remove camera overlay (battery/frequency display)
        if frame.shape[0] > crop_top:
            frame = frame[crop_top:, :, :]

        return frame