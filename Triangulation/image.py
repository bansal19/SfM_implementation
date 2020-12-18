import numpy as np

class Image:
    def __init__(self, image):
        # Initialize the image
        self.data = image
        self.keypoints = None
        self.camera = None

    def add_keypoints(self, kpts):
        # Set the keypoints
        self.keypoints = kpts

    def get_descriptors(self):
        return np.array([kptn.desc for kptn in self.keypoints])

    def get_keypoint_id(self, idx):
        return self.keypoints[idx].id

    def add_camera(self, camera):
        # Add the camera
        self.camera = camera

    def get_camera(self):
        return self.camera