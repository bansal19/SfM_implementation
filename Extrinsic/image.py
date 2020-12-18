import numpy as np
import funcs

class Image:
    def __init__(self, name, data, intrinsic):
        '''
        Initializes the image object
        name - The name of the image
        data - The image data
        intrinsic - The intrinsic matrix of the image (K)
        '''
        # Initialize the image
        self.name = name
        self.data = data
        self.intrinsic = intrinsic
        self.rotation = None
        self.translation = None
        self.relations = {}

        # The keypoint data
        self.keypoints = None
        self.descriptions = None
        self.points = None
        
    def add_relation(self, relation):
        '''
        Adds a relation to the image
        `relation` must have src as this image
        relation - The Relation to add
        '''
        # Check the name
        if self.name != relation.src.name:
            # Does not match
            raise Exception('Image: src name does not match image name')

        # Add the relation
        self.relations[relation.dst.name] = relation
    
    def get_camera(self):
        '''
        Returns the camera projection matrix
        '''
        # Get the rotation and translation
        rotation = self.rotation if self.rotation is not None else np.diag([1, 1, 1])
        translation = self.translation if self.translation is not None else np.zeros(3).reshape((-1, 1))

        # Return the camera matrix
        return funcs.camera_compose(self.intrinsic, rotation, translation)
    
    def get_rotation(self):
        # Return the rotation if set, otherwise the identity
        return self.rotation if self.rotation is not None else np.diag([1, 1, 1])

    def get_translation(self):
        # Return the translation if set, otherwise zero
        return self.translation if self.translation is not None else np.zeros(3).reshape((-1, 1))

    def get_relation(self, image):
        # Return the relation from the image name
        return self.relations[image.name]