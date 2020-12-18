import cv2 as cv
import numpy as np
import funcs

from itertools import combinations
from image import Image
from keypoint import Keypoint

class Sfm:
    def __init__(self, images, cameras = None):
        # Check the number of images
        if len(images) < 2:
            raise Exception("Need more images")

        # Initalize the object
        self.images = [Image(image) for image in images]
        self.cameras = cameras
        self.cloud = []
        self.keypoints = []
        self.buckets = []

        # Check the cameras
        if self.cameras is not None:
            # Add the cameras
            for image, camera in zip(self.images, self.cameras):
                image.add_camera(camera)

    def build(self):
        # Get the features of the images
        self._get_features()

        # Match the keypoints
        self._match_features()

        # Build the cloud
        return self._build_cloud()

    def render(self, cloud):
        pass

    def _get_features(self):
        # Use SIFT
        sift = cv.SIFT_create()

        # Loop throught the images
        for image in self.images:
            # Get the keypoints
            kpts, descs = sift.detectAndCompute(image.data, None)

            # Create the keypoints
            keypoints = [Keypoint(image, kptn, desc) for kptn, desc in zip(kpts, descs)]

            # Add the keypoints to the imagae
            image.add_keypoints(keypoints)

            # Add the keypoint to the general list
            self.keypoints = self.keypoints + keypoints

        # Create the buckets list
        self.buckets = [[] for _ in range(len(self.keypoints))]
    
    def _match_features(self, filter_num = None):
        # Create a brute force matcher
        bf = cv.BFMatcher(crossCheck=True)

        # Loop through the pair of images
        for i, img_a in enumerate(self.images[:-1]):
            for img_b in self.images[i+1:]:
                # Get the descriptors
                desc_a = img_a.get_descriptors()
                desc_b = img_b.get_descriptors()

                # Get the matches sorted
                matches = bf.match(desc_a, desc_b)
                matches = sorted(matches, key = lambda m: m.distance)

                # Check the filter
                if filter_num is not None:
                    # The filtered list
                    filtered = []
                    for match in matches:
                        if match.distance < filter_num:
                            filtered.append(match)
                else:
                    # Keep the whole list
                    filtered = matches

                # Go through the filter list
                for match in filtered:
                    # Get the ids
                    query_id = img_a.get_keypoint_id(match.queryIdx)
                    train_id = img_b.get_keypoint_id(match.trainIdx)

                    # Add the keypoint to the bucket
                    self.buckets[query_id].append(train_id)

    def _build_cloud(self):
        # The cloud
        cloud = []

        # Add the cameras
        for image in self.images:
            K, R, T = funcs.camera_decompose(image.get_camera())
            cloud.append([T[0,0], T[1,0], T[2,0], 0, 0, 255])

        # Loop through the buckets
        for i, bucket in enumerate(self.buckets):
            # Check the bucket
            if len(bucket) < 5:
                continue

            # The list of keypoints
            keypoints = [self.keypoints[j] for j in [i] + bucket]

            # Get the points and cameras
            points = [keypoint.get_point() for keypoint in keypoints]
            cameras = [keypoint.get_camera() for keypoint in keypoints]

            # Get the pixel
            pixel = keypoints[0].get_pixel()

            if 0 in pixel:
                continue

            # Get the point
            vec = funcs.triangulate(points, cameras)

            # Add the point to the cloid
            cloud.append([vec[0], vec[1], vec[2], pixel[2], pixel[1], pixel[0]])

        # Return the cloud
        return np.array(cloud)

    