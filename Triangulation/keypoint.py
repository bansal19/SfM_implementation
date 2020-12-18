
class Keypoint:
    # ID counter
    _id = 0

    def __init__(self, image, kptn, desc):
        # Initialize the keypoint
        self.id = Keypoint._id
        self.image = image
        self.kptn = kptn
        self.desc = desc

        # Increment the id counter
        Keypoint._id += 1


    def get_point(self):
        return self.kptn.pt

    def get_camera(self):
        return self.image.get_camera()

    def get_pixel(self):
        point = self.kptn.pt
        return self.image.data[int(point[1])][int(point[0])]
