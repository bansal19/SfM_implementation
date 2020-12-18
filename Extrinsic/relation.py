
class Relation:
    def __init__(self, E, src, dst, src_matches, dst_matches):
        '''
        Initializes the relation
        Represents a relation between two images
        E - The essential matrix between the images
        src - The source image
        dst - The destination image
        src_matches - The source matches
        dst_matches - Same as src_matches but columns swapped
        '''
        # Initialize the relation
        self.E = E
        self.src = src
        self.dst = dst
        self.src_matches = src_matches
        self.dst_matches = dst_matches

    def get_essential(self):
        return self.E

    def get_src_points(self):
        # Return the list of points on the src
        return self.src.points[self.src_matches[:,0]]

    def get_dst_points(self):
        # Return the list of points on the dst
        return self.dst.points[self.dst_matches[:,0]]

    def get_num_matches(self):
        # Return the number of matches
        return self.src_matches.shape[0]
    