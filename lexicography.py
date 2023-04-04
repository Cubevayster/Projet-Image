import matplotlib.pyplot as plt
import cv2
import numpy as np

class Lexicography:

    def __init__(self, path, split_width, split_height):
        self.img = cv2.imread(path)
        self.width = self.img.shape
        self.height = self.img.shape
        self.split_width = split_width
        self.split_height = split_height

    def start_points(self, size, split_size, overlap=0):
        points = [0]
        stride = int(split_size*(1-overlap))
        cpt = 1
        while True:
            pt = stride * cpt
            if pt + split_size >= size:
                if split_size == size:
                    break
                points.append(size-split_size)
                break
            else:
                points.append(pt)
            cpt += 1
        return points

    def computeXY_points(self):
        X_points = self.start_points(self.width, self.split_width, 0.5)
        Y_points = self.start_points(self.height, self.split_height, 0.5)
        return X_points, Y_points

    def write_image(self, X_points, Y_points, count, name, format):
        for i in X_points:
            for j in Y_points:
                split = self.img[i:i+self.split_height, j:j+self.split_width]
                cv2.imwrite('{}_{}.{}'.format(name, count, format), split)
                count += 1