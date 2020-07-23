import numpy as np
import cv2
class Region:
    def __init__(self, number):
        self.number = number
        self.neighbor = []
        self.points = []
        self.mean_rgb = [0, 0, 0]
        self.mean_hsv = [0, 0, 0]
        self.gray_mean = 0
        self.gray_std = 0.0
        self.islight = True

    def add_neighbor(self, neighbor_number):
        if not neighbor_number in self.neighbor:
            self.neighbor.append(neighbor_number)

    def add_points(self, xy_pair):
        if not xy_pair in self.points:
            self.points.append(xy_pair)

    def calc_mean_rgb(self, image):
        mean_rgb = np.array([0,0,0])
        for pair in self.points:
            mean_rgb += image[pair[0], pair[1],:]
        self.mean_rgb = mean_rgb / len(self.points)

    def calc_mean_hsv(self, hsv_image):
        mean_hsv = np.array([0,0,0])
        for pair in self.points:
            mean_hsv += hsv_image[pair[0], pair[1],:]
        self.mean_hsv = mean_hsv / len(self.points)

    def gaussian_mean_std(self, gray_image):
        gray_scale_list = []
        for pair in self.points:
            gray_scale_list.append(gray_image[pair[0], pair[1]])
        self.gray_mean = np.mean(gray_scale_list)
        self.gray_std = np.std(gray_scale_list)

    # def merge(self, merge_Region, image):
    #     self.neighbor.remove(merge_Region.number)
    #     self.neighbor.extend(merge_Region.neighbor)
    #     self.points.extend(merge_Region.points)
    #     self.calc_mean_rgb(image)