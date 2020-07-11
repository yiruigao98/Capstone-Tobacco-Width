from PIL import Image
from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from region import Region
import cv2
import os
# from width_detect import *
from hsv import *
import time


# Parameter set 1: number of segemented pieces and compactness
num_seg = 100
compactness = 10

# Parameter set 2: get rid of background
br_thres_r = 130
br_thres_g = 130
br_thres_b = 125

# Parameter set 3: merging threshold
mg_thres_rgb = 15
mg_thres_hsv = 100
mg_thres_gray_his_mean = 15
mg_thres_gray_his_std = 5


print("---------------------------------------------------------")
print("Start to count the time...")
t0 = time.time()

# 通道转换
def change_image_channels(image, image_path):
    # 4通道转3通道
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
        image.save(image_path)
    return image

np.set_printoptions(threshold=np.inf)

# img = Image.open("target.bmp")

# img = change_image_channels(img, '3rgb_target.bmp')
# img = io.imread("3rgb_dense.png")
img = io.imread("target.bmp")
gray_img = cv2.imread("target.bmp", cv2.IMREAD_GRAYSCALE)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img = img[500:2500,1000:3000,:]
gray_img = gray_img[500:2500,1000:3000]
hsv_img = hsv_img[500:2500,1000:3000,:]

print(img.shape)
img_h, img_w = img.shape[:2]

segments = slic(img, n_segments = 5000, compactness = 10)
print("shit!")
# get rid of background color:
# print(segments)

for h in range(img_h):
    for w in range(img_w):
        if img[h, w][0] >= br_thres_r and img[h, w][1] >= br_thres_g and img[h, w][2] >= br_thres_b:
            segments[h,w] = 0 

# Create mapped dictionary:
segment_list = np.unique(segments)
N = len(segment_list)
map_dic = {}
for item in zip(segment_list, range(N)):
    map_dic[item[0]] = item[1]
print(map_dic)
print("The length of labels including background is {}".format(N))


region_list = []
delete_region_number = []
dum_var = 0

# Push points into each region:
for i in segment_list:
    print("The current seach image number is: {}".format(i))
    region = Region(i)
    if i == 0:
        region_list.append(region)
        continue
    match_pairs = np.argwhere(segments == i)
    for j in range(match_pairs.shape[0]):
        x, y = match_pairs[j]
        region.add_points((x, y))
    # get_width(region.points, img)
        if y > 0:
            if segments[x, y-1] != i and segments[x, y-1] != 0 and not segments[x, y-1] in region.neighbor:
                region.add_neighbor(segments[x, y-1])
        if y < img_w-1:
            if segments[x, y+1] != i and segments[x, y+1] != 0 and not segments[x, y+1] in region.neighbor:
                region.add_neighbor(segments[x, y+1])
        if x > 0:
            if segments[x-1, y] != i and segments[x-1, y] != 0 and not segments[x-1, y] in region.neighbor:
                region.add_neighbor(segments[x-1, y])
        if x < img_h-1:
            if segments[x+1, y] != i and segments[x+1, y] != 0 and not segments[x+1, y] in region.neighbor:
                region.add_neighbor(segments[x+1, y])
    region.calc_mean_rgb(img)
    region.calc_mean_hsv(hsv_img)
    region.gaussian_mean_std(gray_img)        
    region_list.append(region)


# Merge:
merged_dic = {}
for item in zip(segment_list, segment_list):
    merged_dic[item[0]] = item[1]


for region in region_list:
    if merged_dic[region.number] != region.number:
        continue
    print("The current targeted region is Region {}".format(region.number))
    target_mean_rgb = region.mean_rgb
    target_mean_hsv = region.mean_hsv
    target_gray_mean = region.gray_mean
    target_gray_std = region.gray_std
    neighbors = list(set([merged_dic[m] for m in region.neighbor if m != 0]))
    print("The neighbors of Region {} is {}".format(region.number, neighbors))
    # merged_list = []
    for n in neighbors:
        print("The current checked region for Region {} is Region {}".format(region.number, n))
        if n in delete_region_number:
            compared_region = region_list[map_dic[merged_dic[n]]]
        else:
            compared_region = region_list[map_dic[n]]
        if np.linalg.norm(target_mean_rgb - compared_region.mean_rgb) <= mg_thres_rgb or (abs(target_gray_mean - compared_region.gray_mean) <= mg_thres_gray_his_mean and abs(target_gray_std - compared_region.gray_std) <= mg_thres_gray_his_std) or (HSVDistance(target_mean_hsv, compared_region.mean_hsv) <= mg_thres_hsv):
            print("We now merge Region {} into Region {}".format(n, region.number))
            # merged_list.append(map_dic[n.number])
            merged_dic[n] = region.number
            delete_region_number.append(n)
            region.points.extend(compared_region.points)
    region.calc_mean_rgb(img)
    region.calc_mean_hsv(hsv_img)
    region.gaussian_mean_std(gray_img)
        
print(merged_dic, len(merged_dic))


# Re-label:
merged_segments = segments.copy()
for h in range(img_h):
    for w in range(img_w):
        merged_segments[h,w] = 0 
# print(delete_region_number)
remain_region_number = list(set([item for item in list(merged_dic.values()) if not item in delete_region_number]))
for number in remain_region_number:
    # print("The current transformed region is Region {}".format(number))
    region = region_list[map_dic[number]]
    label = number
    for j in range(len(region.points)):
        x, y = region.points[j]
        merged_segments[x, y] = label
            

out = mark_boundaries(img,segments)

print("---------------------------------------------------------")
print("End of super-pixel segmentation + merging...")
print("It takes {} seconds...".format(time.time() - t0))

# print(segments)
plt.subplot(121)
plt.title("n_segments=10000")
plt.imshow(out)

out2=mark_boundaries(img,merged_segments)
plt.subplot(122)
plt.title("n_segments=1000")
plt.imshow(out2)
plt.show()


for (i, segVal) in enumerate(np.unique(merged_segments)):
    # construct a mask for the segment
    print("[x] inspecting segment {}, for {}".format(i, segVal))
    mask = np.zeros(img.shape[:2], dtype="uint8")
    mask[merged_segments == segVal] = 255

    # show the masked region
    # cv2.imshow("Mask", mask)
    cv2.imwrite(os.path.join("SegmentationData", "{}.jpg".format((str(i)))), mask)