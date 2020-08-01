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
mg_thres_rgb = 25
mg_thres_rgb_black = 7
mg_thres_hsv = 500
mg_thres_gray_his_mean = 8
mg_thres_gray_his_std = 5


size = 800

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

# 中心裁剪
def center_crop(image, size, dim):
    x, y = image.shape[0], image.shape[1]
    new_x = x // 2 - size // 2
    new_y = y // 2 - size // 2
    if dim == 3:
        return image[new_x:(new_x + size), new_y:(new_y + size),:]
    else:
        return image[new_x:(new_x + size), new_y:(new_y + size)]

np.set_printoptions(threshold=np.inf)

# img = Image.open("target.bmp")

# img = change_image_channels(img, '3rgb_target.bmp')
# img = io.imread("3rgb_dense.png")
img = io.imread("target5.bmp")
gray_img = cv2.imread("target5.bmp", cv2.IMREAD_GRAYSCALE)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# img = img[1000:2000,2000:3000,:]
# gray_img = gray_img[1000:2000,2000:3000]
# hsv_img = hsv_img[1000:2000,2000:3000,:]

img = center_crop(img, size, 3)
gray_img = center_crop(gray_img, size, 2)
hsv_img = center_crop(hsv_img, size, 3)


print(img.shape)
img_h, img_w = img.shape[:2]

print("*******************************************************************")
print("Step 1: Super-pixel segmentation: pieces 5000, compactness 10...")
print("*******************************************************************")
segments = slic(img, n_segments = 600,
                     compactness = 8, 
                     sigma = 2.5)

# get rid of background color:
# print(segments)

# for h in range(img_h):
#     for w in range(img_w):
#         if img[h, w][0] >= br_thres_r and img[h, w][1] >= br_thres_g and img[h, w][2] >= br_thres_b:
#             segments[h,w] = 0 

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
# dum_var = 0

# Push points into each region:
for i in segment_list:
    # t00 = time.time()
    print("The current seach image number is: {}".format(i))
    region = Region(i)
    if i == 0:
        region_list.append(region)
        continue
    match_pairs = np.argwhere(segments == i)
    # print(match_pairs)
    region.add_points(match_pairs)
    # for j in range(match_pairs.shape[0]):
    #     x, y = match_pairs[j]
    #     region.add_points((x, y))
    # get_width(region.points, img)
        # if y > 0:
        #     if segments[x, y-1] != i and segments[x, y-1] != 0 and not segments[x, y-1] in region.neighbor:
        #         region.add_neighbor(segments[x, y-1])
        # if y < img_w-1:
        #     if segments[x, y+1] != i and segments[x, y+1] != 0 and not segments[x, y+1] in region.neighbor:
        #         region.add_neighbor(segments[x, y+1])
        # if x > 0:
        #     if segments[x-1, y] != i and segments[x-1, y] != 0 and not segments[x-1, y] in region.neighbor:
        #         region.add_neighbor(segments[x-1, y])
        # if x < img_h-1:
        #     if segments[x+1, y] != i and segments[x+1, y] != 0 and not segments[x+1, y] in region.neighbor:
        #         region.add_neighbor(segments[x+1, y])
    tmp1 = sorted(match_pairs, key =  lambda x: x[0])
    tmp2 = sorted(match_pairs, key =  lambda x: x[1])
    x_0, x_1 = tmp1[0][0], tmp1[-1][0]
    y_0, y_1 = tmp2[0][1], tmp2[-1][1]
    # left:
    if x_0 > 0:
        for y_t in range(y_0, y_1+1):
            if segments[x_0-1, y_t] != i and not segments[x_0-1, y_t] in region.neighbor:
                region.add_neighbor(segments[x_0-1, y_t]) 
    # right:
    if x_1 < size-1:
        for y_t in range(y_0, y_1+1):
            if segments[x_1+1, y_t] != i and not segments[x_1+1, y_t] in region.neighbor:
               region.add_neighbor(segments[x_1+1, y_t])     
    # top:    
    if y_0 > 0:
        for x_t in range(x_0, x_1+1):
            if segments[x_t, y_0-1] != i and not segments[x_t, y_0-1] in region.neighbor:
               region.add_neighbor(segments[x_t, y_0-1])     
    # bottom:    
    if y_1 < size-1:
        for x_t in range(x_0, x_1+1):
            if segments[x_t, y_1+1] != i and not segments[x_t, y_1+1] in region.neighbor:
               region.add_neighbor(segments[x_t, y_1+1])            
    # print("The if and add costs {}".format(time.time() - t00))

    region.calc_mean_rgb(img)
    region.calc_mean_hsv(hsv_img)
    region.gaussian_mean_std(gray_img)        
    region_list.append(region)
    if region.mean_rgb[1] < 110 and region.mean_rgb[2] < 110:
        region.islight = False

    

# Merge:
print("*******************************************************************")
print("Step 2: Merging, based on RGB, HSV and gray scale histogram for each region...")
print("*******************************************************************")
merged_dic = {}
for item in zip(segment_list, segment_list):
    merged_dic[item[0]] = item[1]

for region in region_list:
    if merged_dic[region.number] != region.number or region.number == 0:
        continue
    print("The current targeted region is Region {}".format(region.number))

    # print(region.points)
    # print(region.neighbor)
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

        # both dark regions:
        if region.islight and compared_region.islight:
            if np.linalg.norm(target_mean_rgb - compared_region.mean_rgb) <= mg_thres_rgb or (abs(target_gray_mean - compared_region.gray_mean) <= mg_thres_gray_his_mean and abs(target_gray_std - compared_region.gray_std) <= mg_thres_gray_his_std) or (HSVDistance(target_mean_hsv, compared_region.mean_hsv) <= mg_thres_hsv):


                print("We now merge Region {} into Region {}".format(n, region.number))
                # merged_list.append(map_dic[n.number])
                merged_dic[n] = region.number
                delete_region_number.append(n)
                # region.add_points(compared_region.points)
                region.points.extend(compared_region.points)

        # both light regions or one light + one dark region:
        else:
            if np.linalg.norm(target_mean_rgb - compared_region.mean_rgb) <= mg_thres_rgb_black or (abs(target_gray_mean - compared_region.gray_mean) <= mg_thres_gray_his_mean and abs(target_gray_std - compared_region.gray_std) <= mg_thres_gray_his_std) or (HSVDistance(target_mean_hsv, compared_region.mean_hsv) <= mg_thres_hsv):
                print("We now merge Region {} into Region {}".format(n, region.number))
                # merged_list.append(map_dic[n.number])
                merged_dic[n] = region.number
                delete_region_number.append(n)
                # region.add_points(compared_region.points)
                region.points.extend(compared_region.points)
                # region.points = np.concatenate((region.points, compared_region.points), axis = 0)

    region.calc_mean_rgb(img)
    region.calc_mean_hsv(hsv_img)
    region.gaussian_mean_std(gray_img)
    if region.mean_rgb[1] < 110 and region.mean_rgb[2] < 110:
        region.islight = False
    else:
        region.islight = True
        
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
    for pair in region.points:
        x, y = pair[0], pair[1]
        merged_segments[x, y] = label
            

out = mark_boundaries(img,segments)

print("---------------------------------------------------------")
print("End of super-pixel segmentation + merging...")
print("It takes {} seconds...".format(time.time() - t0))

print("*******************************************************************")
print("Results displaying...")
print("*******************************************************************")
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
    # mask = np.zeros(img.shape[:2], dtype="uint8")
    # mask[merged_segments == segVal] = 255
    mask = np.zeros(img.shape, dtype="uint8")
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         mask[i,j,:] = np.array([255,255,255])
    mask[:,:,:] = 255
    # mask = img.copy()
    # mask[merged_segments != segVal] = np.array([255,255,255])
    mask[merged_segments == segVal] = img[merged_segments == segVal]

    # show the masked region
    write_path = './SegmentationData5'
    # if os.path.exists(write_path):
    #     # not empty
    #     if os.listdir(write_path):
    #         os.remove(write_path)
    #         os.makedirs(write_path)
    # else:
    #     os.makedirs(write_path)
    cv2.imwrite(os.path.join(write_path, "{}.jpg".format((str(i)))), mask)