# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:49:26 2017

@author: zander
"""

import os
import cv2
import utils
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
import line
import tensorflow as tf
from PIL import Image
import time

def thresholding(img):
    #setting all sorts of thresholds
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=90 ,thresh_max=280)
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 170))
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = utils.hls_select(img, thresh=(160, 255))
    lab_thresh = utils.lab_select(img, thresh=(155, 210))
    luv_thresh = utils.luv_select(img, thresh=(225, 255))

    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1

    return threshholded

def RotateClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip(trans_img, 1)

    return new_img

def processing(img,M,Minv,left_line,right_line):
    #img = RotateClockWise90(img)
    #img = np.rot90(img)
    prev_time = time.time()
    img = Image.fromarray(img)
    undist = img
    #get the thresholded binary image
    img = np.array(img)
    thresholded = thresholding(img)
    #perform perspective  transform
    thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    #perform detection
    if left_line.detected and right_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_by_previous(thresholded_wraped,left_line.current_fit,right_line.current_fit)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(thresholded_wraped)
    left_line.update(left_fit)
    right_line.update(right_fit)
    # #draw the detected laneline and the information
    undist = Image.fromarray(img)
    area_img, gre1 = utils.draw_area(undist,thresholded_wraped,Minv,left_fit, right_fit)
    curvature,pos_from_center = utils.calculate_curv_and_pos(thresholded_wraped,left_fit, right_fit)
    area_img = np.array(area_img)
    result = utils.draw_values(area_img,curvature,pos_from_center)
    curr_time = time.time()
    exec_time = curr_time - prev_time
    info = "time: %.2f ms" % (1000 * exec_time)
    print(info)
    return result,thresholded_wraped


left_line = line.Line()
right_line = line.Line()
#cal_imgs = utils.get_images_by_dir('camera_cal')
#object_points,img_points = utils.calibrate(cal_imgs,grid=(9,6))
M,Minv = utils.get_M_Minv()


'''
project_outpath = 'vedio_out/Rail_detection.mp4'
project_video_clip = VideoFileClip("Rail.mp4")
project_video_out_clip = project_video_clip.fl_image(lambda clip: processing(clip,M,Minv,left_line,right_line))
project_video_out_clip.write_videofile(project_outpath, audio=False)
'''

#draw the processed test image
test_imgs = utils.get_images_by_dir('pic')
undistorted = []
for img in test_imgs:
   #img = utils.cal_undistort(img,object_points,img_points)
   undistorted.append(img)

result=[]
t2=[]
c=1
for img in undistorted:
    prev_time = time.time()
    res,t1 = processing(img,M,Minv,left_line,right_line)
    curr_time = time.time()
    exec_time = curr_time - prev_time
    info = "time: %.2f ms" % (1000 * exec_time)
    print(info)
    #aa1 = expand(t1)
    #cv2.line(res, (550, 1080), (850, 300), color=(0, 255, 0), thickness=4)
    #cv2.line(res, (1400, 1080), (1000, 300), color=(0, 255, 0), thickness=4)
    #cv2.imwrite('./picpro/'+str(c)+'.jpg',res)
    #c = c + 1
    result.append(res)
    t2.append(t1)
# plt.figure(figsize=(20,68))
# for i in range(len(result)):
#
#    plt.subplot(len(result),1,i+1)
#    plt.title('thresholded_wraped image')
#    plt.imshow(result[i][:,:,::-1])
# plt.figure(0)
plt.figure(0)
plt.imshow(result[0])
plt.figure(1)
img1 = t2[0]
#cv2.line(img1, (0,0), (992,345), color=(255, 0, 0), thickness=4)
#plt.imshow(img1)
#print(img1[345,900])
print(result[0].shape)
# plt.figure(1)
plt.show()
#'''
