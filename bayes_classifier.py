# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:22:27 2017

@author: Yi-hao Mo
"""
import os
import cv2
import numpy as np
import time
import scipy


'''
image is a 5000*5000*3 numpy array
labled image has the same shape and pixel depth as those of image

computing log likelihood is easier and we don't need actual value
we only need to compare the probability of different gaussian distribution
'''

'''
    return current time in millisecond
'''
curr_time = lambda: int(round(time.time() * 1000))

'''
    compute mean and covariance
'''
def fun(img, label):
    count = np.count_nonzero(label[:,:,0])
    summation = np.array([0,0,0],dtype = np.int64)
    temp = img * label
    summation[0] = np.sum(temp[:,:,0])
    summation[1] = np.sum(temp[:,:,1])
    summation[2] = np.sum(temp[:,:,2])
    mean = (summation / count).copy()
    
    cov = np.zeros([3,3])
    non_zero_idx = np.array(np.nonzero(label[:,:,0]))
    non_zero_idx = np.stack((non_zero_idx[0],non_zero_idx[1]), axis=-1)
    for index in non_zero_idx:
        temp = (img[tuple(index)] - mean).reshape(3,1)
        cov = cov + temp * temp.transpose()
    cov = (cov / count).copy()
    return mean, cov

'''
    compute log likelihood
    only need to conpute exponent part
    ln p(x|omega_i) = -1/2 * ( (n*ln(2pi) + ln(determinant(C)) + (x-mean)^T * inverse of C * (x-mean)) )
    where omega_i is i-th set, n is dimension, x is pixel value, C is covariance matrix
    Note: the term n*ln(2pi) is a constant, so it can be ignored
'''
def log_likelihood(x, mean, covariance):
    temp = (x - mean).reshape([3,1])
    p = -0.5 * (np.log(np.linalg.det(covariance)) + temp.transpose()*np.mat(np.linalg.inv(covariance))*temp)
    return p

'''
    main process
'''

image_name = "../NEW-AerialImageDataset/AerialImageDataset/train/images/austin1.tif"
gt_image_name = "../NEW-AerialImageDataset/AerialImageDataset/train/gt/austin1.tif"
#gt_image_name = "../output/road_label.tif"

#these two lines are redundant
source = np.array(cv2.imread(image_name))
gt_img = np.array(cv2.imread(gt_image_name))

#Pre-process
blurred = cv2.GaussianBlur(source, (5,5), 0)
image = blurred

# map value of label image to 0~1
mask = gt_img / 255

#number of pixel labeled as building
num_bd_pixel = np.count_nonzero(mask[:,:,0])

# omega 1 is the building set and omega 2 is the nonbuilding set
ln_prob_omega_1 = np.log(num_bd_pixel/(5000*5000))
ln_prob_omega_2 = np.log(1.0 - num_bd_pixel/(5000*5000))

mean1, cov1 = fun(source,mask)
mean2, cov2 = fun(source,1.0 - mask)

start_t = curr_time()
# labeling
# take 2927435 msec = 48.78 min.....QQ
print("labeling")
ground_truth = np.zeros((5000,5000),dtype=np.uint8)
for index in np.ndindex(5000,5000):
        L1 = ln_prob_omega_1 + log_likelihood(image[index], mean1, cov1)
        L2 = ln_prob_omega_2 + log_likelihood(image[index], mean2, cov2)
        if L1 > L2:
            ground_truth[index] = 255
        
        
print("Execution time: ",end="")
print(curr_time() - start_t)

cv2.imwrite("../output/ground_truth.tif",ground_truth)
#cv2.imwrite("../output/road.tif",ground_truth)


