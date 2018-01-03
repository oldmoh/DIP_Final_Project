# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:22:27 2017

@author: Yi-hao Mo
"""
import os
import cv2
import numpy as np
import scipy


'''
image is a 5000*5000*3 numpy array
labled image has the same shape and pixel depth as those of image

computing log likelihood is easier and we don't need actual value
we only need to compare the probability of different gaussian distribution
'''

'''
    compute mean
'''
def compute_mean(labeled_image, count):
    summation = np.array([0,0,0])
    summation[0] = np.sum(labeled_image[:,:,0])
    summation[1] = np.sum(labeled_image[:,:,1])
    summation[2] = np.sum(labeled_image[:,:,2])
    mean = summation / count
    return mean

'''
    compute covariance matrix
'''
def compute_covariance(labeled_image, mean, count):
    cov = np.zeros([3,3])
    zeroIndex = np.array(np.nonzero(labeled_image[:,:,0]))
    zeroIndex = np.stack((zeroIndex[0],zeroIndex[1]),axis=-1)
    for index in zeroIndex:
        temp = (labeled_image[tuple(index)] - mean).reshape(3,1) 
        cov = cov + temp * temp.transpose()
    cov = cov / count
    return cov

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

image_name = "./NEW-AerialImageDataset/AerialImageDataset/train/images/austin1.tif"
gt_image_name = "./NEW-AerialImageDataset/AerialImageDataset/train/gt/austin1.tif"

#these two lines are redundant
image = np.array(cv2.imread(image_name))
gt_img = np.array(cv2.imread(gt_image_name))

# map value of label image to 0~1
mask = gt_img / 255
#number of pixel labeled as building
num_bd_pixel = np.count_nonzero(mask[:,:,0])

# masked image
building_img = image * mask
nonbuilding_img = image - building_img

# omega 1 is the building set and omega 2 is the nonbuilding set
ln_prob_omega_1 = np.log(num_bd_pixel/(5000*5000))
ln_prob_omega_2 = np.log(1.0 - num_bd_pixel/(5000*5000))


mean1 = compute_mean(building_img, num_bd_pixel)
cov1 = compute_covariance(building_img, mean1, num_bd_pixel)

mean2 = compute_mean(nonbuilding_img, 5000*5000-num_bd_pixel)
cov2 = compute_covariance(nonbuilding_img, mean2, 5000*5000-num_bd_pixel)


# labeling
ground_truth = np.zeros((5000,5000),dtype=np.uint8)
for y in range(0,5000):
    for x in range(0, 5000):
        L1 = ln_prob_omega_1 + log_likelihood(image[x,y], mean1, cov1)
        L2 = ln_prob_omega_2 + log_likelihood(image[x,y], mean2, cov2)
        if L1 > L2:
            ground_truth[x,y] = 255


cv2.imwrite("ground_truth.tif",ground_truth)

cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.imshow("image",ground_truth)
cv2.waitKey(0)

#os.system("pause")
