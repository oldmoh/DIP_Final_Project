# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:22:27 2017

@author: Yi-hao Mo
"""
import cv2
import numpy as np
import time


'''
image is a 5000*5000*3 numpy array
labled image has the same shape and pixel depth as those of image

computing log likelihood is easier and we don't need actual value
we only need to compare the probability of different gaussian distribution

我已經試過把圖片分成到建築、道路、植被 但是效果非常差
反而單純區分建築物與非建築物的結果比較好 至少不會標記大量非建築物
'''

'''
    return current time in millisecond
'''
curr_time = lambda: int(round(time.time() * 1000))

'''
    compute mean and covariance
'''
def compute_mean_cov(img, label):
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
    main process
'''

#read image and labels
image_name = "../NEW-AerialImageDataset/AerialImageDataset/train/images/austin1.tif"
gt_image_name = "../NEW-AerialImageDataset/AerialImageDataset/train/gt/austin1.tif"
#gt_road = "../output/road_label.tif"
#gt_plant = "../output/plant_label.tif"

#these two lines are redundant
source = np.array(cv2.imread(image_name))
gt_img = np.array(cv2.imread(gt_image_name))

#normalizing
mask = gt_img / 255

'''
    Pre-processing
'''
blurred = cv2.GaussianBlur(source, (5,5), 0)
image = blurred

'''
    training
    It takes 180352 millisecond
'''
#start_t = curr_time()
#compute mean and covariance of building and nonbuilding
mean1, cov1 = compute_mean_cov(source, mask)
mean2, cov2 = compute_mean_cov(source, 1.0 - mask)

#number of pixel labeled as building
num_bd_pixel = np.count_nonzero(mask[:,:,0])

# these are ln(P(building)) and ln(P(non-building))
ln_prob_omega_1 = np.log(num_bd_pixel/(5000*5000))
ln_prob_omega_2 = np.log(1.0 - num_bd_pixel/(5000*5000))

#print("Training time: ",end="")
#print(curr_time() - start_t)

'''
    Predicting
    It takes 33635 msec
    
    compute log likelihood
    only need to conpute exponent part
    ln p(x|omega_i) = -1/2 * ( (n*ln(2pi) + ln(determinant(C)) + (x-mean)^T * inverse of C * (x-mean)) )
    where omega_i is i-th set, n is dimension, x is pixel value, C is covariance matrix
    Note: the term n*ln(2pi) is a constant, so it can be ignored

'''
#start_t = curr_time()
ground_truth = np.zeros((5000,5000),dtype=np.uint8)
temp = image.reshape((5000*5000,3)) - mean1
L1 = ((np.sum(np.array(temp * np.mat(np.linalg.inv(cov1))) * temp,axis=1) + np.log(np.linalg.det(cov1))) * -0.5 + ln_prob_omega_1).reshape(5000,5000)
temp = image.reshape((5000*5000,3)) - mean2
L2 = ((np.sum(np.array(temp * np.mat(np.linalg.inv(cov2))) * temp,axis=1) + np.log(np.linalg.det(cov2))) * -0.5 + ln_prob_omega_2).reshape(5000,5000)

for index in np.ndindex(5000,5000):
    if L1[index] > L2[index]:
        ground_truth[index] = 255
                    
#print("Execution time: ",end="")
#print(curr_time() - start_t)



'''
    Post-processing
    morphological opening
'''
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
result = cv2.morphologyEx(ground_truth, cv2.MORPH_OPEN, kernel)


cv2.imwrite("../output/ground_truth.tif",result)

