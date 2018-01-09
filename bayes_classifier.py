# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:22:27 2017

@author: Yi-hao Mo
"""
import cv2
import numpy as np
import time


'''
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
    see the slide Object Recognition page 12
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
    Training Loop
'''
image_name = "../NEW-AerialImageDataset/AerialImageDataset/train/images/austin1.tif"
gt_image_name = "../NEW-AerialImageDataset/AerialImageDataset/train/gt/austin1.tif"

#number of training images
num_of_training = 4

mean1 = np.array([0,0,0])
cov1 = np.zeros((3,3))
mean2 = np.array([0,0,0])
cov2 = np.zeros((3,3))
num_bd_pixel = 0

#read 4 images
for x in range(1,num_of_training+1):
    image_name = image_name.replace(str(x-1),str(x),1)
    gt_image_name = gt_image_name.replace(str(x-1),str(x),1)
    
    #load image and ground truth
    source = np.array(cv2.imread(image_name))
    gt_img = np.array(cv2.imread(gt_image_name))
    #normalize ground truth
    mask = gt_img / 255
    
    #compute mean and covariance
    temp_mean1, temp_cov1 = compute_mean_cov(source, mask)
    temp_mean2, temp_cov2 = compute_mean_cov(source, 1.0 - mask)
    
    #number of pixels labeled as building
    num_bd_pixel = num_bd_pixel + np.count_nonzero(mask[:,:,0])
    
    mean1 = mean1 + temp_mean1
    mean2 = mean2 + temp_mean2
    cov1 = cov1 + temp_cov1
    cov2 = cov2 + temp_cov2

mean1 = mean1 / num_of_training
mean2 = mean2 / num_of_training
cov1 = cov1 / num_of_training
cov2 = cov2 / num_of_training

#log probability of labeled pixel and unlabeled pixel in ground truth
ln_prob_omega_1 = np.log(num_bd_pixel/(num_of_training * 5000*5000))
ln_prob_omega_2 = np.log(1.0 - num_bd_pixel/(num_of_training * 5000*5000))

print("Training Finish")


'''
    Pre-processing
    median smoothing
'''
test_name = "../NEW-AerialImageDataset/AerialImageDataset/train/images/austin1.tif"
test_img = cv2.imread(test_name)
blurred = cv2.medianBlur(test_img, 7)
image = blurred

'''
    Predicting
'''
ground_truth = np.zeros((5000,5000),dtype=np.uint8)
temp = image.reshape((5000*5000,3)) - mean1
L1 = ((np.sum(np.array(temp * np.mat(np.linalg.inv(cov1))) * temp,axis=1) + np.log(np.linalg.det(cov1))) * -0.5 + ln_prob_omega_1).reshape(5000,5000)
temp = image.reshape((5000*5000,3)) - mean2
L2 = ((np.sum(np.array(temp * np.mat(np.linalg.inv(cov2))) * temp,axis=1) + np.log(np.linalg.det(cov2))) * -0.5 + ln_prob_omega_2).reshape(5000,5000)

for index in np.ndindex(5000,5000):
    if L1[index] > L2[index]:
        ground_truth[index] = 255



'''
    Post-processing
    morphological opening
'''
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
result = cv2.morphologyEx(ground_truth, cv2.MORPH_OPEN, kernel)


cv2.imwrite("../output/ground_truth.tif",result)

