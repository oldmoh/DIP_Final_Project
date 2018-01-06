# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:30:32 2018

@author: User
"""

import cv2
import numpy as np


gt_image_name = "../NEW-AerialImageDataset/AerialImageDataset/train/gt/austin1.tif"
test_name = "../output/result77.tif"

gt = cv2.imread(gt_image_name)
test = cv2.imread(test_name)

#normalize
gt = gt / 255
test = test / 255

#
correct = gt * test

print("Correct classified ratio: ")
print(np.sum(correct[:,:,0]) / np.sum(gt[:,:,0]))
print("Labled vs Predict ")
print(np.sum(gt[:,:,0]),end=" vs ")
print(np.sum(test[:,:,0]))
print("Missed ratio: ",end="")
print(np.sum((gt-correct)[:,:,0]) / np.sum(gt[:,:,0]))
print("Misclassified ratio: ",end="")
print(np.sum((test-correct)[:,:,0]) / np.sum(gt[:,:,0]))

