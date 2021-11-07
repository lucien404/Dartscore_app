# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import json
from PIL import Image
import argparse
import random
import matplotlib.pyplot as plt
from glob import glob

img0 = cv2.imread('x0.jpeg', 0)
print(img0.shape)

json_file = json.load(open('/Users/nickccnie/Desktop/feibiao/x0.json', 'r'))
print(json_file)
labels = json_file['shapes']
print(labels)

templete = np.zeros_like(img0, dtype=np.uint8)

for label in labels:
    label_score = label['label']
    points = label['points']
    if label_score[0:2] == 'c1':
        center_points = points
        cv2.fillPoly(templete, np.int32([points]), 50)
    elif label_score[0:2] == 'c2':
        cv2.fillPoly(templete, np.int32([points]), 25)
        cv2.fillPoly(templete, np.int32([center_points]), 50)
    elif label_score[0:2] == 'c3':
        score = int(label_score[3:])
        cv2.fillPoly(templete, np.int32([points]), score)
    elif label_score[0:2] == 'c4':
        score = int(label_score[3:]) * 3
        cv2.fillPoly(templete, np.int32([points]), score)
    elif label_score[0:2] == 'c5':
        score = int(label_score[3:])
        cv2.fillPoly(templete, np.int32([points]), score)
    elif label_score[0:2] == 'c6':
        score = int(label_score[3:]) * 2
        cv2.fillPoly(templete, np.int32([points]), score)

cv2.imwrite('tempelete.png', templete)


plt.imshow(templete)
plt.show()