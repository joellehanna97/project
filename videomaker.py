import cv2 as cv
import numpy as np
import glob
import os
import re

img_array = []


def sort_alphanum(a):

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(a, key = alphanum_key)

all_files = [f for f in os.listdir('/Users/joellehanna/Desktop/Master/Semestre 2/Projet/project/video_test') if f.endswith('.jpg')]
sorted_files = sort_alphanum(all_files)
for file in sorted_files:
#for filename in glob.glob('/Users/joellehanna/Desktop/Master/Semestre 2/Projet/frames_to_test/*.jpg'):
    #img = cv2.imread(file)
    #height, width, layers = img.shape
    file_corrupt = os.path.join('/Users/joellehanna/Desktop/Master/Semestre 2/Projet/project/video_test/', file)
    print(file_corrupt)
    img = cv.imread(file_corrupt)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv.VideoWriter('video_HR.avi',cv.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
