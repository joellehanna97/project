import os
import subprocess
import cv2 as cv
import re

fixed_dir = '/Users/joellehanna/Desktop/Master/Semestre 2/Projet/frames_to_test/'
original_dir = '/Users/joellehanna/Desktop/Master/Semestre 2/Projet/project/video_test/'

os.makedirs(fixed_dir)

def sort_alphanum(a):

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(a, key = alphanum_key)
cpt = 0
all_files = [f for f in os.listdir('/Users/joellehanna/Desktop/Master/Semestre 2/Projet/project/video_test') if f.endswith('.jpg')]
sorted_files = sort_alphanum(all_files)
for file in sorted_files:
      print(file)
      if cpt%2 == 0:
          file_corrupt = os.path.join(original_dir, file)
          print(file_corrupt)
          image = cv.imread(file_corrupt)
          str_name = "frame_{:d}.jpg".format(cpt)
          cv.imwrite(os.path.join(fixed_dir, str_name), image)
      cpt = cpt+1
