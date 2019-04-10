import os
import subprocess
import re


def sort_alphanum(a):

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(a, key = alphanum_key)


all_files = [f for f in os.listdir('/home/best_student/Documents/SR_Joelle/project/frames_to_test') if f.endswith('.jpg')]
sorted_files = sort_alphanum(all_files)
