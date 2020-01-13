# Copyright (c) 2015 Lightricks. All rights reserved.


import imageio
from os.path import join, split, isdir, isfile
from os import listdir, makedirs, remove
import numpy as np
import cv2


def put_text(img, text, location=(10, 50)):
    if location[0] < 0:
        location = (location[0] + img.shape[0], location[1])
    if location[1] < 0:
        location = (location[0], location[1] + img.shape[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = location
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

def merge_videos(src1, src2, name1, name2, dst):
    reader1 = imageio.get_reader(src1)
    reader2 = imageio.get_reader(src2)
    fps = reader1.get_meta_data()['fps']
    writer = imageio.get_writer(dst, fps=fps)

    for f1, f2 in zip(reader1, reader2):
        put_text(f1, name1)
        put_text(f2, name2)
        f_merged = np.concatenate([f1, f2], axis=1)
        writer.append_data(f_merged)
    writer.close()


def merge_videos_dir(path1, path2, dst_path):
    name1 = split(path1)[1]
    name2 = split(path2)[1]
    if not isdir(dst_path):
        makedirs(dst_path)
    for f_name in listdir(path1):
        if f_name not in listdir(path2) or not f_name.endswith('.mp4'):
            continue
        src1 = join(path1, f_name)
        src2 = join(path2, f_name)
        dst = join(dst_path, f_name)
        try:
            merge_videos(src1, src2, name1, name2, dst)
        except:
            print('Error in:', f_name)
            if isfile(dst):
                remove(dst)



path1 = '/Users/ravid/Pictures/face_videos/YOLACT/output/orig_yolact'
path2 = '/Users/ravid/Pictures/face_videos/YOLACT/output/person_yolact'
dst = '/Users/ravid/Pictures/face_videos/YOLACT/output/tolact_orig_vs_person'
merge_videos_dir(path1, path2, dst)