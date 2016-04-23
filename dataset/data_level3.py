# -*- coding: utf-8 -*-
"""
Convert data for LEVEL3 training data.

@author: York(xy_xie@nuaa.edu.cn)
"""
from collections import defaultdict
import time
import cv2
import numpy as np
import h5py
from utils import getDataFromTxt, createDir, logger, shuffle_in_unison_scary, processImage, getPatch, randomShiftWithArgument

## number(0.16/0.18) froms table 2 in paper
types = [(0, 'LE1', 0.11),
         (0, 'LE2', 0.12),
         (1, 'RE1', 0.11),
         (1, 'RE2', 0.12),
         (2, 'N1', 0.11),
         (2, 'N2', 0.12),
         (3, 'LM1', 0.11),
         (3, 'LM2', 0.12),
         (4, 'RM1', 0.11),
         (4, 'RM2', 0.12),]
for t in types:
    d = 'train/3_%s' % t[1]
    createDir(d)

def generate(ftxt, mode, argument=False):
    """
        Generate Training Data for LEVEL-3
        mode = train or test
    """
    data = getDataFromTxt(ftxt)

    trainData = defaultdict(lambda: dict(patches=[], landmarks=[]))
    for (imgPath, bbox, landmarkGt) in data:
        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)

        landmarkPs = randomShiftWithArgument(landmarkGt, 0.01)
        if not argument:
            landmarkPs = [landmarkPs[0]]

        for landmarkP in landmarkPs:
            for idx, name, padding in types:
                patch, patch_bbox = getPatch(img, bbox, landmarkP[idx], padding)
                patch = cv2.resize(patch, (15, 15))
                patch = patch.reshape((1, 15, 15))
                trainData[name]['patches'].append(patch)
                _ = patch_bbox.project(bbox.reproject(landmarkGt[idx]))
                trainData[name]['landmarks'].append(_)

    for idx, name, padding in types:
        logger('writing training data of %s'%name)
        patches = np.asarray(trainData[name]['patches'])
        landmarks = np.asarray(trainData[name]['landmarks'])
        patches = processImage(patches)

        shuffle_in_unison_scary(patches, landmarks)

        with h5py.File('train/3_%s/%s.h5'%(name, mode), 'w') as h5:
            h5['data'] = patches.astype(np.float32)
            h5['landmark'] = landmarks.astype(np.float32)
        with open('train/3_%s/%s.txt'%(name, mode), 'w') as fd:
            fd.write('train/3_%s/%s.h5'%(name, mode))


if __name__ == '__main__':
    np.random.seed(int(time.time()))
    # trainImageList.txt
    generate('/home/yhb-pc/CNN_Face_keypoint/dataset/train/trainImageList.txt', 'train', argument=True)
    # testImageList.txt
    generate('/home/yhb-pc/CNN_Face_keypoint/dataset/train/testImageList.txt', 'test')
    # Done
