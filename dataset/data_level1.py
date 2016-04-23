# -*- coding: utf-8 -*-
"""
Convert data for LEVEL1 training data.

@author: York(xy_xie@nuaa.edu.cn)
"""

import os
from os.path import join, exists
import cv2
import numpy as np
import h5py
from utils import flip, getDataFromTxt, rotate, logger, shuffle_in_unison_scary, processImage, createDir

TRAIN = '/home/yhb-pc/CNN_Face_keypoint/dataset/train'
OUTPUT = '/home/yhb-pc/CNN_Face_keypoint/train'
if not exists(OUTPUT): os.mkdir(OUTPUT)
assert(exists(TRAIN) and exists(OUTPUT))



def generate_hdf5_data(filelist, output, fname, argument=False):
    data = getDataFromTxt(filelist)
    F_imgs = []
    F_landmarks = []
    EN_imgs = []
    EN_landmarks = []
    NM_imgs = []
    NM_landmarks = []
    for (imgPath, bbox, landmarkGt) in data:    
        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)
        # Paper Table2 jitter F-layer
        f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
        f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]
        
        
        ## data argument
        if argument and np.random.rand() > -1:
            ### flip
            face_flipped, landmark_flipped = flip(f_face, landmarkGt)
            face_flipped = cv2.resize(face_flipped, (39, 39))
            F_imgs.append(face_flipped.reshape((1, 39, 39)))
            F_landmarks.append(landmark_flipped.reshape(10))
            ### rotation +5 degrees
            if np.random.rand() > 0.5:
                face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, \
                    bbox.reprojectLandmark(landmarkGt), 5)
                landmark_rotated = bbox.projectLandmark(landmark_rotated)
                face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (39, 39))
                F_imgs.append(face_rotated_by_alpha.reshape((1, 39, 39)))
                F_landmarks.append(landmark_rotated.reshape(10))
                ### flip with rotation
                face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                face_flipped = cv2.resize(face_flipped, (39, 39))
                F_imgs.append(face_flipped.reshape((1, 39, 39)))
                F_landmarks.append(landmark_flipped.reshape(10))
            ### rotation -5 degrees
            if np.random.rand() > 0.5:
                face_rotated_by_alpha, landmark_rotated = rotate(img, f_bbox, \
                    bbox.reprojectLandmark(landmarkGt), -5)
                landmark_rotated = bbox.projectLandmark(landmark_rotated)
                face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (39, 39))
                F_imgs.append(face_rotated_by_alpha.reshape((1, 39, 39)))
                F_landmarks.append(landmark_rotated.reshape(10))
                ### flip with rotation
                face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                face_flipped = cv2.resize(face_flipped, (39, 39))
                F_imgs.append(face_flipped.reshape((1, 39, 39)))
                F_landmarks.append(landmark_flipped.reshape(10))

        f_face = cv2.resize(f_face, (39, 39))
        en_face = f_face[:31, :]
        nm_face = f_face[8:, :]

        f_face = f_face.reshape((1, 39, 39))
        f_landmark = landmarkGt.reshape((10))
        F_imgs.append(f_face)
        F_landmarks.append(f_landmark)
        
        ## data argument for EN   
        if argument and np.random.rand() > 0.5:
            ### flip
            face_flipped, landmark_flipped = flip(en_face, landmarkGt)
            face_flipped = cv2.resize(face_flipped, (31, 39)).reshape((1, 31, 39))
            landmark_flipped = landmark_flipped[:3, :].reshape((6))
            EN_imgs.append(face_flipped)
            EN_landmarks.append(landmark_flipped)

        en_face = cv2.resize(en_face, (31, 39)).reshape((1, 31, 39))
        en_landmark = landmarkGt[:3, :].reshape((6))
        EN_imgs.append(en_face)
        EN_landmarks.append(en_landmark) 
        ## data argument for NM
        
        if argument and np.random.rand() > 0.5:
            ### flip
            face_flipped, landmark_flipped = flip(nm_face, landmarkGt)
            face_flipped = cv2.resize(face_flipped, (31, 39)).reshape((1, 31, 39))
            landmark_flipped = landmark_flipped[2:, :].reshape((6))
            NM_imgs.append(face_flipped)
            NM_landmarks.append(landmark_flipped)

        nm_face = cv2.resize(nm_face, (31, 39)).reshape((1, 31, 39))
        nm_landmark = landmarkGt[2:, :].reshape((6))
        NM_imgs.append(nm_face)
        NM_landmarks.append(nm_landmark)
    #Convert the list to array
    F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
    EN_imgs, EN_landmarks = np.asarray(EN_imgs), np.asarray(EN_landmarks)
    NM_imgs, NM_landmarks = np.asarray(NM_imgs),np.asarray(NM_landmarks)
    ### normalize the data and shu
    F_imgs = processImage(F_imgs)
    shuffle_in_unison_scary(F_imgs, F_landmarks)
    EN_imgs = processImage(EN_imgs)
    shuffle_in_unison_scary(EN_imgs, EN_landmarks)
    NM_imgs = processImage(NM_imgs)
    shuffle_in_unison_scary(NM_imgs, NM_landmarks)
    
    # full face
    base = join(OUTPUT, '1_F')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = F_imgs.astype(np.float32)
        h5['landmark'] = F_landmarks.astype(np.float32)
        
    # eye and nose
    base = join(OUTPUT, '1_EN')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = EN_imgs.astype(np.float32)
        h5['landmark'] = EN_landmarks.astype(np.float32)

    # nose and mouth
    base = join(OUTPUT, '1_NM')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = NM_imgs.astype(np.float32)
        h5['landmark'] = NM_landmarks.astype(np.float32)

if __name__ == '__main__':
    # train data
    train_txt = join(TRAIN, 'trainImageList.txt')
    generate_hdf5_data(train_txt, OUTPUT, 'train.h5', argument=True)

    test_txt = join(TRAIN, 'testImageList.txt')
    generate_hdf5_data(test_txt, OUTPUT, 'test.h5')

    with open(join(OUTPUT, '1_F/train.txt'), 'w') as fd:
        fd.write('train/1_F/train.h5')
    with open(join(OUTPUT, '1_EN/train.txt'), 'w') as fd:
        fd.write('train/1_EN/train.h5')
    with open(join(OUTPUT, '1_NM/train.txt'), 'w') as fd:
        fd.write('train/1_NM/train.h5')
    with open(join(OUTPUT, '1_F/test.txt'), 'w') as fd:
        fd.write('train/1_F/test.h5')
    with open(join(OUTPUT, '1_EN/test.txt'), 'w') as fd:
        fd.write('train/1_EN/test.h5')
    with open(join(OUTPUT, '1_NM/test.txt'), 'w') as fd:
        fd.write('train/1_NM/test.h5')
    # Done
