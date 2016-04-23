# -*- coding: utf-8 -*-
"""
This is a file to show weight and feature map.

@author: York(xy_xie@nuaa.edu.cn)
"""
import cv2
from utils import getDataFromTxt, logger, processImage
import numpy as np
import matplotlib.pyplot as plt
from cnns import getCNNs

TXT = '/home/yhb-pc/CNN_Face_keypoint/dataset/test/lfpw_test_249_bbox.txt'

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def level1_Forward(img, bbox):
    """
        LEVEL-1
        img: gray image
        bbox: bounding box of face
    """
    F, _, _ = getCNNs(level=1)
    # F
    f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]
    f_face = cv2.resize(f_face, (39, 39))
    
    f_face = f_face.reshape((1, 1, 39, 39))
    f_face = processImage(f_face)
    F.forward(f_face)
    return F

def vis_square(data, name, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure() #新的绘图区
    plt.imshow(data)
    plt.savefig('/home/yhb-pc/CNN_Face_keypoint/log/%s.png'%name)
    
    
if __name__ == '__main__':
    data = getDataFromTxt(TXT, with_landmark=False)
    imgPath, bbox = data[18]
    img = cv2.imread(imgPath)
    assert(img is not None)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logger("process %s" % imgPath)
    net = level1_Forward(imgGray, bbox).cnn
    for i in range(4):
        feat = net.blobs['conv%s'%str(i+1)].data
        vis_square(feat[0], 'conv%s'%str(i+1), padval=1)
    feat = net.blobs['pool2'].data
    vis_square(feat[0], 'pool2', padval=1)
    
    for i in range(4):
        filters = net.params['conv%s'%str(i+1)][0].data
        vis_square(filters.reshape(len(filters)*len(filters[0]),len(filters[0,0])
        ,len(filters[0,0,0])), 'conv%s_para'%str(i+1), padval=1)
        
        