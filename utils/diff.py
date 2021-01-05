# -*- coding:UTF-8 -*-

"""
文件说明：
    结果差异性对比
"""

import cv2 as cv
import numpy as np
import glob


def pred_resul(lab_path, pred_path, save_path):
    labs = glob.glob(lab_path)
    labs.sort()
    preds = glob.glob(pred_path)
    preds.sort()
    for num in range(len(labs)):
        print('第{}张图像'.format(num + 1))
        name = labs[num].split('\\')[-1].split('.')[0]
        lab = cv.imread(labs[num])
        pred = cv.imread(preds[num])
        h, w, c = lab.shape
        zers = np.zeros((h, w, c))
        for i in range(h):
            for j in range(w):
                if (lab[i, j, 0] == 255) and (pred[i, j, 0] == 255):
                    zers[i, j] = [255, 30, 30]

                if (lab[i, j, 0] == 0) and (pred[i, j, 0] == 255):
                    zers[i, j] = [30, 255, 30]

                if (lab[i, j, 0] == 255) and (pred[i, j, 0] == 0):
                    zers[i, j] = [200, 200, 255]
        cv.imwrite(save_path + '/diff_{}.png'.format(name), zers)