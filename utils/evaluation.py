# -*- coding:UTF-8 -*-

"""
文件说明：
    评价指标
"""
import cv2 as cv
import glob


def tpfn(a, b):
    TP = 1
    TN = 0
    FP = 0
    FN = 0
    aa = a
    bb = b
    for i in range(len(aa)):
        if ((aa[i] == 0) & (bb[i] == 0)):
            TN += 1
        if ((aa[i] > 0) & (bb[i] > 0)):
            TP += 1
        if ((aa[i] == 0) & (bb[i] > 0)):
            FP += 1
        if ((aa[i] > 0) & (bb[i] == 0)):
            FN += 1
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    IoU = TP / (TP + FP + FN)
    MIoU = (TP / (TP + FP + FN) + TN / (TN + FN + FP)) / 2
    Dice = (2 * TP) / (2 * TP + FP + FN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    print('Accuracy={}'.format(Accuracy))
    print('Precision={}'.format(Precision))
    print('Recall={}'.format(Recall))
    print('IoU={}'.format(IoU))
    print('MIoU={}'.format(MIoU))
    print('Dice={}'.format(Dice))
    print('F1={}'.format(F1))
    print('TN={}'.format(TN))
    print('TP={}'.format(TP))
    print('FP={}'.format(FP))
    print('FN={}'.format(FN))


def test_func(biaozhu_path, predict_path, number_photo):
    aa = []
    bb = []
    labs = glob.glob(biaozhu_path)
    preds = glob.glob(predict_path)
    for i in range(number_photo):
        print('{}：'.format(labs[i].split('\\')[-1]))
        print('{}张图像的评价指标如下：'.format(i + 1))
        a = cv.imread(labs[i])
        b = cv.imread(preds[i])

        binary = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
        binary1 = cv.cvtColor(b, cv.COLOR_BGR2GRAY)

        h, w = binary.shape
        for k in range(0, h):
            for j in range(0, w):
                aa.append(binary[k][j])
                bb.append(binary1[k][j])
        tpfn(aa, bb)

        aa = []
        bb = []
    print('\n')