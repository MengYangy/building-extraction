from model.unet import train_model
from utils.DataPro import data_pro
from utils.SingleArea import Contour_area
from utils.SinglePred import img_pred
from utils.evaluation import test_func
from utils.diff import pred_resul

'''
1、数据增强 data_pro(), 
    参数有7: 输入图像，输入标签，输出图像，输出标签，每张随机裁剪数量，裁剪尺寸w,h
    效果：对一张图像进行随机裁剪、上下镜像、左右镜像
'''

data_pro('./images/src/*', './images/lab/*', './images/trains', './images/labels', 15, 64, 64)

'''
2、模型训练 train_model()
    参数有8： 输入图像，输入标签， 模型保存路径， 输入图像尺寸 w, h, 训练批次epoch， batch size, 分类数
    训练得到Unet模型
'''
train_model('./images/trains', './images/labels', './results/unet.h5', 64, 64, 2, 1, 2)

'''
3、预测 img_pred()
    参数有4： 待预测图像， 训练好的模型， 结果保存路径， 模型输入大小
'''
img_pred('./images/test/img/2_102.tif', './results/unet.h5', './results/pred/pred_2_102.tif', 64)

'''
4、轮廓检测+面积计算 Contour_area
    参数有4： 待检测原始图像， 待检测预测后的二值化图像， 结果保存路径， 待检测图像的空间分辨率
'''
Contour_area('./images/test/2_102.tif', './results/pred/pred_2_102.tif', './results/contour_2_102.tif', 0.3)

'''
5、模型性能评价指标 test_func()
    参数有3： 原始标签， 预测二值化图像， 需要使用多少张图像做评价指标
'''
test_func('./images/test/label/2_102.tif', './results/pred/pred_2_102.tif', 1)

'''
6、差异性对比 pred_resul()
    参数有3： 原始标签，预测得到的二值化图像， 保存位置
'''
pred_resul('./images/test/label/*', './results/pred/*', './results')