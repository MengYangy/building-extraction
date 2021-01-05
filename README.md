# building-extraction
Remote sensing image building extraction and optimization, using TF2

遥感影像建筑物自动提取，使用Unet模型，框架tensorflow2.0.0

# 依赖库
  `tensorflow==2.0.0` `numpy==1.18.3` `opencv-python==3.4.2.16`

# 文件介绍
  images文件夹中有5个子文件夹  
    `src` : 原始数据  
    `lab` : 原始标签  
    `trains` : 数据增强后的数据  
    `labels` : 数据增强后的标签  
    `test` ： 测试图像文件夹  
        `img` : 测试图像  
        `label` : 测试标签  
  model 文件夹中存放的是Unet.py unet模型  
  utils 文件夹中存放的是：  
    `DataPro.py` 数据增强代码  
    `diff.py` 预测结果与真实标签差异性对比代码  
    `evaluation.py` 评价指标代码  
    `SingleArea.py` 单张轮廓检测、面积计算代码  
    `SinglePred.py` 单张预测代码  
  results 文件夹中是代码运行过程中保存的各种数据  
  `train.py` 是执行程序，在这个文件中可以运行所有的代码。  
