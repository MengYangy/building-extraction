try:
    import sys
    import cv2 as cv
    import numpy as np
    import random
    import tensorflow as tf
    import glob
except ModuleNotFoundError as reason:
    print('错误原因是：' + str(reason))


def data_pro(a, b, c, d, img_num, w, h):
    img_num = int(img_num)
    img_w = int(w)
    img_h = int(h)
    counts = 3
    count = 0
    count_s = 1
    images = glob.glob(a)
    labels = glob.glob(b)
    for i in range(len(images)):
        image = cv.imread(images[i])
        label = cv.imread(labels[i])
        for j in range(counts):
            [weight, height, tong] = image.shape  # 获得原始图片的大小
            if j == 1:  # 数据增强，左右翻转
                image = tf.image.flip_left_right(image).numpy()
                label = tf.image.flip_left_right(label).numpy()
                count_s += 1
            elif j == 2:  # 数据增强， 上下翻转
                image = tf.image.flip_up_down(image).numpy()
                label = tf.image.flip_up_down(label).numpy()
                count_s += 1
            while count < img_num * count_s:
                #                 print(count_s)
                random_width = random.randint(0, weight - img_w - 1)
                random_height = random.randint(0, height - img_h - 1)
                train_roi = image[random_width:random_width + img_w, random_height:random_height + img_h]
                label_roi = label[random_width:random_width + img_w, random_height:random_height + img_h]
                cv.imwrite(c + '/%d.png' % (count), train_roi)
                cv.imwrite(d + '/%d.png' % (count), label_roi)
                count += 1
                print(count)
                sys.stdout.flush()
        count_s = count_s + 1




if __name__ == '__main__':
    try:
        data_pro(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    except Exception as e:
        print('错误原因是： ' + str(e))


#    data_pro(a='D:\Algorithm\src',
#            b=r'D:\Algorithm\label',
#            c=r'D:\Algorithm\trains',
#            d=r'D:\Algorithm\labels',
#            img_num=5,
#            w=64,
#            h=64)
