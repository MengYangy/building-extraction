try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import cv2 as cv
    import random
    import numpy as np
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import sys
except ModuleNotFoundError as reason:
    print('错误原因是：' + str(reason))


def img_pred(imgpath, modelpath, savepath, img_h_w):
    print('/************ 加载模型 *******************/')
    sys.stdout.flush()
    model = load_model(modelpath)
    stride = int(img_h_w)
    image_size = int(img_h_w)
    print('/************ 图片预测 *******************/')
    sys.stdout.flush()
    #     image=tf.io.read_file(images[im])
    #     image=tf.image.decode_png(image,channels=3)
    image = cv.imread(imgpath)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    h, w, c = image.shape
    if h % stride != 0:
        padding_h = (h // stride + 1) * stride
    else:
        padding_h = (h // stride) * stride
    if w % stride != 0:
        padding_w = (w // stride + 1) * stride
    else:
        padding_w = (w // stride) * stride
    padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
    padding_img[0:h, 0:w, :] = image[:, :, :]
    mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)

    for i in range(padding_h // stride):
        print(i)
        for j in range(padding_w // stride):
            crop = padding_img[i * stride:i * stride + image_size, j * stride:j * stride + image_size, :3]
            pred_result = np.ones((image_size, image_size), np.int8)
            crop = tf.cast(crop, tf.float32)
            test_part = crop
            test_part = test_part / 127.5 - 1
            test_part = tf.expand_dims(test_part, axis=0)
            pred_part = model.predict(test_part)
            pred_part = tf.argmax(pred_part, axis=-1)
            pred_part = pred_part[..., tf.newaxis]
            pred_part = tf.squeeze(pred_part)
            pred_result = pred_part.numpy()
            mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred_result[:h, :w]
    cv.imwrite(savepath, mask_whole)
    print('完成图像预测')
    sys.stdout.flush()


if __name__ == '__main__':
    # img_pred(imgpath=r'C:\Users\Administrator\Desktop\WeData\test\1.png',
    # savepath=r'C:\Users\Administrator\Desktop\WeData\pred\pred_1.png',
    #       modelpath=r'C:\Users\Administrator\Desktop\WeData\model\wedata_unet_v1.h5')
    try:
        img_pred(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    except Exception as e:
        print('错误原因是： ' + str(e))
