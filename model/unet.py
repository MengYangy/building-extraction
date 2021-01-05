try:
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import sys
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import *
    from tensorflow.keras.layers import *
    from tensorflow.keras.optimizers import *
    from tensorflow.keras import backend as keras
    from tensorflow.keras.preprocessing.image import img_to_array
    import random
    import cv2 as cv
except ModuleNotFoundError as reason:
    print('错误原因是：' + str(reason))


def unet(imgw, imgh, clas_num):
    inputs = tf.keras.Input(shape=(imgw, imgh, 3))
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D()(conv1)

    conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D()(conv2)

    conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D()(conv3)

    conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D()(conv4)

    conv5 = tf.keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(conv5)
    conv5_up = tf.keras.layers.Conv2DTranspose(512,
                                               kernel_size=3,
                                               strides=(2, 2),
                                               padding='same',
                                               activation='relu')(conv5)

    t1_concat = tf.concat([conv4, conv5_up], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(t1_concat)
    conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(conv6)
    conv6_up = tf.keras.layers.Conv2DTranspose(256,
                                               kernel_size=3,
                                               strides=(2, 2),
                                               padding='same',
                                               activation='relu')(conv6)
    t2_concat = tf.concat([conv3, conv6_up], axis=3)

    conv7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(t2_concat)
    conv7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv7)
    conv7_up = tf.keras.layers.Conv2DTranspose(128,
                                               kernel_size=3,
                                               strides=(2, 2),
                                               padding='same',
                                               activation='relu')(conv7)

    t3_concat = tf.concat([conv2, conv7_up], axis=3)

    conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(t3_concat)
    conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv8)
    conv8_up = tf.keras.layers.Conv2DTranspose(64,
                                               kernel_size=3,
                                               strides=(2, 2),
                                               padding='same',
                                               activation='relu')(conv8)

    t4_concat = tf.concat([conv1, conv8_up], axis=3)

    conv9 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(t4_concat)
    conv9 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv9)

    out_put_layer = tf.keras.layers.Conv2D(clas_num, (3, 3), padding='same', activation='softmax')(conv9)

    new_model = tf.keras.models.Model(inputs=inputs,
                                      outputs=out_put_layer)

    new_model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])

    new_model.summary()
    return new_model


def load_img(path, grayscale=False):
    if grayscale:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img = np.array(img, dtype='float') / 255
    else:
        img = cv.imread(path)
        img = np.array(img, dtype='float') / 127.5 - 1
    return img


def get_train_val(imgpath, val_rate=0.1):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(imgpath):
        train_url.append(pic)
    total_num = len(train_url)
    val_num = int(total_num * val_rate)
    random.shuffle(train_url)

    for i in range(total_num):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


def generateData(imgpath, labpath, batch_size, data):
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in range(len(data)):
            url = data[i]
            batch += 1
            img = load_img(imgpath + '/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(labpath + '/' + url, grayscale=True)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0


def generateValidData(imgpath, labpath, batch_size, data):
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(imgpath + '/' + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(labpath + '/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0


def train_model(m_Inputimgspath, m_Inputlabspath, m_Outmodelpath, m_w, m_h, epo, bs, class_num):
    print('/************ train unet model *******************/')
    sys.stdout.flush()
    EPOCHS = int(epo)
    BS = int(bs)
    img_path = m_Inputimgspath
    lab_path = m_Inputlabspath
    model_path = m_Outmodelpath
    img_w = int(m_w)
    img_h = int(m_h)
    img_class = int(class_num)
    train_set, val_set = get_train_val(imgpath=img_path, val_rate=0.1)
    train_num = int(len(train_set) / BS)
    val_num = int(len(val_set) / BS)
    myGene = generateData(imgpath=img_path, labpath=lab_path, batch_size=BS, data=train_set)
    myGene_test = generateValidData(imgpath=img_path, labpath=lab_path, batch_size=BS, data=val_set)
    model = unet(imgw=img_w, imgh=img_h, clas_num=img_class)
    model.fit_generator(myGene, steps_per_epoch=train_num, epochs=EPOCHS, verbose=1,
                        validation_data=myGene_test, validation_steps=val_num)
    print('/**********   Save the model   ************/')
    sys.stdout.flush()
    model.save(model_path)



if __name__ == '__main__':
    try:
        train_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
    except Exception as e:
        print('错误原因是： ' + str(e))
    # train_model(m_Inputimgspath='D:\\Algorithm\\trains', m_Inputlabspath='D:\\Algorithm\\labels',
    #           m_Outmodelpath='D:\\Algorithm\\unet.h5', m_w=64, m_h=64, epo=1, bs=1, class_num=2)

    '''
    train_model定义的算法
    def train_model(m_Inputimgspath, m_Inputlabspath, m_Outmodelpath, m_w, m_h, epo, bs)
    m_Inputimgspath = sys.argv[1] 输入训练图片
    m_Inputlabspath = sys.argv[2] 输入训练标签
    m_Outmodelpath = sys.argv[3] 模型保存位置
    m_w = sys.argv[4] 输入图片的宽度
    m_h = sys.argv[5] 输入图片的高度
    epo = sys.argv[6] 模型迭代次数
    bs = sys.argv[7] 每一步输入图片的训练量
    class_num = sys.argv[8]   分类类别
    '''
