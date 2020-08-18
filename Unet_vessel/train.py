import tensorflow as tf
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
from generator import ImageDataGenerator
from model import buildModel_U_net
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
import cv2
from scipy import misc
import scipy.ndimage as ndimage
import numpy as np
import os
import time
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops


class LossHistory(Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 按照batch来进行追加数据
    def on_batch_end(self, batch, logs={}):
        # 每一个batch完成后向容器里面追加loss，acc
        self.losses['batch'].append(logs.get('loss'))

    # self.accuracy['batch'].append(logs.get('acc'))
    # self.val_loss['batch'].append(logs.get('val_loss'))
    # self.val_acc['batch'].append(logs.get('val_acc'))
    # 每五秒按照当前容器里的值来绘图
    # if int(time.time()) % 5 == 0:
    # self.draw_p(self.losses['batch'], 'loss', 'train_batch')
    # self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
    # self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
    # self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')

    def on_epoch_end(self, batch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        # 每五秒按照当前容器里的值来绘图
        if int(time.time()) % 5 == 0:
            self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
            self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
            self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
            self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')

    # 绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type + '_' + label + '.jpg')
        plt.close()

    # 由于这里的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程（最后一次绘图结束，有训练了0-5秒的时间）
    # 所以这里的方法会在整个训练结束以后调用
    def end_draw(self):
        self.draw_p(self.losses['batch'], 'loss', 'train_batch')
        self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
        self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
        self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
        self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
        self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')


logs_loss = LossHistory()

rets = []
base_path1 = './dataset/images/'
base_path2 = './dataset/labels/'
data = []
anno = []


def step_decay(epoch):
    step = 16
    num = epoch // step
    if num % 10 == 0:
        lrate = 1e-4
    elif num % 200 == 1:
        lrate = 5e-5
    else:
        lrate = 1e-5
    # lrate = initial_lrate * 1/(1 + decay * (epoch - num * step))
    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lrate))
    return np.float(lrate)


def read_data(base_path1, base_path2):
    imList = os.listdir('./dataset/img/')
    data = []
    anno = []
    for i in range(len(imList)):
        img = cv2.imread('./dataset/img/' + imList[i])
        # img = cv2.medianBlur(img, 5)
        img = cv2.resize(img, (224, 224))
        mask = cv2.imread('./dataset/mask/' + imList[i])
        mask = cv2.resize(mask, (224, 224))
        mask = mask[:, :, 0] / 255.0

        data.append(img)
        anno.append(mask)
    return (np.asarray(data, dtype=np.float32), np.asarray(anno, dtype=np.uint8))


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return [cx, cy]
    else:
        return None


def train_(base_path1, base_path2):
    train_bool = False
    print('-' * 30)
    print('Creating and compiling the fully convolutional regression networks.')
    print('-' * 30)

    model = buildModel_U_net(input_dim=(224, 224, 3))
    model_checkpoint = ModelCheckpoint('Unet_weights.hdf5', monitor='loss', save_best_only=True)
    model.summary()
    print('...Fitting model...')
    print('-' * 30)
    change_lr = LearningRateScheduler(step_decay)

    if train_bool:
        data, anno = read_data(base_path1, base_path2)
        anno = np.expand_dims(anno, axis=-1)
        data = data / 255.0
        train_data = data[:int(0.7 * len(data))]
        train_anno = anno[:int(0.7 * len(data))]

        val_data = data[int(0.7 * len(data)):]
        val_anno = anno[int(0.7 * len(data)):]

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            zoom_range=0.1,
            shear_range=0.,
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True,  # randomly flip images
            fill_mode='constant',
            dim_ordering='tf')

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(train_data, train_anno, batch_size=8),
                            samples_per_epoch=train_data.shape[0],
                            nb_epoch=1500,
                            validation_data=(val_data, val_anno),
                            callbacks=[model_checkpoint, logs_loss],
                            )
    else:
        model.load_weights('Unet_weights.hdf5')
        # single_model = model.layers[-2]
        # single_model.save_weights('Unet_weights_single.hdf5')

        val_data = []
        for file in os.listdir('./dataset/img/')[-10:]:
            img = cv2.imread('./dataset/img/' + file)
            # img = cv2.medianBlur(img, 5)
            img = cv2.resize(img, (224, 224))
            val_data.append(img)
        val_data = np.array(val_data, dtype=np.float32)
        val_data = val_data / 255.0
        A = model.predict(val_data)
        a = A[:, :, :, 0] * 255
        a = np.reshape(a, (len(a), 224, 224))
        a = np.array(a, dtype=np.uint8)
        for i in range(len(a)):
            print(a[i])
            ex = cv2.imread('./dataset/img/' + os.listdir('./dataset/img/')[-10:][i])
            cv2.imwrite('./predict/' + os.listdir('./dataset/img/')[-10:][i][:-4] + 'example.jpg', ex)
            img = cv2.resize(np.array((a[i] > 0) * 255.0, dtype=np.uint8), (len(ex[0]), len(ex)))
            cv2.imwrite('./predict/' + os.listdir('./dataset/img/')[-10:][i], img)


if __name__ == '__main__':
    train_(base_path1, base_path2)
