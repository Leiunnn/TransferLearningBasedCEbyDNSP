from keras.models import Sequential,  Model
from keras.layers import Convolution2D,Input,BatchNormalization,Conv2D,Activation,Lambda,Subtract,Conv2DTranspose, PReLU,Reshape
from keras.regularizers import l2
from keras.layers import  Reshape,Dense,Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from scipy.io import loadmat
import keras.backend as K
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
import numpy as np
import math
import os
from scipy import interpolate
#from scipy.misc import imresize
from keras.utils.vis_utils import plot_model
import tensorflow as tf

# ========信道估计网络============
def SRCNN_model():

    input_shape = (256,64,1)
    x = Input(shape = input_shape)
    c1 = Convolution2D( 64 , 9 , 9 , activation = 'relu', init = 'he_normal', border_mode='same',dim_ordering = 'tf')(x)
    c2 = Convolution2D( 32 , 1 , 1 , activation = 'relu', init = 'he_normal', border_mode='same',dim_ordering = 'tf')(c1)
    c3 = Convolution2D( 1 , 5 , 5 , init = 'he_normal', border_mode='same',dim_ordering = 'tf')(c2)
    c4 = Convolution2D( 1 , 5 , 5 , init = 'he_normal', border_mode='same',dim_ordering = 'tf')(c3)
    c5 = Convolution2D( 1 , 5 , 5 , init = 'he_normal', border_mode='same',dim_ordering = 'tf')(c4)
    model = Model(input = x, output = c5)

    ##compile
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def SRCNN_train(train_data, train_label, val_data, val_label, channel_model, path):
    path1 = path + 'SRCNN/'  # --保存路径
    os.makedirs(path1)

    srcnn_model = SRCNN_model()
    print(srcnn_model.summary())  # --打印网络结构
    plot_model(srcnn_model, to_file=path1 + 'srcnn_model.png', show_shapes=True)  # --保存网络结构

    # --返回训练以及校验损失
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses_train = []
            self.losses_val = []

        def on_batch_end(self, batch, logs={}):
            self.losses_train.append(logs.get('loss'))

        def on_epoch_end(self, epoch, logs={}):
            self.losses_val.append(logs.get('loss'))

    history = LossHistory()

    weights_file = path1 + "srcnn_model_" + channel_model + "_" + ".h5"
    checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='min')  # --5次不变则降学习速率

    srcnn_model.fit(train_data, train_label, batch_size=20, validation_data=(val_data, val_label),
                    callbacks=[history, checkpoint, reduce_lr], shuffle=True, epochs=200, verbose=0)

    return history.losses_train, history.losses_val, path1


def SRCNN_predict(input_data, channel_model, path1):
    srcnn_model = SRCNN_model()
    srcnn_model.load_weights(path1 + "srcnn_model_" + channel_model + "_" + ".h5")
    predicted = srcnn_model.predict(input_data)
    return predicted


def DNCNN_model():
    inpt = Input(shape=(256,64,1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',data_format='channels_last')(inpt)
    x = Activation('relu')(x)
    # 18 layers, Conv+BN+relu
    for i in range(18):
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',data_format="channels_last")(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
        # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same',data_format="channels_last")(x)
    x = Subtract()([inpt, x])  # input - noise
    model = Model(inputs=inpt, outputs=x)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def DNCNN_train(train_data, train_label, val_data, val_label, channel_model, path):
    path2 = path + 'DNCNN/'  # --保存路径
    os.makedirs(path2)

    dncnn_model = DNCNN_model()
    print(dncnn_model.summary())
    plot_model(dncnn_model, to_file=path2 + 'dncnn_model.png', show_shapes=True)  # --保存网络结构

    # --返回训练以及校验损失
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses_train = []
            self.losses_val = []

        def on_batch_end(self, batch, logs={}):
            self.losses_train.append(logs.get('loss'))

        def on_epoch_end(self, epoch, logs={}):
            self.losses_val.append(logs.get('loss'))

    history = LossHistory()

    weights_file = path2 + "dncnn_model_" + channel_model + "_" + ".h5"

    checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='min')  # --5次不变则降学习速率

    dncnn_model.fit(train_data, train_label, batch_size=10, validation_data=(val_data, val_label),
                    callbacks=[history, checkpoint, reduce_lr], shuffle=True, epochs=50, verbose=0)

    return history.losses_train, history.losses_val, path2


def DNCNN_predict(input_data, channel_model, path2):
    dncnn_model = DNCNN_model()
    dncnn_model.load_weights(path2 + "dncnn_model_" + channel_model + "_" + ".h5")
    predicted = dncnn_model.predict(input_data)
    return predicted



# =======Transfer_net==========
def Transfer_net(rr, lr):

    input_shape = (512,)
    x = Input(shape=input_shape)
    y = Dense(512, activation='linear', use_bias=True, kernel_regularizer= l2(rr))(x)

    model = Model(x, y)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mse')

    return model



def TrainTransfer_net(tr_data, tr_label, val_data, val_label, rr, lr, path,snr):

    path3 = path + 'Transfer_'+ str(snr) + '/'
    os.makedirs(path3)

    dnnSD_model = Transfer_net(rr, lr)
    print(dnnSD_model.summary())
    plot_model(dnnSD_model, to_file=path3+'dnnTranser_model.png', show_shapes=True)

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses_train_sd = []
            self.losses_val_sd = []

        def on_batch_end(self, batch, logs={}):
            self.losses_train_sd.append(logs.get('loss'))

        def on_epoch_end(self, epoch, logs={}):
            self.losses_val_sd.append(logs.get('loss'))

    history_Sd = LossHistory()
    weights_file_Sd = path3 + 'Trans_nn.h5'
    checkpoint = ModelCheckpoint(weights_file_Sd, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=2)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='min')  # 5次不变则降学习速率
    dnnSD_model.fit(tr_data, tr_label,
                    batch_size=70,
                    validation_data=(val_data, val_label),
                    callbacks=[history_Sd, checkpoint, reduce_lr],
                    epochs=50,
                    verbose=2,
                    shuffle=True)

    return history_Sd.losses_train_sd, path3


def PrediTransfer_net( te_data, rr, lr, path3):

    dnnTran_model = Transfer_net(rr, lr)
    dnnTran_model.load_weights(path3 + 'Trans_nn.h5')
    predicted = dnnTran_model.predict(te_data)

    return predicted





def TransferCNN_model(FlagTr):

    inpt = Input(shape=(256,16,1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1), padding='same',data_format='channels_last',trainable=FlagTr)(inpt)
    x = Activation('relu')(x)
    # 18 layers, Conv+BN+relu
    for i in range(12):
        x = Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1), padding='same',data_format="channels_last",trainable=FlagTr)(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
        # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), padding='same',data_format="channels_last",trainable=FlagTr)(x)
    x = Subtract()([inpt, x])  # input - noise
    x = Reshape((1,256*16))(x)
    x = Dense(256*16, activation='relu', use_bias=True,trainable=True)(x)
    x = Dense(256*16, activation='linear', use_bias=True,trainable=True)(x)
    model = Model(inputs=inpt, outputs=x)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])


    return model



def TransferCNN_train(train_data, train_label, val_data, val_label, channel_model, path, FlagTr):
    path4 = path + 'TransferCNN_Pre/'  # --保存路径
    os.makedirs(path4)

    transcnn_model = TransferCNN_model(FlagTr)
    print(transcnn_model.summary())
    plot_model(transcnn_model, to_file=path4 + 'transcnn_model.png', show_shapes=True)  # --保存网络结构

    # --返回训练以及校验损失
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses_train = []
            self.losses_val = []

        def on_batch_end(self, batch, logs={}):
            self.losses_train.append(logs.get('loss'))

        def on_epoch_end(self, epoch, logs={}):
            self.losses_val.append(logs.get('loss'))

    history = LossHistory()

    weights_file = path4 + "transcnn_model_" + channel_model + "_" + ".h5"

    checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='min')  # --5次不变则降学习速率

    transcnn_model.fit(train_data, train_label, batch_size=70, validation_data=(val_data, val_label),
                    callbacks=[history, checkpoint, reduce_lr], shuffle=True, epochs=50, verbose=0)

    return history.losses_train, history.losses_val, path4


def TransferCNN_finetuning(train_data, train_label, val_data, val_label, channel_model, path, path4, FlagTr):
    path5 = path + 'TransferCNN_Fine/'  # --保存路径
    os.makedirs(path5)
    transcnn_model = TransferCNN_model(FlagTr)
    transcnn_model.load_weights(path4 + "transcnn_model_" + channel_model + "_" + ".h5")

    # --返回训练以及校验损失
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses_train = []
            self.losses_val = []

        def on_batch_end(self, batch, logs={}):
            self.losses_train.append(logs.get('loss'))

        def on_epoch_end(self, epoch, logs={}):
            self.losses_val.append(logs.get('loss'))

    history = LossHistory()

    weights_file = path5 + "transcnn_model_" + channel_model + "_" + ".h5"

    checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='min')  # --5次不变则降学习速率

    transcnn_model.fit(train_data, train_label, batch_size=70, validation_data=(val_data, val_label),
                    callbacks=[history, checkpoint, reduce_lr], shuffle=True, epochs=5, verbose=0)

    return history.losses_train, history.losses_val, path5


def TransferCNN_predict(input_data, channel_model, path4, FlagTr):
    transcnn_model = TransferCNN_model(FlagTr)
    transcnn_model.load_weights(path4 + "transcnn_model_" + channel_model + "_" + ".h5")

    predicted = transcnn_model.predict(input_data)
    return predicted


