import os
from os.path import join as osj
import imp
import pickle
import random
random.seed(334)
import warnings
warnings.simplefilter('ignore', FutureWarning)

import h5py
import pandas as pd
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.layers import BatchNormalization, Dropout, Conv2D, TimeDistributed
from keras.layers import Lambda, Flatten, Activation, Dense, Input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, list_pictures
from keras.regularizers import l2
from keras.utils.training_utils import multi_gpu_model
from keras import optimizers
import keras.backend as K
import matplotlib.pyplot as plt
from time import gmtime, strftime, time, localtime

from model import *
from utils import save_logs_models
from config import MAIN_PATH_CGCN, MAIN_PATH_MASTER

# ROI_N = 236
gpu_count = 1
epochs = 100
ROI_N = 116
frames = 140
graph_path = "graph/FC_aug_full_large.npy"
aug_num = 1 # データ数を何倍にするか、1-30 1だと元のROIのタイムコース
target = "Kohs" # Kohs 気うつ
threshold = 100 # クラス分類での正負を分ける閾値
save_name = "12_7_classification_poly" # test結果を保存するディレクトリ、save以下

# ########################################## Load data ########################################
# Download the data from https://drive.google.com/file/d/1l029ZuOIUY5gehBZCAyHaJqMNuxRHTFc/view?usp=sharing
# with h5py.File('HCP.h5', 'r') as f:
#     x_train, x_val, x_test = f['x_train'][()], f['x_val'][()], f['x_test'][()]
#     y_train, y_val, y_test = f['y_train'][()], f['y_val'][()], f['y_test'][()]


# x_train = x_train.transpose((1,0,2))
# x_val = x_val.transpose((1,0,2))
# x_test = x_test.transpose((1,0,2))

############################

def train(ids, ys, fold):
    id_train, id_val, id_test = ids
    y_train, y_val, y_test = ys
    x_train, x_val, x_test = [], [], []
    for id_tr in id_train:
        # print(id_tr)
        if aug_num == 1:
            x_train.append(dict_shimane[id_tr]["timeseries"])
        else:
            for i in range(aug_num):
                x_train.append(dict_shimane[id_tr][f"time_series_{i}"])
            y_train.extend([int(dict_shimane[id_tr][target][0]>=threshold)]*aug_num)

    for id_va in id_val:
        if aug_num == 1:
            x_val.append(dict_shimane[id_va]["timeseries"])
        else:
            for i in range(aug_num):
                x_val.append(dict_shimane[id_va][f"time_series_{i}"])
            y_val.extend([int(dict_shimane[id_va][target][0]>=threshold)]*aug_num)

    for id_te in id_test:
        if aug_num == 1:
            x_test.append(dict_shimane[id_te]["timeseries"])
        else:
            for i in range(aug_num):
                x_test.append(dict_shimane[id_te][f"time_series_{i}"])
            y_test.extend([int(dict_shimane[id_te][target][0]>=threshold)]*aug_num)
        
    print("chance score")
    print(np.array([np.sum(y_train), np.sum(y_val), np.sum(y_test)]) / np.array(list(map(len, [y_train, y_val, y_test]))))
    print(np.array([np.sum(y_train), np.sum(y_val), np.sum(y_test)]))
    print(list(map(len, [y_train, y_val, y_test])))

    #exit()
    data = [x_train, x_val, x_test, y_train, y_val, y_test]
    x_train, x_val, x_test, y_train, y_val, y_test = list(map(np.array, data))

    ############################
    x_train = np.expand_dims(x_train, -1) # (200, 100, 236, 1)
    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)


    # Convert class vectors to binary class matrices.
    num_classes = 2


    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)

    # print("evaluate")
    # tmp(x_val, y_val, x_test, y_test)
    # exit()
    # DEBUG
    if gpu_count == 0:
        x_train, y_train = x_train[:100], y_train[:100]
    ################################ Set parameter ###############################
    # print()

    k = 5 # (3, 5, 10, 20)
    batch_size = 32 * gpu_count
    l2_reg = 1e-4
    dp = 0.5
    lr = 1e-5

    print('dp:', dp)
    print('l2:', l2_reg)
    print('batch_size:', batch_size)
    print('epochs:', epochs)
    print('lr:', lr)

    file_name='avg_edgeconv_k_' + str(k) + '_l2_' + str(l2_reg) + '_dp_' + str(dp) + f'_fold_{fold}'
    print('file_name:', file_name)

    os.system('mkdir tmp') # folder for the trained model
    tmp_name = 'tmp/tmp_' + file_name + '_' + strftime("%Y_%m_%d_%H_%M_%S", localtime()) + '.hdf5'
    # tmp_name = "tmp/tmp_avg_edgeconv_k_5_l2_0.0001_dp_0.5_2022_11_01_05_25_40.hdf5"
    print('output tmp name:', tmp_name)

    ############################################### Get pre-trained model  ############################
    weight_name = None

    # Find and load best pre-trained model
    # weight_path = 'tmp/%s/'%(site)
    # all_weights = os.listdir(weight_path)
    # all_right_models = {}
    # for n in all_weights:
    #     if '.hdf5' in n:
    #         n_split = n.split('_')
    #         if int(n_split[1+n_split.index('k')]) == k:
    #         # if int(n_split[1+n_split.index('k')]) == k and \
    #         #     float(n_split[1+n_split.index('l2')]) == l2_reg:
    #             all_right_models[float(n_split[1+n_split.index('valAcc')])] = n

    # if all_right_models:
    #     best_acc = np.max(list(all_right_models.keys()))
    #     print('-------best acc %f, model name: %s'%(best_acc, all_right_models[best_acc]))
    #     weight_name = weight_path+all_right_models[best_acc]

    ############################### get model  ######################################################
    # Download 'FC.npy' from https://drive.google.com/file/d/1WP4_9bps-NbX6GNBnhFu8itV3y1jriJL/view?usp=sharing
    model_ = get_model(
        graph_path=graph_path, 
        ROI_N=ROI_N,
        frames=frames,
        kernels=[8,8,8,16,32,32], 
        k=k, 
        l2_reg=l2_reg, 
        dp=dp,
        num_classes=num_classes, 
        weight_path=weight_name, 
        skip=[0,0])
    model_.summary()


    ######################################## Training ####################################################
    if gpu_count > 1:
        model = multi_gpu_model(model_, gpus=gpu_count)
        print("multi_gpu_model loaded.")
    else:
        model = model_

    model.compile(loss=['categorical_crossentropy'], 
                optimizer=optimizers.Adam(lr=lr),
                metrics=['accuracy'])

    print('Train...')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                                                patience=30, min_lr=1e-6)
    lr_hist = []
    class Lr_record(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs={}):
            tmp = K.eval(model.optimizer.lr)
            lr_hist.append(tmp)
            print('Ir:', tmp)
    lr_record = Lr_record()
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    # checkpointer = keras.callbacks.ModelCheckpoint(monitor='val_acc', filepath=tmp_name, 
                                                    # verbose=1, save_best_only=True)
    class multi_gpu_checkpointer(keras.callbacks.ModelCheckpoint):
        def set_model(self, model):
            self.model = model_
    checkpointer = multi_gpu_checkpointer(monitor='val_acc', filepath=tmp_name, 
                                                    verbose=1, save_best_only=True)

    model_history = model.fit(x_train, y_train,
                                shuffle=True,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(x_val, y_val),
                                callbacks=[checkpointer, lr_record, reduce_lr, earlystop])

    # multi_gpuでは、multi_gpu_model()に入れる前のモデルを保存する。（重みは共有している）
    # TODO callbackに入れらるようにしないとマルチGPUでは、最後のモデルしか保存できない
    # model_.save(tmp_name)

    print('validation...')
    # with open(model_path, 'rb') as fp:
    #     tmp = imp.load_module(model_path[:-3], fp, model_path,('.py', 'rb', imp.PY_SOURCE))

    model_best = get_model(
        graph_path=graph_path, 
        ROI_N=ROI_N,
        frames=frames,    
        kernels=[8,8,8,16,32,32], 
        k=k,
        l2_reg=l2_reg, 
        num_classes=num_classes, 
        weight_path=tmp_name, 
        # weight_path=None, 
        skip=[0,0])

    model_best.compile(loss=['categorical_crossentropy'], 
                optimizer=optimizers.Adam(lr=lr),
                metrics=['accuracy'])

    if gpu_count > 1:
        batch_size = int(batch_size / gpu_count)

    val_tmp = model_best.evaluate(x=x_val, y=y_val, batch_size=batch_size, verbose=1)
    print('validation:', val_tmp)

    test_tmp = model_best.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
    print('test:', test_tmp)

    predict = model_best.predict(x=x_test, batch_size=batch_size, verbose=1)
    predict_bin = np.argmax(predict, axis=1)
    print('test:', test_tmp)
    
    save_dir = osj(".", "save", save_name)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(osj(save_dir, f"fold_{fold+1}_test"), y_test=y_test, predict=predict, predict_bin=predict_bin)

    ######################################## save log and model #######################################

    save_logs_models(model, model_history, acc=val_tmp[1],
        folder=f'save/{save_name}/', 
        lr_hist=lr_hist, file_name=file_name, loss_name='loss', 
        acc_name='acc', tmp_name=tmp_name)


if __name__ == "__main__":
    main_path = MAIN_PATH_MASTER
    with open(osj(main_path, "input/shimane/data_dict_augumentation_kohs_large.pkl"), "rb") as f:
        dict_shimane = pickle.load(f)
    # print(dict_shimane)
    # exit()
    # タイムコースはあるが気うつデータがない
    # del dict_shimane["D6115"]
    # ids = [key for key in dict_shimane.keys() if (not np.isnan(dict_shimane[key]["気うつ"][0]))]
    health = []
    kiutu = []
    health_y = []
    kiutu_y = []
    # print([key for key in dict_shimane.keys()])
    for key in dict_shimane.keys():
        dict_shimane[key]["Kohs"] = [dict_shimane[key]["Kohs"]] 
    # exit()
    def threshold_func(x):
        return -0.02 * x**2 + 1.6 * x + 87.18
    for key in dict_shimane.keys():
        if np.isnan(dict_shimane[key][target][0]):
            continue
        if int(dict_shimane[key][target][0]) < threshold_func(int(dict_shimane[key]["Age"])):
            health.append(key)
            health_y.append(0)
        else:
            kiutu.append(key)
            kiutu_y.append(1)
    print(len(kiutu_y), len(health_y))
    # print()
    # exit()
    if len(kiutu_y) < len(health_y):
        ids = kiutu + random.sample(health, len(kiutu))
        ids_y = kiutu_y + health_y[:len(kiutu_y)]
    elif len(kiutu_y) > len(health_y):
        ids = health + random.sample(kiutu, len(health))
        ids_y = health_y + kiutu_y[:len(health_y)]    
    else:
        ids = kiutu + health
        ids_y = kiutu_y + health_y
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    ids, ids_y = list(map(np.array, [ids, ids_y]))
    for i, (traval_index, test_index) in enumerate(skf.split(ids, ids_y)):
        print(f"{i+1}/{n_splits} fold start")
        # print(traval_index, test_index)
        traval_ids = ids[traval_index]
        traval_ids_y = ids_y[traval_index]
        test_ids = ids[test_index]
        test_ids_y = ids_y[test_index]
        
        X_train, X_val, y_train, y_val = train_test_split(traval_ids, traval_ids_y, random_state=0, stratify=traval_ids_y)
        y_train, y_val, test_ids_y = list(map(list, [y_train, y_val, test_ids_y]))
        train(ids=[X_train, X_val, test_ids], ys=[y_train, y_val, test_ids_y], fold=i)
        # print(trval_id, test_id)
        

    # train()