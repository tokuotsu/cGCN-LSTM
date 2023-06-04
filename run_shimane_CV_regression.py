# -*- coding: utf-8 -*-
import os
from os.path import join as osj
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
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
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from nilearn.connectome import ConnectivityMeasure
from keras.layers import BatchNormalization, Dropout, Conv2D, TimeDistributed
from keras.layers import Lambda, Flatten, Activation, Dense, Input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, list_pictures
from keras.regularizers import l2
from keras.utils.training_utils import multi_gpu_model
from keras import optimizers
import keras.backend as K
import matplotlib.pyplot as plt
from time import gmtime, strftime, time, localtime

from model_regression import *
from utils import save_logs_models
from config import MAIN_PATH_CGCN, MAIN_PATH_MASTER

# ROI_N = 236
gpu_count = 1
epochs = 200
ROI_N = 116
frames = 140
# graph_path = "graph/FC_aug_full_large.npy"
aug_num = 1 # データ数を何倍にするか、1-30 1だと元のROIのタイムコース
# target = "Kohs(立方体組み合わせテスト)" # Kohs 気うつ Age
threshold = 100 # クラス分類での正負を分ける閾値
# save_name = "12_12_regression_k=10_kohs" # test結果を保存するディレクトリ、save以下
# print(f"save_name: {save_name}")

# ########################################## Load data ########################################
# Download the data from https://drive.google.com/file/d/1l029ZuOIUY5gehBZCAyHaJqMNuxRHTFc/view?usp=sharing
# with h5py.File('HCP.h5', 'r') as f:
#     x_train, x_val, x_test = f['x_train'][()], f['x_val'][()], f['x_test'][()]
#     y_train, y_val, y_test = f['y_train'][()], f['y_val'][()], f['y_test'][()]


# x_train = x_train.transpose((1,0,2))
# x_val = x_val.transpose((1,0,2))
# x_test = x_test.transpose((1,0,2))

############################

def train(ids, ys, fold, dict_data, save_name):
    id_train, id_val, id_test = ids
    y_train, y_val, y_test = ys
    x_train, x_val, x_test = [], [], []
    for id_tr in id_train:
        # print(id_tr)
        if aug_num == 1:
            ts = dict_data[id_tr]["timeseries"]
            # ts = (ts - np.mean(ts, axis=0).reshape(1,ROI_N))/np.std(ts, axis=0).reshape(1,ROI_N)
            x_train.append(ts)
        
    for id_va in id_val:
        if aug_num == 1:
            ts = dict_data[id_va]["timeseries"]
            # ts = (ts - np.mean(ts, axis=0).reshape(1,ROI_N))/np.std(ts, axis=0).reshape(1,ROI_N)
            x_val.append(ts)
        
    for id_te in id_test:
        if aug_num == 1:
            ts = dict_data[id_te]["timeseries"]
            # ts = (ts - np.mean(ts, axis=0).reshape(1,ROI_N))/np.std(ts, axis=0).reshape(1,ROI_N)
            x_test.append(ts)
        
    # print("chance score")
    # print(np.array([np.sum(y_train), np.sum(y_val), np.sum(y_test)]) / np.array(list(map(len, [y_train, y_val, y_test]))))
    print("list(map(len, [y_train, y_val, y_test])): ", list(map(len, [y_train, y_val, y_test])))

    data = [x_train, x_val, x_test, y_train, y_val, y_test]

    x_train, x_val, x_test, y_train, y_val, y_test = list(map(np.array, data))    
    
    # グループFCを求める。
    cm = ConnectivityMeasure(kind="correlation")

    train_FC_tmp = cm.fit_transform(x_train)
    train_group_FC = np.mean(train_FC_tmp, axis=0)
    graph_folder = f"graph/{save_name}"
    print(graph_folder)
    os.makedirs(graph_folder, exist_ok=True)

    graph_path = osj(graph_folder, f"FC_graph{fold}.npy")
    np.save(graph_path, train_group_FC)
    
    ############################
    x_train = np.expand_dims(x_train, -1) # (200, 100, 236, 1)

    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    # Convert class vectors to binary class matrices.
    num_classes = 2


    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_val = keras.utils.to_categorical(y_val, num_classes)

    # y_test = keras.utils.to_categorical(y_test, num_classes)
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
    batch_size = 16 * gpu_count
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
    model_.summary(line_length = 120)


    ######################################## Training ####################################################
    if gpu_count > 1:
        model = multi_gpu_model(model_, gpus=gpu_count)
        print("multi_gpu_model loaded.")
    else:
        model = model_
        print("single_gpu_model loaded")

    model.compile(loss=['mean_squared_error'], 
                optimizer=optimizers.Adam(lr=lr),
                metrics=["accuracy"])

    print('Train...')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                patience=30, min_lr=1e-6)
    lr_hist = []
    class Lr_record(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs={}):
            tmp = K.eval(model.optimizer.lr)
            lr_hist.append(tmp)
            print('Ir:', tmp)
    lr_record = Lr_record()
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    class multi_gpu_checkpointer(keras.callbacks.ModelCheckpoint):
        def set_model(self, model):
            self.model = model_
    checkpointer = multi_gpu_checkpointer(monitor='val_loss', filepath=tmp_name, 
                                                    verbose=1, save_best_only=True)

    model_history = model.fit(x_train, y_train,
                                shuffle=True,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(x_val, y_val),
                                callbacks=[checkpointer, lr_record, reduce_lr, earlystop])

    # multi_gpuでは、multi_gpu_model()に入れる前のモデルを保存する。（重みは共有している）

    print('validation...')

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

    model_best.compile(loss=['mean_squared_error'], 
                optimizer=optimizers.Adam(lr=lr),
                metrics=["accuracy"])

    if gpu_count > 1:
        batch_size = int(batch_size / gpu_count)

    val_tmp = model_best.evaluate(x=x_val, y=y_val, batch_size=batch_size, verbose=1)
    print('validation:', val_tmp)

    test_tmp = model_best.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
    print('test:', test_tmp)

    predict = model_best.predict(x=x_test, batch_size=batch_size, verbose=1)
    # predict_bin = np.argmax(predict, axis=1)
    print('test:', test_tmp)
    
    save_dir = osj(".", "save", save_name)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(osj(save_dir, f"fold_{fold+1}_test"), y_test=y_test, predict=predict)

    ######################################## save log and model #######################################

    save_logs_models(model, model_history, acc=val_tmp[1],
        folder=f'save/{save_name}/', 
        lr_hist=lr_hist, file_name=file_name, loss_name='loss', 
        acc_name='acc', tmp_name=tmp_name)

def train_target(target, save_name):
    main_path = MAIN_PATH_MASTER
    with open(osj(main_path, "heavy_files/input/shimane/data_dict_augumentation_kohs_large.pkl"), "rb") as f:
        dict_shimane = pickle.load(f)
    with open(osj(main_path, "input/shimane/subject_info.pkl"), "rb") as f:
        dict_subject = pickle.load(f)

    health = []
    kiutu = []
    health_y = []
    kiutu_y = []

    for key in dict_shimane.keys():
        dict_subject[key][target] = [dict_subject[key][target]] 

    ids, ids_y = [], []
    age, sex = [], []
    for key in dict_shimane.keys():
        if np.isnan(dict_subject[key][target][0]):
            continue
        # if dict_shimane[key][target][0] == 124:
        #     continue
        ids.append(key)
        ids_y.append(dict_subject[key][target][0])
        age.append(dict_subject[key]["Age"])
        sex.append(int(dict_subject[key]["Sex"]=="男"))
    age, sex = map(np.array, [age, sex])
    age = (age - np.mean(age))/np.std(age)
    sex = (sex - np.mean(sex))/np.std(sex)
    
    ids_y = np.array(ids_y)
    # ids_y = (ids_y - np.mean(ids_y))/np.std(ids_y)
    # ids_y = ids_y - (-0.59056619 * age + 0.2135483 * sex)

    # 反応時間を正規分布に
    if target == "反応時間1":
        from sklearn.preprocessing import PowerTransformer
        ids_y = ids_y.reshape(len(ids_y), -1)
        pt = PowerTransformer()
        pt.fit(ids_y)
        ids_y = pt.transform(ids_y)
    else:
        ids_y = (ids_y - np.mean(ids_y))/np.std(ids_y)
    n_splits = 5
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    ids, ids_y = map(np.array, [ids, ids_y])
    for i, (traval_index, test_index) in enumerate(skf.split(ids, ids_y)):
        print(f"{i+1}/{n_splits} fold start")
        # print(traval_index, test_index)
        traval_ids = ids[traval_index]
        traval_ids_y = ids_y[traval_index]
        test_ids = ids[test_index]
        test_ids_y = ids_y[test_index]
        
        x_train, X_val, y_train, y_val = train_test_split(traval_ids, traval_ids_y, random_state=0)
        y_train, y_val, test_ids_y = map(list, [y_train, y_val, test_ids_y])
        train(
            ids=[x_train, X_val, test_ids], 
            ys=[y_train, y_val, test_ids_y], 
            fold=i, 
            dict_data=dict_shimane,
            save_name=save_name
            )
        # print(trval_id, test_id)
    # train()

if __name__ == "__main__":
    # train_target("Kohs(立方体組み合わせテスト)")
    # target_list = ["かなひろい", "反応時間1"]
    # target_list = ["岡部"]
    # target_list = ["Kohs(立方体組み合わせテスト)"]
    target_list = ["Age"]
    # save_label = ["kana", "rt"]
    save_label = ["age"]
    for i, target in enumerate(target_list):
        print(f"{target} start")
        train_target(target, save_name=f"03_24_regression_{save_label[i]}_rgout_200epoch")