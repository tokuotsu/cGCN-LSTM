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
import copy

import h5py
# import tqdm
import pandas as pd
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
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
# from model_regression_age import *
from utils import save_logs_models
from config import MAIN_PATH_CGCN, MAIN_PATH_MASTER

# ROI_N = 236
gpu_count = 1
epochs = 100
ROI_N = 116
frames = 140
# graph_path = "graph/FC_aug_full_large.npy"
aug_num = 1 # データ数を何倍にするか、1-30 1だと元のROIのタイムコース
target = "Age" # 気うつ Age かなひろい Kohs(立方体組み合わせテスト)
threshold = 100 # クラス分類での正負を分ける閾値
save_name = "01_30_regression_age_rgout_200epoch" # test結果を保存するディレクトリ、save以下

def val(ids, ys, fold, dict_data, save_name, is_network=False):
    id_train, id_val, id_test = ids
    y_train, y_val, y_test = ys
    x_train, x_val, x_test = [], [], []

    for id_te in id_test:
        if aug_num == 1:
            x_test.append(dict_data[id_te]["timeseries"])
        else:
            for i in range(aug_num):
                x_test.append(dict_data[id_te][f"time_series_{i}"])
            y_test.extend([dict_data[id_te][target][0]]*aug_num)
    
    x_test = np.expand_dims(x_test, -1)
    print(x_test.shape)

    k = 5
    l2_reg = 1e-4
    lr = 1e-5
    batch_size = 48
    num_classes=2
    # TODO: foldによって変える。
    tmp_name = os.path.join(MAIN_PATH_CGCN, f"save/{save_name}/model_avg_edgeconv_k_5_l2_0.0001_dp_0.5_fold_{fold}_acc_0.0.hdf5")
    model_best = get_model(
        graph_path=os.path.join(MAIN_PATH_CGCN, f"graph/{save_name}/FC_graph{fold}.npy"), 
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
                metrics=['accuracy'])

    # kekka_corr = []
    # y_predicts = np.array([])
    # y_tests = np.array([])
    if not is_network:
        y_predicts = [[] for _ in range(ROI_N)]
        for i in range(ROI_N):
            x_test_tmp = copy.deepcopy(x_test)
            x_test_tmp[:,:,i,:] = 0
            predict = model_best.predict(x=x_test_tmp, batch_size=batch_size, verbose=1)
            # print(predict.shape)
            # print(np.array(y_test).shape)
            y_predicts[i] = list(predict.reshape(-1))
    else:
        y_predicts = np.zeros((ROI_N, ROI_N, x_test.shape[0]))
        count = 0
        for i in range(ROI_N):
            for j in range(ROI_N):
                if j > i:
                    continue
                count+=1
                print(f"{count}/{116*115/2}")
                x_test_tmp = copy.deepcopy(x_test)
                x_test_tmp[:,:,i,:] = 0
                x_test_tmp[:,:,j,:] = 0
                predict = model_best.predict(x=x_test_tmp, batch_size=batch_size, verbose=1)
                # print(predict.shape)
                # print(np.array(y_test).shape)
                y_predicts[i, j, :] = predict.reshape(-1)



    return [y_predicts, y_test]





def main(target, save_name):
    is_network = True
    main_path = MAIN_PATH_MASTER
    with open(osj(main_path, "heavy_files/input/shimane/data_dict_augumentation_kohs_large.pkl"), "rb") as f:
        dict_shimane = pickle.load(f)
    with open(osj(main_path, "input/shimane/subject_info.pkl"), "rb") as f:
        dict_subject = pickle.load(f)

    health = []
    kiutu = []
    health_y = []
    kiutu_y = []
    for key in dict_subject.keys():
        dict_subject[key][target] = [dict_subject[key][target]] 

    ids, ids_y = [], []
    for key in dict_subject.keys():
        if np.isnan(dict_subject[key][target][0]):
            continue
        ids.append(key)
        ids_y.append(dict_subject[key][target][0])
    ids_y = np.array(ids_y)
    ids_y = (ids_y - np.mean(ids_y))/np.std(ids_y)
    n_splits = 5
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    ids, ids_y = map(np.array, [ids, ids_y])
    y_tests = []
    if not is_network:
        y_predicts = np.array([[] for _ in range(ROI_N)])
        for i, (traval_index, test_index) in enumerate(skf.split(ids, ids_y)):
            print(f"{i+1}/{n_splits} fold start")
            # print(traval_index, test_index)
            traval_ids = ids[traval_index]
            traval_ids_y = ids_y[traval_index]
            test_ids = ids[test_index]
            test_ids_y = ids_y[test_index]
            
            X_train, X_val, y_train, y_val = train_test_split(traval_ids, traval_ids_y, random_state=0)
            y_train, y_val, test_ids_y = map(list, [y_train, y_val, test_ids_y])
            y_predict, y_test = val(
                ids=[X_train, X_val, test_ids], 
                ys=[y_train, y_val, test_ids_y], 
                fold=i,
                dict_data=dict_shimane,
                save_name=save_name
                )
            y_predict = np.array(y_predict)
            y_predicts = np.hstack([y_predicts, y_predict])
            y_tests.extend(y_test)
    else:
        y_predicts = np.zeros((116,116,0))
        for i, (traval_index, test_index) in enumerate(skf.split(ids, ids_y)):
            print(f"{i+1}/{n_splits} fold start")
            # print(traval_index, test_index)
            traval_ids = ids[traval_index]
            traval_ids_y = ids_y[traval_index]
            test_ids = ids[test_index]
            test_ids_y = ids_y[test_index]
            
            X_train, X_val, y_train, y_val = train_test_split(traval_ids, traval_ids_y, random_state=0)
            y_train, y_val, test_ids_y = map(list, [y_train, y_val, test_ids_y])
            y_predict, y_test = val(
                ids=[X_train, X_val, test_ids], 
                ys=[y_train, y_val, test_ids_y], 
                fold=i,
                dict_data=dict_shimane,
                save_name=save_name,
                is_network=is_network
                )
            y_predict = np.array(y_predict)
            y_predicts = np.concatenate([y_predicts, y_predict], axis=-1)
            y_tests.extend(y_test)


    # print(all_results)
    print(y_predicts.shape)
    # print(y_tests.shape)
    if not is_network:
        np.savez(os.path.join(MAIN_PATH_CGCN, f"save/{save_name}/corr_occulusion"), y_predicts=np.array(y_predicts), y_tests=np.array(y_tests))
    else:
        np.savez(os.path.join(MAIN_PATH_CGCN, f"save/{save_name}/corr_occulusion_net"), y_predicts=np.array(y_predicts), y_tests=np.array(y_tests))

if __name__=="__main__":
    targets = [
    # "Kohs(立方体組み合わせテスト)", 
    # "かなひろい", 
    "Age"
    ]
    save_names = [
        # "12_16_regression_Kohs_200epoch",
        # "12_13_regression_kana_200epoch",
        "01_30_regression_age_rgout_200epoch"
    ]
    for target, save_name in zip(targets, save_names):
        main(target, save_name)