import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import Lasso
from nilearn.connectome import ConnectivityMeasure
from tqdm import tqdm

path_dict = "/mnt/Master_study/input/shimane/subject_info.pkl"
with open(path_dict, "rb") as f:
    dic_tmp = pickle.load(f)
path_aug = "/mnt/Master_study/heavy_files/input/AAL_66%_10times_aug_dict.pkl"
with open(path_aug, "rb") as f:
    dict_aug = pickle.load(f)

df_result = pd.DataFrame()
target = "Kohs(立方体組み合わせテスト)"

for id in tqdm(dic_tmp):
    if "timeseries" not in dict_aug[id].keys():
        continue
    # print(dic_tmp[id])
    time_series = dict_aug[id]["timeseries"]
    # time_series[:, 0] = 0
    # print(time_series.shape)
    cm = ConnectivityMeasure(kind="correlation")
    train_FC_tmp = cm.fit_transform(time_series.reshape((1, 140, 116)))
    feature = np.sum(train_FC_tmp, axis=1) - 1
    feature = list(feature.reshape(-1))
    feature = [dic_tmp[id][target], int(dic_tmp[id]["Sex"]=="男"), dic_tmp[id]["Age"]] + feature
    # print(feature.shape)
    df_tmp = pd.DataFrame(
        np.array(feature).reshape(-1, len(feature)),
        columns=["target", "Sex", "Age"] + [f"feature_{i+1}" for i in range(116)],
        index=[id]
    )
    df_result = pd.concat([df_result, df_tmp], axis=0)

print(df_result)

    



n_splits = 10
N = 100
kf = KFold(n_splits=n_splits)
scaler = StandardScaler()
lambdas = np.logspace(-5, 0, num = N)
# X = df_data.iloc[:, 2:]
# y = df_data.gm_bhq
mean_list = []
std_list = []
sem_list = []
coef_list = []
for c in lambdas:
    mse_list = []
    for n_fold, (train_idx, test_idx) in enumerate(kf.split(df_result)):
        # print(f"========================== {n_fold + 1} / {n_splits} ==========================")
        # subID除去,trainとtestにsplit -> 標準化
        train_data = scaler.fit_transform(df_result.iloc[train_idx,1:])
        test_data = scaler.transform(df_result.iloc[test_idx,1:])
        X_train, y_train = train_data[:, 1:], train_data[:, 0]
        X_test, y_test = test_data[:, 1:], test_data[:, 0]
        reg = Lasso(alpha=c, fit_intercept=False, random_state = 42) #  data is centered.
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = MSE(y_test, y_pred)
        mse_list.append(mse)
        coef_list.append(reg.coef_)
    mean = np.mean(mse_list)
    std = np.std(mse_list)
    sem = std / np.sqrt(len(mse_list))
    mean_list.append(mean)
    std_list.append(std)
    sem_list.append(sem)
print(mean_list)
print(std_list)
print(sem_list)