import os
from os.path import join as osj
from collections import defaultdict
import pickle

import glob
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt
from tools import uncorrelated
from nilearn import datasets, plotting
from nilearn.plotting import plot_glass_brain, plot_stat_map
import nibabel as nib
import japanize_matplotlib


# root_path = "/gs/hs0/tga-akamalab/tokuhiro/Workspaces"
root_path = ".."
params = {
    "Kohs"       : {"mean" : 104.4, "std" : 19.16},
    "kana-hiroi" : {"mean" : 42.37, "std" : 10.36},
    "WCST RT"    : {"mean" : 202.2, "std" : 108.5}
}

def make_graph(save_name, target):
    global root_path
    global params
    y_tests = []
    predicts = []
    # print(glob.glob(osj(root_path, "cGCN_fMRI/save/11_25_regression/*")))
    # save_name = "12_13_regression_rt_200epoch"
    save_path = osj(root_path, f"save/{save_name}")
    for path in glob.glob(osj(save_path, "fold*.npz")):
        file = np.load(path)
        y_tests.extend(file["y_test"])
        predicts.extend(file["predict"].reshape(-1))
    print(len(predicts), len(y_tests))
    y_tests, predicts = map(np.array, [y_tests, predicts])
    y_tests = params[target]["mean"] + y_tests * params[target]["std"]
    predicts = params[target]["mean"] + predicts * params[target]["std"]

    r2 = r2_score(y_tests, predicts)
    print("r2", r2)
    r = np.corrcoef(y_tests, predicts)[0][1]
    print("correlation", r)
    p_value = uncorrelated(r, len(y_tests))
    print(p_value)

    df = pd.DataFrame({
        f"true {target} score" : y_tests,
        f"predict {target} score" : predicts})
    a, b = np.polyfit(y_tests, predicts, 1)
    fig = plt.figure()
    plt.scatter(y_tests, predicts)
    plt.xlabel(f"true {target} score")
    plt.ylabel(f"predict {target} score")
    plt.plot(np.linspace(40,130,1000), a*np.linspace(40,130,1000)+b, color="black")
    plt.text(-3.5,1.0,f"y={a:.2f}x{b:.2f}")
    # plt.text(-3.5,0.8,f"r={r:.2f} (p-value{uncorrelated(r, len(y_tests)):.2e})")
    if p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = ""
    plt.text(-3.5,0.8,f"r={r:.2f}{stars} p_value:{p_value:.2e}")
    plt.text(-3.5,0.6,f"r2={r2:.2f}")
    fig.savefig(osj(save_path, "regression_result.png"))
    boxdic = {
    "facecolor" : "white",
    "edgecolor" : "darkblue",
    "boxstyle" : "Round",
    # "fill" : False
    # "linewidth" : 2
}
    plt.close(fig)
    fig = plt.figure()
    sns.regplot(
        x=f"true {target} score", 
        y=f"predict {target} score", 
        data=df,
        color="black",
        )
    plt.text(
        0.14, 
        0.75, 
        # f"$y={a:.2f}x+{b:.2f}$\n$R={r:.2f}$\n$R^2={r2:.2f}$", 
        "$y= %.2f x+ %.2f $\n$R=%.2f^{**}}$\n$R^2=%.2f$" % (a, b, r ,r2), 
        size="small",
        transform=fig.transFigure, 
        bbox=boxdic)
    plt.savefig(osj(save_path, "regression_result_sns.png"))
    return [r2, r, p_value]

def classification():
    global root_path
    y_tests = []
    predicts = []
    predict_bin = []
    save_name = "12_7_classification_poly"
    save_path = osj(root_path, f"save/{save_name}")
    for path in glob.glob(osj(save_path, "fold*.npz")):
        # print(path)
        file = np.load(path)
        y_tests.extend(file["y_test"])
        predicts.extend(file["predict"])
        predict_bin.extend(file["predict_bin"].reshape(-1))
    print(len(predicts), len(y_tests), len(predict_bin))
    y_tests = [int(np.dot(np.array([0,1]), y_test))for y_test in y_tests]
    print(y_tests)
    print(predict_bin)
    # print(predicts)
    print(f"acc:{accuracy_score(y_tests, predict_bin):.2f}")

def occulusion(save_name, target, r2, r, p_value):
    mean = r
    atlas = datasets.fetch_atlas_aal()
    file = np.load(osj(root_path, f"save/{save_name}/corr_occulusion.npz"))
    y_predicts = file["y_predicts"]
    y_tests = file["y_tests"]

    kekka = []
    for roi_num, predict in enumerate(y_predicts):
        kekka.append(np.corrcoef(y_tests, predict)[0][1])
    difference = np.array(kekka) - mean
    print(atlas["labels"][list(difference).index(np.min(difference))])
    print(np.min(difference))
    print(list(difference).index(np.min(difference)))
    arg = np.argsort(difference)
    print(difference[31])
    print(arg)
    print(difference[arg])
    sort_rois = np.array(atlas["labels"])[arg]
    print(sort_rois)
    label_network = pd.read_csv("/mnt/Master_study/input/labels_and_networks_aal.csv", skiprows=2, header=None)
    fig = plt.figure(figsize=(20,10))
    plt.bar(
        list(range(len(sort_rois)))[0:60], 
        list(difference[arg])[0:30]+list(difference[arg])[-31:-1], 
        tick_label=list(sort_rois)[0:30] + list(sort_rois)[-31:-1]
        )
    plt.xticks(rotation=45)
    fig.savefig(osj(root_path, f"save/{save_name}/occulusion_result.png"))

    dic = defaultdict(str)
    for index, row in label_network.iterrows():
        dic[row[1]] = row[2]
    print([dic[roi_name] for roi_name in sort_rois])

    with open(osj(root_path, f"save/{save_name}/occulusion_result.txt"), "w") as f:
        f.write(f"correlation : {r}\nR2 : {r2}(p-value:{p_value})\n\n{difference[arg]}\n\n{sort_rois}\n\n{[dic[roi_name] for roi_name in sort_rois]}")
    df_result = pd.DataFrame({
        "rank" : range(1, 117),
        "r reduction" : difference[arg],
        "ROI" : sort_rois,
        "network" : [dic[roi_name] for roi_name in sort_rois]
    })
    df_result.to_csv(osj(root_path, f"save/{save_name}/occulusion_result.csv"))
    # nii.gzで保存
    indices = atlas["indices"]
    atlas_nii = nib.load(atlas["maps"])
    atlas_fdata = atlas_nii.get_fdata()
    affine = atlas_nii.affine
    header = atlas_nii.header
    for i, index in enumerate(indices):
        atlas_fdata[atlas_fdata==int(index)] = difference[i]
    nib_img = nib.Nifti1Image(atlas_fdata, affine, header)
    nii_path  = osj(root_path, f"save/{save_name}/occulusion_result.nii.gz")
    nib.save(nib_img, nii_path)
    view = plotting.view_img_on_surf(nii_path, threshold='70%', surf_mesh='fsaverage')
    view.save_as_html(osj(root_path, f"save/{save_name}/occulusion_result.html"))
    plot_glass_brain(
        nii_path, 
        threshold=-0.01,
        plot_abs=False,
        colorbar=True,
        cmap="hsv",
        # cmap="jet",
        display_mode='lzry',
        output_file=osj(root_path, f"save/{save_name}/plot_glass.png")
    )
    plot_stat_map(
        nii_path, 
        threshold=-0.01,
        # plot_abs=False,
        colorbar=True,
        cmap="hsv",
        cut_coords=10,
        display_mode='mosaic',
        output_file=osj(root_path, f"save/{save_name}/plot_stat.png"),
        title=f"{target} correlation reduction"
    )

    # 辞書で保存
    # with open("/mnt/Master_study/input/AAL_label_network.pkl", "wb") as f:
    #     pickle.dump(dic, f)
    # with open(osj(main_path, "input/shimane/data_dict.pkl"), "rb") as f:
    #     dict_shimane = pickle.load(f)


if __name__ == "__main__":
    save_names = [
        "12_16_regression_Kohs_200epoch", 
        "12_13_regression_kana_200epoch", 
        "12_13_regression_rt_200epoch"
        ]
    targets = [
        "Kohs",
        "kana-hiroi",
        "WCST RT"
        ]
    for save_name, target in zip(save_names, targets):
        r2, r, p_value = make_graph(save_name, target)
        # classification()
        occulusion(save_name, target, r2, r, p_value)
