# cGCN-LSTM
## 元論文情報
* [元論文](https://www.mitpressjournals.org/doi/pdf/10.1162/netn_a_00171)
* [github](https://github.com/Lebo-Wang/cGCN_fMRI)

## environments

* TSUBAME
```bash
iqrsh -l h_rt=23:59:59

module load python/3.6.5
python -m venv [環境名]
python -m pip install --upgrade pip setuptools
pip install -r requirement.txt

# config.pyのMAIN_PATH_CGCN, MAIN_PATH_MASTERをTSUBAMEの方に設定。
python run_shimane_CV.py # classification
python run_shiimane_CV_regression.py # regression
python occlusion.py # occlusion
python scripts/analysis.py
```

* akamalab-DL-ubnutu
```bash
# docker pull nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04 # pull済み
docker run -it -d --mount type=bind,source=/home/akamalab_123/Workspaces,destination=/mnt --gpus all --name [container_name] nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04 bash
docker exec -it [container_name] bash

#################################################################
# 以下は、
source /mnt/install.sh
# (/home/akamalab_123/Workspaces/install.shにバインドしている)
# でも可
#################################################################

apt-get update
apt-get install -y git curl wget pkg-config zip \
g++ zlib1g-dev libssl-dev libbz2-dev

# pyenv
git clone https://github.com/yyuu/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
source ~/.bash_profile
pyenv install 3.6.4
pyenv global 3.6.4
python -m pip install --upgrade pip setuptools
cd /home/
mkdir Workspace
cd Workspace
git clone https://github.com/tokuotsu/cGCN-LSTM.git
cd cGCN-LSTM
pip install -r requirement.txt

# config.pyのMAIN_PATH_CGCN, MAIN_PATH_MASTERをakamalab-DL-ubuntuの方に設定。
python run_shimane_CV.py # classification
python run_shiimane_CV_regression.py # regression
python occlusion.py # occlusion
python scripts/analysis.py
```

## directory
│  .gitignore
│  config.py
│  model.py
│  model_regression.py
│  model_regression_age.py
│  occlusion.py
│  README.md
│  requirement.txt
│  run_shimane_CV.py
│  run_shimane_CV_regression.py
│  utils.py
│
├─graph
├─save
└─scripts
        analysis.py
        lasso.py