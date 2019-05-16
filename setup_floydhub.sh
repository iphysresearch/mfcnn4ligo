#!/bin/bash
# The script that help make my code ready
# for fun in Floydhub.com

# update pip
pip install --upgrade pip


# install Mxnet for GPU
pip install -U --pre mxnet-cu92
sudo ldconfig /usr/local/cuda-9.2/lib64

# Others
pip install pyinstrument
pip install loguru
pip install tqdm


wget 
source /root/.bashrc
pip install gwosc