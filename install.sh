#!/bin/bash
conda install -c pytorch pytorch=1.5 torchvision cudatoolkit=10.1 -y
conda install -c conda-forge -c fvcore fvcore -y
conda install pytorch3d -c pytorch3d -y
conda install pillow=6.0.0 -y
conda install numpy scipy -y
conda install shapely rtree graph-tool pyembree -c conda-forge  -y
conda install -c conda-forge scikit-image -y
conda install pip -y
pip install visdom dominate yattag
pip install trimesh[all]
conda install -c menpo opencv3 -y
conda install -c conda-forge gtk2 -y
conda install vim htop  -c conda-forge -y
conda install h5py -y
pip install pyrender
pip install yacs
pip intall tensorboardX,loguru
#python setup.py build_ext --inplace
#python setup_im2mesh.py build_ext --inplace
