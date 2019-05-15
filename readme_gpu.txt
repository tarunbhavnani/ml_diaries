ssh -X ubuntu@192.168.1.135
yes


password: snecorner

#source conda_bash
#conda create -n tarun python=3.6

source activate tarun

pip install numba
conda install cudatoolkit
conda install numba cudatoolkit pyculib

nvidia-smi
#to check


# install the tensorflow-gpu
pip3 install tensorflow-gpu
# Make sure the package is up to date
pip3 install --upgrade tensorflow-gpu

# enter python 
python
# check tensorflow
>>>import tensorflow as tf
>>>from tensorflow.python.client import device_lib
>>>print(device_lib.list_local_devices())
