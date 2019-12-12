ssh -X ubuntu@192.168.1.135
ssh -X ubuntu@192.168.99.152
yes


password: smecorner



export PATH="/home/ubuntu/anaconda3/bin:$PATH"
#source conda_bash
#conda create -n tarun python=3.6

#jupyter notebook --ip=127.0.0.1 --no-browser --allow-root

jupyter notebook --ip=192.168.99.152 --no-browser --allow-root




source activate tarun


#transfer file
scp check.txt ubuntu@192.168.1.135:~/tarun 

#transfer file
scp -r check ubuntu@192.168.1.135:~/tarun 

scp check.txt tarun.bhavnani@dev.smecorner.com@192.168.1.200:~/Desktop/bit



pip install numba
conda install cudatoolkit
conda install numba cudatoolkit pyculib

nvidia-smi
#to check


# install the tensorflow-gpur
pip3 install tensorflow-gpu
# Make sure the package is up to date
pip3 install --upgrade tensorflow-gpu

# enter python 
python
# check tensorflow
>>>import tensorflow as tf
>>>from tensorflow.python.client import device_lib
>>>print(device_lib.list_local_devices())
