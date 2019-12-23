
bash conda 5.2

yum install updates

yum group install "Development Tools"

conda install pyopengl



pip install msgpack
pip install service_identity


#conda environments on jupyter
conda install nb_conda




#install requirements
pip install -r req.txt#but this stops even if one package doesnt install

cat req.txt | xargs -n 1 pip install
#this takes one at a time and installs, if cant moves forward to next.


pip install 'prompt_toolkit==1.0.14'
pip install 'ipykernel<5.0.0'
