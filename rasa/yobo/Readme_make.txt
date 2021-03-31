
## install

source activate rasa5
thi is on python 3.5



pip install 'prompt_toolkit==1.0.14'
pip install 'ipykernel<5.0.0'


python train_nlu.py
#python train_core.py

#token not found-----------pip install 'prompt_toolkit==1.0.14'


#get in interactive learning and train
python -m rasa_core.train interactive -o models/dialogue -d domain.yml -s data/stories.md --nlu models/nlu/default/latest_nlu --endpoints endpoints.yml

#run
python -m rasa_core.run --core models/dialogue --nlu models/nlu/default/latest_nlu --endpoints endpoints.yml








