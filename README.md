# SCTR


## Requirements
* pytorch1.7
* torchio<=0.18.20
* python>=3.6


## Training
* without pretrained-model
```
set hparam.train_or_test to 'train'
python main.py
```
* with pretrained-model
```
set hparam.train_or_test to 'train'
python main.py -k True
```
  
## Inference
* testing
```
set hparam.train_or_test to 'test'
python main.py
```
