# regressionFlow

## Installation 

It is essential to install install torchdiffeq
```
git clone https://github.com/rtqichen/torchdiffeq.git
cd torchdiffeq
pip install -e . 
```

## Toy example 

The toy example provided in the paper can be run with the script:

```angular2html
python train_regression_SDD.py --log_name experiment_regression_flow_toy
--lr 1e-3 --num_blocks 1 --batch_size 1000 --epochs
1000 --save_freq 5 --viz_freq 1 --log_freq 1 --gpu 0 --hyper_dims
128-32 --dims 32-32-32 --input_dim 1 --weight_decay 1e-5
```

The script saves the model with `--save_freq` and visualize the results with `viz_freq` frequencies.

The model can be loaded and tested using:

```angular2html
python test_toy.py --resume_checkpoint /experiment_regression_flow_toy/experiment_regression_flow_toy/checkpoint-latest.pt 
--num_blocks 1 --gpu 0 --hyper_dims 128-32 
--dims 32-32-32 --input_dim 1 
```

## Stanford Drone Dataset (SDD) 

The train and test data are located at:

https://lmb.informatik.uni-freiburg.de/resources/binaries/Multimodal_Future_Prediction/sdd_train.zip

https://lmb.informatik.uni-freiburg.de/resources/binaries/Multimodal_Future_Prediction/sdd_test.zip

The datasets should be located in `data\SDD\train` and `data\SDD\test` locations. 

You can run the training procedure with the script `train_regression_SDD.py` using the following settings:

```
train_regression_SDD.py --log_name "experiment_regression_flow_SSD"
--lr 2e-5 --num_blocks 1 --batch_size 20 --epochs 100 --save_freq 1
--viz_freq 1 --log_freq 1 --gpu 0 --dims 128-128-128 --input_dim 2
```

Validation can be run using script `test_SDD` with parameters:

```
python test_SDD.py --data_dir /data/SDD 
--resume_checkpoint /experiment_regression_flow_SSD/checkpoint-latest.pt 
--num_blocks 1 --gpu 0 --dims 128-128-128 --input_dim 2
```

## NGSIM  Dataset

First, the data should be obtained from:

https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj

```angular2html
Specifically you will need these files:
US-101:
'0750am-0805am/trajectories-0750am-0805am.txt'
'0805am-0820am/trajectories-0805am-0820am.txt'
'0820am-0835am/trajectories-0820am-0835am.txt'

I-80:
'0400pm-0415pm/trajectories-0400-0415.txt'
'0500pm-0515pm/trajectories-0500-0515.txt'
'0515pm-0530pm/trajectories-0515-0530.txt'
```

The files should be further processed using `prepocess_data.m` from:

https://github.com/nachiket92/conv-social-pooling

After processing files `TrainSet.mat`, `ValSet.mat`, and `TestSet.mat` should be created.  

The model for this dataset can be trained with the following script:

```angular2html
python networks_regression_NGSIM.py 
--log_name experiment_regression_flow_NGSIM
--lr 1e-3 --num_blocks 1 --batch_size 10000
--epochs 100 --save_freq 1 --viz_freq 1 --log_freq
1 --val_freq 6000 --gpu 0 --dims 16-16-16 --input_dim
2 --weight_decay 1e-5 --data_dir /data/files/location
```

The evaluation for the dataset can be run using:

```angular2html
python test_NGSIM.py --data_dir/data/files/location
--resume_checkpoint
/experiment_regression_flow_NGSIM/checkpoint-latest.pt
--num_blocks 1 --gpu 0 --dims 16-16-16
--input_dim 2
```