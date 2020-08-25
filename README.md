# regressionFlow

The train and test data are located at:

https://lmb.informatik.uni-freiburg.de/resources/binaries/Multimodal_Future_Prediction/sdd_train.zip

https://lmb.informatik.uni-freiburg.de/resources/binaries/Multimodal_Future_Prediction/sdd_test.zip

The datasets should be located in `data\SDD\train` and `data\SDD\test` locations. 

It is essential to install install torchdiffeq
```
git clone https://github.com/rtqichen/torchdiffeq.git
cd torchdiffeq
pip install -e . 
```
You can run the training procedure with the script `train_regression_SDD.py` using the following settings:

```
--log_name
"experiment_regression_flow_SSD_test"
--lr
2e-5
--num_blocks
1
--batch_size
20
--zdim
1
--epochs
10000
--save_freq
1
--viz_freq
1
--log_freq
1
--val_freq
6000
--gpu
0
--batch_norm
True
--hyper_dims
16-32-64
--dims
128-128-128
--input_dim
2
```

Validation can be run using script `test_SDD` with parameters:

```
--resume_checkpoint
/home/maciej/PycharmProjects/PointFlow/checkpoints/experiment_regression_flow_SSD_test/checkpoint-39.pt
--lr
2e-4
--num_blocks
1
--batch_size
20
--zdim
1
--epochs
10000
--save_freq
1
--viz_freq
1
--log_freq
1
--val_freq
6000
--gpu
0
--batch_norm
True
--hyper_dims
16-32-64
--dims
128-128-128
--input_dim
2
```