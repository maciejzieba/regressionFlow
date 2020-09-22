# Synthetic CPI dataset

Code adapted from https://github.com/lmb-freiburg/Multimodal-Future-Prediction/tree/master/CPI to fit the SDD dataset format.

To generate the data with the settings analogous to the original authors, run **in this directory**:

* training:
```bash
python CPI-generate.py cpi/train 200000 3 20 20
```
* test:
```
python CPI-generate.py cpi/test 54 3 1000 20
```