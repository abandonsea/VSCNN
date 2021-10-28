# VSCNN
This is an implementation of the VSCNN network presented in the paper "Hyperspectral Image Classification of Convolutional Neural Network Combined with Valuable Samples".
The original implementation of the network is available in this [GitHub repository](https://github.com/ShuGuoJ/3DVSCNN).

# Tested with
Python 3.8 and 3.9

Pytorch 1.9.0  

CPU and GPU

# Run the VSCNN
Please set your parameters in train.py or test.py before running them. 

To train, run:
```bash
# Trains network multiple times (see parameters in file)
python train.py
``` 

To test, run:
```bash
# Tests all runs saved in a given directory
python test.py
```

# About datasets
The datasets are available [here](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes).
The used datasets for this implementation are: PaviaU, Indian Pines and Salinas.

# Config file
Please refer to the config file `config.yaml` for details about the possible configurations of the network/training/testing.

# Authorship disclaimer
While I did write/review/modify alone the entire code, many parts of the code are heavily based on the [original implementation](https://github.com/ShuGuoJ/3DVSCNN) and the [S-DMM implementation](https://github.com/ShuGuoJ/S-DMM).
