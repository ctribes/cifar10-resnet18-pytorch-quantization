*****
# Optimizing quantized network hyperparameters
*****

### Goal
Tests neural network hyperparameters optimization (HPO) on cifar10 with quantized ResNet18 network from a full precision state.

### ResNet18
The ResNet18 from https://github.com/pytorch/vision/tree/master/torchvision/models has been modified to work with 10 classes, 32x32 images of Cifar10.

### Preliminary training, testing and quantization

1- trainFullPrecisionAndSaveState.py -> use a predefined set of hyperparameters to train a full precision ResNet18 on cifar10. Save the best network states for later.

2- loadPretrainedAndTestAccuracy.py -> load a pretrained full precision (FP) ResNet18 network state from a checkpoint and test the accuracy.

3- loadPretrainedAndTrainResNet.py -> load a pretrained FP network state from a checkpoint and train for a given number of epochs (save best states).

5- loadPretrainedFPQuantizeAndTest.py -> load a pretrained FP network state from a checkpoint. Quantize the network, test and save the quantized network state (1/4 the size of the FP one).

5- loadPretrainedFPQuantizeAndTrain.py -> load a pretrained FP network state from a checkpoint. Quantize the network, train it with a subset of hyperparameters (batch size, lr, weight decay, optimizer) and save the best network states. The quantized network saved states is 1/4 the size of the FP one.

### Optimization

* Hyperparameters tuning using Nomad 4 optimizer from https://github.com/orgs/bbopt/teams/nomad. Nomad 4 is a C++ code and requires building. 

* The objective is to find the values of hyperparameters that maximize the network accuracy.

* Nomad takes a parameter file as input to describe the optimization problem and the other Nomad setting. 

* For quantized network, only four (4) hyperparameters are considered: the choice of optimizer (in ["SGD", "Adadelta", "Adagrad", "Adam", "Adamax"], handled as an integer), weight decay of the neural network optimization (in [0, 0.00000001, 0.0000001, 0.000001], handled as an integer), the learning rate (in [10-4, 10-1] as a log uniform) and the batch size (in [32, 64, 128, 256], handled as an integer).

* Before lauchning blackbox evaluation, the variables handled by Nomad are converted in the hyperparameters space (bb.py).

* Nomad launches a single evaluations on given GPU (see eval.py)

* Nomad can launch several parallel evaluations on different GPUs (see evalP.py). The evalP.py manages the execution of the bb.py on each available GPU (list must be provided by user). To use this option, the param.txt file must be changed: 
  BB_EXE "$python evalP.py"
  GENERATE_ALL_POINTS_BEFORE_EVAL yes
  NB_THREADS_OPENMP 4
