# Intoduction

These are experiments for the paper "MARINA: Faster Non-Convex Distributed Learning with Compression" by Eduard Gorbunov, Konstantin Burlachenko, Zhize Li, Peter Richtarik. The paper is accepted for presentation and publication to Thirty-eighth International Conference on Machine Learning (ICML) 2021.

# Reference to the paper
https://arxiv.org/abs/2102.07845 - MARINA paper

# ResNet-18 @ CIFAR100 experiments

## Prepare environment for the experiments 

```bash
conda create -n marina python=3.9.1 -y
conda activate marina
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c conda-forge -y
conda install jupyter matplotlib numpy psutil -y
```

## Description

Please use sources from the following folder for experiments with Neural Nets: [neural_nets_experiments](neural_nets_experiments)
Each script corresponds to a specific algorithm with a specific compressor. Each script produces two main outputs:
- Output in a standard output text stream with various log information
- Serialized binary files with need statistics to display results.

Example of the command line to launch one of the scripts:

```bash
cd ./neural_nets_experiments 
nohup python nn_experiments_parallel_vr_diana_no_compr.py > nn_experiments_parallel_vr_diana_no_compr.txt &
```

After obtaining binary files to generate plots, please use [neural_nets_experiments/show.py](neural_nets_experiments/show.py) script. It should be launched via passing all binary files in a command line to visualize the experiment results and gather information from the experimental results. Example of the command line:
```bash
python show.py experiment_vr_marina_K_100000.bin experiment_vr_marina_K_500000.bin > info.txt
```

# Experiments for binary classification with non-convex loss

## Prepare environment for the experiments
Please install for your Python shell the following libraries via your package manager: matplotlib, numpy, psutil, mpi4py. The python library [mpi4py](https://mpi4py.readthedocs.io/en/stable/index.html) is an MPI wrapper to real Message Passing Interface(MPI) implementation. If your target OS is from the Windows family, we recommend using standard MPI implementation from Microsoft ([Microsoft-MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)). If your target OS for the experiment is from Linux/MacOS we recommend using **MPICH** or **openmpi-bin** package. 

## Description

In that experiment, we carried non-convex linear regression with MPI4PY. Launch scripts are in [linear_model_with_non_convex_loss/linux](linear_model_with_non_convex_loss/linux). The script produces two main outputs:
- Output in a standard output text stream with various log information
- Serialized binary files with need statistics to display results.

After obtaining binary files to generate plots, please use [linear_model_with_non_convex_loss/show.py](linear_model_with_non_convex_loss/show.py) script. It should be launched via passing all "bin" files in a command line. Example of the command line:
```bash
python show.py experiment_vr_marina_K_100000.bin experiment_vr_marina_K_500000.bin > info.txt
```

# Miscellaneous

## Custom logic for working with results of the experiments
You can write your custom logic. To deserialize the results of the experiment, please use the following code snippet:
```python
import utils 
import numpy as np
d = utils.deserialize("experiment.bin")
```
For a more detailed understanding of the structure of serialized experiments, you can look into the [linear_model_with_non_convex_loss/show.py](linear_model_with_non_convex_loss/show.py) script.
