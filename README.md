# BIOMED Seizure Detection Challenge

This repository contains the base code and helper functions for the BIOMED seizure detection challenge, in the context of ICASSP 2023.
For further information, please visit our [website](https://biomedepi.github.io/seizure_detection_challenge/).

The base code included in this repository was used to create the benchmark of **Task 2**. This task involves training the model implemented in the file "*ChronoNet.py*". For the challenge, participants are **required to use this arquitecture** and are **not allowed to change anything in this script**.


## Instalation

Our code is based on python 3.9 and the model was developed on tensorflow version 2.11 (using Keras). The required dependencies can be installed by running:

> `pip install -r requirements.txt`

## Contents

The participants are free to build their own frameworks. Usefull functions can be found in the "*routines.py*" and "*aux_functions.py*" scripts, mainly to:
- read or load the data (in this implementation, the data is first loaded offline - in *routines.split_sets* -  and called in the data generator (defined in the script "*generator_ds.py*"), which is then fed to the training routine).
- read the annotations (with aux_functions.wrangle_tsv)

The current implementation makes use of configuration files (the class is defined in "*config.py*"). You can see the configuration used in the benchmark defined in the "*main.py*" script. For submission, we will ask you to provide the code used for completing the task, together with the configurations used for training the model. It is not mandatory to use the same configuration files as in this repository.

The evaluation metrics are explained in the challenge website. The implementation of the metrics is based on a prediction vector (required to be sent when submitting the challenge) containing consecutive probabilities of a 2-second window to be a seizure. The functions *routines.perf_measure_epoch* and *routines.perf_measure_ovlp* calculate the metrics according to the EPOCH and any-overlap methods (for more information, check https://biomedepi.github.io/seizure_detection_challenge/regulations/). The final scoring is calculated with the function *routines.get_metrics_scoring*.
