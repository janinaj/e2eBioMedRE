# Named Entity Linking

This repository contains PyTorch code to train [ResCNN](https://aclanthology.org/2021.findings-emnlp.140.pdf) for Disease and Chemical-type entity linking, and to integrate the outputs from trained ResCNN with the NER+NEL outputs for other entity types from [PubTator 3.0](https://arxiv.org/pdf/2401.11048).

## Quick links

- [Named Entity Linking](#named-entity-linking)
	- [Quick links](#quick-links)
	- [1. Overview](#1-overview)
	- [2. Setup](#2-setup)
		- [2.1. Environment details](#21-environment-details)
		- [2.2. Installation](#22-installation)
	- [3. Reproducing results](#3-reproducing-results)
	- [3.1. Proprocessing data before training](#31-proprocessing-data-before-training)
	- [3.2. ResCNN training and inference](#32-rescnn-training-and-inference)
	- [3.3. Combine predictions and Evaluate](#33-combine-predictions-and-evaluate)
	- [4. Data](#4-data)

## 1. Overview

For entity linking (EL), we use a hybrid approach that includes a convolutional neural network (CNN) for disease and chemicals, and uses an external entity linker, PubTator 3.0, for other entity types.

## 2. Setup

Note that this repository is mostly based on [ResCNN](https://github.com/laituan245/rescnn_bioel), only with some task-specific modifications.

### 2.1. Environment details

* Python version: 3.8
* GPU: Tesla V100-32
* Note: In order to to reproduce the exact results, you also need to use the same GPU. We have verified that the model will be different if a different GPU is used (even if the random state and hyperparameters are kept constant).

### 2.2. Installation

We recommend to set your own virtual environment, either with conda or pip. Then, install the require packages. Below is the commands for the installation of virtual environment.

```bash
conda create -n rescnn python=3.8
conda activate rescnn
conda install pip
pip install -r requirements.txt
```

Also, you need to place the data for NER, RE, and ND in the upper level of directory for nel folder to replicate the whole commands for NEL without facing errors.

## 3. Reproducing results

## 3.1. Proprocessing data before training
To process the given dataset to train ResCNN, we follow the preprocessing method of [BioSyn](https://github.com/dmis-lab/BioSyn), which was suggested from the ResCNN repository. The processed input queries and ontologies are in the *./resources/bc8biored-disease|chemical-aio* and *./resources/ontologies* folder each. We use the *preprocess_bc8biored_for_rescnn.ipynb* notebook file to prepare input queries and ontologies for ResCNN training. 

Simply put, you can easily use the processed input files and ontologies for ResCNN training.

## 3.2. ResCNN training and inference
Before replicating the training codes, you need to download pretrained embedding file and place it following the directions below:

- Download [pretrained biolinkbert embedding](https://drive.google.com/file/d/1x9F4UfpP9RTC0XA2htfKBXjtl272WpUH/view?usp=share_link) and place them in the folder *./outputs/rescnn/initial_embeddings/biolinkbert-base*.
<!-- - If you want to use the best fine-tuned model, download them using this [link] and set their directory in the training codes. -->

To run the model training and generate predictions based on the trained models for Disease and Chemical entities. run the following command:

```bash
./train_rescnn.sh
```

Predictions and best fine-tuned models are saved in *outputs/final* folder.

<!-- We also included a copy of the trained NER model (*models/pretrained_model/*). To regenerate the test data results using this model, run:

```
./regenerate_results_from_trained_model.sh
``` -->

## 3.3. Combine predictions and Evaluate

After you train the ResCNN model and generate predictions, you need to combine the ResCNN outputs for Disease and Chemical entities with the other types predicted with PubTator 3.0. The prediction file from PubTator 3.0 is already constructed by using PubTator API, and is placed in the folder *./outputs/pubtator*. 

To evaluate the NEL task and integrate the different prediction files for RE input, you need to run all the blocks the following Jupyter Notebook file.

'''bash
./align_and_eval_prediction.ipynb
'''

The integrated prediction files are saved in *outputs/final* folder.

NOTE: To map predicted outputs of ResCNN, you need to generate your own mapping file which links your original form of predicted mentions to the input form of ResCNN. This is because ResCNN follows the protocol of BioSyn to normalize original mentions for training inputs. Now, we placed our mapping files in the *./resources/data/* based on our predicted mentions of NER model. But if you trained your own NER model which would make you have different NER predictions with ours, then you should generate your own mapping file by using [BioSyn](https://github.com/dmis-lab/BioSyn) module.

## 4. Data

We mainly use the shared task dataset, but we also utilize the following datasets and vocabularies:

- To build ontologies for ResCNN, we use
  - [CTD Database](https://ctdbase.org)
    - Disease vocabulaury (MEDIC)
    - Chemical vocabulary (MeSH)

- To extend the original training dataset with external dataset related with Disease and Chemical entities, we add following datasets for our training set.
  - NCBI Disease dataset
  - BC5CDR dataset

