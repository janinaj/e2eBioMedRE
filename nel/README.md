# Named Entity Linking

This repository contains PyTorch code to train [ResCNN](https://aclanthology.org/2021.findings-emnlp.140.pdf) for Disease and Chemical-type entity linking, and to integrate the outputs from trained ResCNN with the NER+NEL outputs for other entity types from [PubTator 3.0](https://arxiv.org/pdf/2401.11048).

## Quick links

- [Named Entity Linking](#named-entity-linking)
	- [Quick links](#quick-links)
	- [Overview](#overview)
	- [Setup](#setup)
		- [Environment details](#environment-details)
		- [Installation](#installation)
	- [Reproducing results](#reproducing-results)
	- [Data](#data)

## Overview

For entity linking (EL), we use a hybrid approach that includes a convolutional neural network (CNN) for disease and chemicals, and uses an external entity linker, PubTator 3.0, for other entity types.

## Setup

Note that this repository is mostly based on [ResCNN](https://github.com/laituan245/rescnn_bioel), only with some task-specific modifications.

### Environment details

* Python version: 3.8
* GPU: Tesla V100-32
* Note: in order to to reproduce the exact results, you also need to use the same GPU. We have verified that the model will be different if a different GPU is used (even if the random state and hyperparameters are kept constant).

### Installation

We recommend to set your own virtual environment, either with conda or pip. Then, execute the command below:

```bash
    pip install -r requirements.txt
```

## Reproducing results

To process the given dataset to train ResCNN, we follow the preprocessing method of [BioSyn](https://github.com/dmis-lab/BioSyn), which was suggested from the ResCNN repository. The processed input queries and ontologies are in the *./resources* folder. We use the *preprocess_bc8biored_for_rescnn.ipynb* file to prepare input queries and ontologies for ResCNN training.

To run the model training and generate predictions based on the trained models for Disease and Chemical entities. run the following command:

```bash
./train_rescnn.sh
```

Predictions and trained models are saved in *outputs/final* folder.

<!-- We also included a copy of the trained NER model (*models/pretrained_model/*). To regenerate the test data results using this model, run:

```
./regenerate_results_from_trained_model.sh
``` -->

After you train the ResCNN model and generate predictions, you need to merge the ResCNN outputs for Disease and Chemical entities with the other types predicted with PubTator 3.0. The prediction file from PubTator 3.0 is in the folder *./outputs/pubtator*. To evaluate the NEL task and integrate the different prediction files for RE input, you need to run the following Jupyter Notebook file.

'''text
./align_and_eval_prediction.ipynb
'''

The integrated prediction files are saved in *outputs/final* folder.

## Data

We mainly use the shared task dataset, but we also utilize the following dataset and vocabularies:

- To build ontologies for ResCNN, we use
  - [CTD Database](https://ctdbase.org)
    - Disease vocabulaury (MEDIC)
    - Chemical vocabulary (MeSH)
- To extend the original training dataset with external dataset related with Disease and Chemical entities, we add following datasets for our training set.
  - NCBI Disease dataset
  - BC5CDR dataset

<!-- ## Model

We viewed the NER task as a token classification problem, where the goal is to label each token as either the beginning of (B-), part of (I-), or outside of an entity (O). Our best model is a pretrained PubMedBERT model finetuned using the entire training set. The predictions from this model are further refined using a small set of rules (see post-processing section below). -->
