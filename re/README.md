# Relation Extraction and Novelty Prediction

## Reproducing results:

To run the entire pipeline, from preprocessing to model training to post-processing, run the following command:
```
./reproduce_results.sh
```
This command includes installation of the required Python packages. Predictions are saved in *models/novelty_model/test_predictions.csv*.

Note: Results might not be entirely the same as the submitted predictions. This is due to random state issues during training (it could be this [issue](https://discuss.huggingface.co/t/fixing-the-random-seed-in-the-trainer-does-not-produce-the-same-results-across-runs/3442) since we are using an older version of trans. Our experiments do show a <1% F1-score difference between multiple identical runs (i.e. same exact parameters).

We also included a copy of the trained models. To regenerate the test data results using this model, run:

```
./regenerate_results_from_trained_models.sh
```

- Note: in order to to reproduce the exact results, you also need to use the same GPU. We have verified that the results will be different if a different GPU is used (even if the random state and hyperparameters are kept constant). We trained the model and generated predictions using a V100 GPU. The differences are small, but we wanted to note that there is a difference. For instance, there is only 1 prediction that is different when using an A100 GPU.

Approximate running time:
	- 530 minutes on a V100 GPU
	- 200 minutes on an A100 GPU

### Notes
- You may need to change file permissions to run the .sh and .py files (e.g. chmod 755).
- Our relation extraction model was only trained on 80% of the training set (randomly-selected). We hypothesize that training the relation extraction model on 100% of the training set will produce better relation predictions.

## Environment details:
- Python version: 3.7
- GPU: Tesla V100

## Data
We used the following files and datafields to train the model:

- abstracts_train.csv
	- abstract_id
	- title
	- abstract
- entities_train.csv
	- abstract_id
	- offset_start
	- offset_finish
	- type
	- mention
	- entity_ids
	
- relations_train.csv
	- abstract_id
	- entity_1_id
	- entity_2_id
	- type
	- novel

Title and abstract are combined into a single text field. Each abstract is a single document/sample. We used the entity ids to denote the same real-world entities. We performed sentence splitting using scispacy, primarily for truncating text. We removed sentences in order to fit the maximum token length.

## Model

Our model is inspired by a 2021 NAACL paper (A Frustratingly Easy Approach for Entity and Relation Extraction) [1], with several key changes. The main approach of this paper is to generate entity span representations, and use these representations for classification (i.e. relation type or no relation). The entity span representations are extracted from the output of a standard BERT model, which is also finetuned during model training. Marker tokens are added at the beginning and end of each entity, and the beginning marker tokens are taken as the span representation. In the following example,

 *We report on a new allele at the [SUBJ_START_GENE] arylsulfatase A [SUBJ_END_GENE] (ARSA) locus causing late-onset [SUBJ_START_DISEASE] metachromatic leukodystrophy [SUBJ_END_DISEASE] (MLD).*

The entity span representation of *arylsulfatase A* is the embedding of token *[SUBJ_START_GENE]*, while the entity span representation of *metachromatic leukodystrophy* is the token embedding of *[SUBJ_START_DISEASE]*. These span representations are then concatenated to produce a relation representation, which is used to classify the presence/absence of a relation and the relation type. Each entity pair is processed as a separate training example.

We made several key changes to this model:
- The concatenation of span representations *[SUBJ, OBJ]* implies directionality. The relations in the LitCoin dataset, have no inherent directionality. We modified the span representations to generate two representations *[SUBJ, OBJ, SUBJ*OBJ]* and  *[OBJ, SUBJ, SUBJ*OBJ]* (the product, SUBJ*OBJ, is to avoid collinearity in the model). Our goal is to classify both of these cases correctly.
- We used a single entity marker for both entities as well as the start and end of the spans. In our experiments, this setting produced the best results over distinguishing start/end tokens as well as by entity types.

 *We report on a new allele at the [ENTITY] arylsulfatase A [ENTITY] (ARSA) locus causing late-onset [ENTITY] metachromatic leukodystrophy [ENTITY] (MLD).*

- We tagged all mentions of the pair of entities in the sample. In the original code, it is assumed that there is single mention for each entity. This is not the case with the Litcoin dataset (or most real-world examples).

*We report on a new allele at the [ENTITY] arylsulfatase A [ENTITY] ([ENTITY] [ARSA] [ENTITY]) locus causing late-onset [ENTITY] metachromatic leukodystrophy [ENTITY] ([ENTITY] [MLD] [ENTITY]).*

- The entire abstract is used for each training sample. In the original code, only the current sentence and/or nearby sentences are used.

We finetuned the PubMedBERT model for this task.

### Relation Representation for the Relation Extraction Task

Because we considered multiple mentions of the same entity, selecting/weighting the mentions becomes a challenging task. Intuitively, not all mentions are important for classifying the relation (e.g. there is likely a single sentence or single pair of mentions that denote the relation). We used two different solutions for the relation extraction task based on training results. For the relation extraction task, we perform selection, i.e., we select the pair of mentions (entity 1, entity 2) that have the best "compatibility". This is akin to calculating the attention between each mention in entity 1 and entity 2. We take the dot product of each mention pair (mathematically, this is performed as a matrix product). This dot product represents the importance of each mention pair to classifying the relation. We take the mention pair with the highest dot product and use it as the relation representation.

### Relation Representation for the Novelty Prediction Task

For the novelty prediction task, we perform weighting. Specifically, we compute the log sum of exponentials (logsumexp) for all mentions of entity 1 and entity 2 separately. The main idea is still the same: some mentions are more important than others, only in this case, we still consider all mentions as possibly contributing to the novelty prediction task.
  
### Adversarial Training
After passing the input through the main model, we introduce noise to the token embeddings of the input and pass it again to the model. We perform this step for a maximum of 3 times for each training batch. This step adds a significant amount of training time. However, we get a more robust model and better results.

### Negative Examples
One of the goals of the relation extraction task is to identify is two entities are related or not. Hence, we added negative examples (i.e. pairs of entities in an abstract that have no relation). In the novelty prediction task, we assume that the training data consists of all positive examples.

### Model Hyperparameters
We specified the following hyperparameters for relation extraction training:
-   8 epochs
-   512 maximum token length
-   2e-5 learning rate
-   8 batch size
   
We specified the following hyperparameters for novelty prediction training:
-   15 epochs
-   512 maximum token length
-   1e-5 learning rate
-   8 batch size for both training and evaluation

## Pipeline
###  Preprocessing

Two scripts perform the following preprocessing steps:
- *preprocess.py*:  This file performs sentence splitting and tokenization. It combines abstract (text), entity, and relation (if present) information into a single JSON-formatted file for input into the model.
- *run_relation.py*: This is the model training file. However, we also perform more preprocessing steps to convert the data into the final input for the model.
	- - Entity markers (i.e. [ENTITY]) are added before and after each mention.
	- - If a document has more than 512 tokens (i.e. the maximum allowed), we removed sentences based on the following rules:
		-   Starting from the first non-title sentence, remove a sentence that does not contain any mentions of the two entities until the number of tokens is less than the maximum length. 
		-  If the token length is still greater than 512 after the above step, starting from the first non-title sentence, remove a sentence that only contains mentions of either entity 1 or entity 2 (i.e. only keep sentences that mention both entities).

Sample commands for preprocessing the training and test data:
```
python preprocess.py \
    --tokenizer microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --text_file data/abstracts_train.csv \
    --entity_file data/entities_train.csv \
    --relation_file data/relations_train.csv \
    --output_file data/processed_train_data.json \
    --num_folds 5
    
python preprocess.py \
    --tokenizer microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --text_file data/abstracts_test.csv \
    --entity_file data/entities_test.csv \
    --output_file data/processed_test_data.json
```

### Model Training, Evaluation, and Prediction

For relation extraction, the following parameter must be specified so the selection approach is used:

```{r}
  --with_mention_attention
```

For novelty prediction, the following parameter must be specified so training is performed only on actual relations:

```{r}
  --no_negative_label
```

A single script performs model training, evaluation, and prediction:
-*run_relation.py*: In our case, each abstract is a single sample.

Sample script for model training, evaluation, and prediction:
```
python run_relation.py \
  --task litcoin \
  --with_mention_attention \
  --do_train --train_file data/processed_train_data_type/train_0.json \
  --do_eval --eval_file data/processed_train_data_type/val_0.json \
  --do_predict --test_file data/processed_test_data.json \
  --model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --checkpoint 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 8 \
  --model_max_seq_length 512 \
  --marker_tokens [ENTITY] \
  --prediction_file test_predictions.json \
  --output_dir models/re_model
```

If the ``--do_predict`` option is specified, the test data predictions are stored in the model output folder as a JSON file named *predictions.json* (unless the ``--prediction_file`` parameter is specified).

### Post-processing

- *format_result_for_novelty_prediction.py*: This script converts the file into the input proper format for the novelty prediction task.
```
python format_result_for_novelty_prediction.py \
    --predictions_file models/re_model/test_predictions.json \
    --output_file models/re_model/test_predictions_formatted_for_novelty_prediction.json
```

- *combine_relation_and_novelty_predictions.py*: This script combines the relation extraction and novelty model predictions into a single file. The output file is also stored in the final submission format.

```
python combine_relation_and_novelty_predictions.py \
  --relation_file models/re_model/test_predictions.csv \
  --novelty_file models/novelty_model/test_predictions.csv \
  --output_file models/final_output.csv
```

[1] Zhong, Z., & Chen, D. (2020). A frustratingly easy approach for entity and relation extraction. arXiv preprint arXiv:2010.12812.