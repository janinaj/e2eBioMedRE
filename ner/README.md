# Named Entity Recognition

## Reproducing results:

To run the entire pipeline, from preprocessing to model training to post-processing, run the following command:
```
./reproduce_results.sh
```
This command includes installation of the required Python packages. Predictions are saved in *models/final_model/postprocessed_predictions.csv*.

We also included a copy of the trained NER model (*models/pretrained_model/*). To regenerate the test data results using this model, run:

```
./regenerate_results_from_trained_model.sh
```

Predictions are saved in *models/pretrained_model/postprocessed_predictions.csv*.

Approximate running time (after all required packages have been installed):
	- 2.5 minutes on an A100 GPU
	- 3.5 minutes on a V100 GPU
	- 4.5 minutes on a P100 GPU

**Note that you may need to change file permissions to run the .sh and .py files (e.g. chmod 755).**

## Environment details:
- Python version: 3.7
- GPU: Tesla P100
- Note: in order to to reproduce the exact results, you also need to use the same GPU. We have verified that the model will be different if a different GPU is used (even if the random state and hyperparameters are kept constant).

## Installation:
```
    pip install -r requirements.txt
```

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

Title and abstract are combined into a single text field. We did not perform any sentence splitting. Each abstract is a single document/sample. We did not use entity ids (i.e. our model does not recognize if multiple mentions are for the same real-world entity. 

## Model

We viewed the NER task as a token classification problem, where the goal is to label each token as either the beginning of (B-), part of (I-), or outside of an entity (O). Our best model is a pretrained PubMedBERT model finetuned using the entire training set. The predictions from this model are further refined using a small set of rules (see post-processing section below).

### Model Hyperparameters
Only two hyperparameters need to be specifically set:
- Maximum sequence length: 512
- Training epochs: 10

## Pipeline
###  Preprocessing

Two scripts perform the following preprocessing steps:
- *combine_title_and_abstract.py*:  takes in a tab-delimited abstracts file (see abstracts_train.csv for format), combines title and abstract (space is added in between) into a single field named *text*, and generates a new tab-delimited file with the added field. Two input files are required: the abstracts file and the entities file. It is assumed that the format follows the format of *abstracts_train.csv* and *entities_train.csv*.
- *tokenize_and_align_labels.py*: tokenizes text using the pretained BERT tokenizer (needs to be available in the Huggingface platform), and generates labels for each token. We follow the BIO labeling convention (B-beginning of token entity, I-part of entity, O-not part of entity). Entity types are also added as part of the label types (e.g., B-ChemicalEntity, I-CellLine). This script generates a line-by-line JSON-formatted file. Two input files are required: the abstracts file and the entities file. It is assumed that the *text* field has been  added to the abstracts file.

Notes
- 17 out of 400 training documents have over 512 tokens (extra tokens are truncated)
-  3 out of 100 test documents have over 512 tokens
- 24 training entities were misaligned. The token boundaries do not match with these tokens. We ignored these mis-aligned entities (i.e. their labels are O).

Sample commands for preprocessing the training and test data:
```
python combine_title_and_abstract.py \
	--input_file data/abstracts_train.csv \
	--output_file data/abstracts_train_text.csv
python tokenize_and_align_labels.py \
	--tokenizer microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
	--max_num_tokens 512 \
	--text_file data/abstracts_train_text.csv \
	--entities_file data/entities_train.csv \
	--output_file data/processed_train_data.json

python combine_title_and_abstract.py \
	--input_file data/abstracts_test.csv \
	--output_file data/abstracts_test_text.csv
python tokenize_text.py --tokenizer microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
	--max_num_tokens 512 \
	--text_file data/abstracts_test_text.csv \
	--output_file data/processed_test_data.json
```

Additional note: One of the major challenges is to find a tokenizer than aligns with the entity boundaries. Initially, we experimented with using spacy and scispacy for tokenization, but many entities were misaligned, which produced less accurate results. Using the pretrained PubMedBERT model for tokenization produced the best results.

### Model Training, Evaluation, and Prediction
Our model training and prediction script is a modified version of the [run_ner.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py) script from the Huggingface Python transformers library.

*run_ner.py*: performs model training, evaluation, and prediction

Sample script for model training, evaluation, and prediction:
```
python run_ner.py 
	--model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \ 
	--max_seq_length 512 \
	--num_train_epochs 10 \
    --do_train --do_eval --do_predict \
    --text_column_name text \
    --label_column_name labels \
    --train_file data/processed_train_data.json \
    --eval_file data/processed_eval_data.json \
    --test_file data/processed_test_data.json  \
    --output_dir model
```

If the ``--do_predict`` option is specified, the test data predictions are stored in the model output folder as a CSV file named *predictions.csv*.

### Post-processing

Two scripts perform the following post-processing steps:
- *convert_output_to_csv_format.py*: The model script outputs predictions in a JSON-formatted file. This script converts it to the final format (i.e. the submission format).
- *postprocess .py*: This script combines two consecutive entities IF:
	- there is no space in between them and they have the same entity type
	- there is a space in between them and their entity types are both DiseaseOrPhenotypicFeature
The output file of *postprocess .py* contains the final set of predictions.

Sample script for preprocessing:
```
python convert_output_to_csv_format.py \
	--tokenized_input_file data/processed_test_data.json \
	--predictions_file models/final_model/predictions.txt \
	--output_file models/final_model/predictions.csv
	
python postprocess.py \
	--predictions_file models/final_model/predictions.csv \
	--output_file models/final_model/postprocessed_predictions.csv
```

## Training Results
Since training time is fast, we ran 10-fold cross validation by training on 90%. The results range from 0.87-0.90 F1-score. Adding the post-processing step boosts these results by 0.01-0.03. For more details about model training results, please see *training_results.txt*.