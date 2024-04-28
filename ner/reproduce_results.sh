model=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract

pip install -r requirements.txt

echo "----------Preprocessing training data----------"
python combine_title_and_abstract.py --input_file data/abstracts_train.csv --output_file data/abstracts_train_text.csv
python tokenize_and_align_labels.py --tokenizer $model --max_num_tokens 512 --text_file data/abstracts_train_text.csv --entities_file data/entities_train.csv --output_file data/processed_train_data.json

echo "----------Preprocessing test data----------"
python combine_title_and_abstract.py --input_file data/abstracts_test.csv --output_file data/abstracts_test_text.csv
python tokenize_text.py --tokenizer $model --max_num_tokens 512 --text_file data/abstracts_test_text.csv --output_file data/processed_test_data.json

echo "----------Training NER model----------"
python run_ner.py --model_name_or_path $model --max_seq_length 512 --num_train_epochs 10 \
    --do_train --do_predict \
    --text_column_name text --label_column_name labels \
    --train_file data/processed_train_data.json \
    --test_file data/processed_test_data.json  \
    --output_dir models/final_model --overwrite_output_dir
    
echo "----------Postprocessing final output----------"
python convert_output_to_csv_format.py --tokenized_input_file data/processed_test_data.json --predictions_file models/final_model/predictions.txt --output_file models/final_model/predictions.csv
python postprocess.py --predictions_file models/final_model/predictions.csv --output_file models/final_model/postprocessed_predictions.csv
