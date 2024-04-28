# This script converts the JSON-formatted prediction output to the final CSV format.

import argparse
import json
import pandas as pd

def generate_test_results(tokenized_input_file, prediction_file, output_file):
    tokens_and_indices = []
    with open(tokenized_input_file, 'r') as f:
        for line in f:
            tokens_and_indices.append(json.loads(line))

    predictions = []
    with open(prediction_file, 'r') as f:
        for line in f:
            predictions.append(line.split())

    final_output = []
    for ti, p in zip(tokens_and_indices, predictions):
        i = 0
        id_ = 0
        while i < len(p):
            bio_label = p[i]
            if bio_label.startswith('B'):
                offset_start = ti['start_indices'][i]
                offset_end = ti['end_indices'][i]
                entity_type = bio_label[2:]
    
                while i + 1 < len(p) and p[i + 1].startswith('I'):
                    i += 1
                    offset_end = ti['end_indices'][i]
        
                final_output.append({'id' : id_,
                                      'abstract_id' : ti['abstract_id'],
                                      'offset_start' : offset_start,
                                      'offset_finish' : offset_end,
                                      'type' : entity_type})
                id_ += 1
            i += 1
      
    pd.DataFrame(final_output).to_csv(output_file, sep = '\t', index = False)
  
def main():
    parser = argparse.ArgumentParser(description='Combine title and abstract into a single text field')
    parser.add_argument('--tokenized_input_file', type=str, help='Path to file with input text data')
    parser.add_argument('--predictions_file', type=str, help='Path to file with entity predictions')
    parser.add_argument('--output_file', type=str, help='Path to output file')

    args = parser.parse_args()
    
    generate_test_results(args.tokenized_input_file, args.predictions_file, args.output_file)
    
if __name__ == "__main__":
    main()
  
