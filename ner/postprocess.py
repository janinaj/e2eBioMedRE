# This script takes in a tab-delimited file containing predictions as input
# and performs the following postprocessing rules:
# combine 2 consecutive entities w/ no space in between IF they have the same type
# combine 2 consecutive entities w/ spaces in between IF their types are both DiseaseOrPhenotypicFeature

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Combine title and abstract into a single text field')
    parser.add_argument('--predictions_file', type=str, help='Path to tab-delimited predictions file')
    parser.add_argument('--output_file', type=str, help='Path to output file')

    args = parser.parse_args()
    
    predictions = pd.read_csv(args.predictions_file, sep = '\t')

    prev_row = None
    row_combined = False
    total_num_combined = 0
    postprocessed_output = []
    for index, row in predictions.iterrows():
        if prev_row is not None:
            if row['type'] == prev_row['type'] and (row['offset_start'] - prev_row['offset_finish'] == 0 or \
                (row['offset_start'] - prev_row['offset_finish'] == 1 and row['type'] == 'DiseaseOrPhenotypicFeature')):
                total_num_combined += 1
    
                row['offset_start'] = prev_row['offset_start']
                
                postprocessed_output.append(row.to_dict())
                row_combined = True
            else:
                if not row_combined:
                    postprocessed_output.append(prev_row.to_dict()) 
                row_combined = False
        prev_row = row
    if not row_combined:
        postprocessed_output.append(prev_row.to_dict()) 
        
    pd.DataFrame(postprocessed_output).to_csv(args.output_file, sep = '\t', index = False)
    
    print(f'{total_num_combined} total entity combinations of consecutive entities')
    print(f'Final output saved to {args.output_file}')
    
if __name__ == "__main__":
    main()