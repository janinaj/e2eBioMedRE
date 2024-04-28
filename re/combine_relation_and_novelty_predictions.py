import argparse
import json
import pandas as pd
from collections import defaultdict

# this is the part where we separate multientities into individual entities
# add all multientity relations if relation already exists w/ one of the multientities, only skip if not same relation
# if there are multiple predictions with diff types, get majority rule
def get_all_single_entity_predictions(predictions):
  new_pairs = dict()
  for i, prediction in predictions[predictions['entity_1_id'].str.contains(',')].iterrows():
    other_pred_exists = False

    # find existing predicted relation
    for entity_1_id in prediction['entity_1_id'].split(','):
      for entity_2_id in prediction['entity_2_id'].split(','):
        other_predictions = predictions[((predictions['entity_1_id'] == entity_1_id) & (predictions['entity_2_id'] == entity_2_id)) |
                                ((predictions['entity_2_id'] == entity_1_id) & (predictions['entity_1_id'] == entity_2_id))]
        if len(other_predictions) > 0 and (len(other_predictions['type'].unique()) > 1 or list(other_predictions['type'].unique())[0] != prediction['type']):
          other_pred_exists = True
          break
      if other_pred_exists: break
    
    if not other_pred_exists:
      for entity_1_id in prediction['entity_1_id'].split(','):
        for entity_2_id in prediction['entity_2_id'].split(','):
          rel = tuple([prediction['abstract_id']] + sorted([entity_1_id, entity_2_id]))
          if rel not in new_pairs:
            new_pairs[rel] = defaultdict(int)
          new_pairs[rel][prediction['type']] += 1

  # do the same thing for the other entity column
  for i, prediction in predictions[predictions['entity_2_id'].str.contains(',')].iterrows():
    other_pred_exists = False

    # find existing predicted relation
    for entity_1_id in prediction['entity_1_id'].split(','):
      for entity_2_id in prediction['entity_2_id'].split(','):
        other_predictions = predictions[((predictions['entity_1_id'] == entity_1_id) & (predictions['entity_2_id'] == entity_2_id)) |
                                ((predictions['entity_2_id'] == entity_1_id) & (predictions['entity_1_id'] == entity_2_id))]
        if len(other_predictions) > 0 and (len(other_predictions['type'].unique()) > 1 or list(other_predictions['type'].unique())[0] != prediction['type']):
          other_pred_exists = True
          break
      if other_pred_exists: break

    if not other_pred_exists:
      for entity_1_id in prediction['entity_1_id'].split(','):
        for entity_2_id in prediction['entity_2_id'].split(','):
          rel = tuple([prediction['abstract_id']] + sorted([entity_1_id, entity_2_id]))
          if rel not in new_pairs:
            new_pairs[rel] = defaultdict(int)
          new_pairs[rel][prediction['type']] += 1

  predictions_to_add = []
  for (abstract_id, entity_1_id, entity_2_id), types in new_pairs.items():
    rel_type = max(types, key=types.get)
    predictions_to_add.append({'id' : 1,
                                      'abstract_id' : abstract_id,
                                      'type' : rel_type,
                                      'entity_1_id' : entity_1_id,
                                      'entity_2_id' : entity_2_id})

  final_predictions = pd.concat([predictions, pd.DataFrame(predictions_to_add)])
  final_predictions = final_predictions[~(final_predictions['entity_1_id'].str.contains(',')) & ~(final_predictions['entity_2_id'].str.contains(','))]
  final_predictions = final_predictions[final_predictions['entity_1_id'] != final_predictions['entity_2_id']]

  return final_predictions

def main():
    parser = argparse.ArgumentParser(description='Process relation predictions for novelty prediction')
    parser.add_argument('--relation_file', type=str, help='Path to file with relation predictions')
    parser.add_argument('--novelty_file', type=str, help='Path to file with novelty predictions')
    parser.add_argument('--output_file', type=str, help='Path to output file')
    
    args = parser.parse_args()
    
    rel_predictions = pd.read_csv(args.relation_file, sep = '\t')
    nov_predictions = pd.read_csv(args.novelty_file, sep = '\t')
    
    rel_predictions = rel_predictions[(rel_predictions['entity_1_id'].notnull()) & (rel_predictions['entity_2_id'].notnull())]
    nov_predictions = nov_predictions[(nov_predictions['entity_1_id'].notnull()) & (nov_predictions['entity_2_id'].notnull())]
    
    rel = get_all_single_entity_predictions(rel_predictions)
    nov = get_all_single_entity_predictions(nov_predictions)

    for i, r in rel.iterrows():
      if len(nov[(nov['entity_1_id'] == r['entity_1_id']) & (nov['entity_2_id'] == r['entity_2_id']) | \
            (nov['entity_2_id'] == r['entity_1_id']) & (nov['entity_1_id'] == r['entity_2_id'])]) > 0:
        rel.loc[i, 'novel'] = 'Novel'
      else:
        rel.loc[i, 'novel'] = 'No'

    rel.to_csv(args.output_file, index = False, sep = '\t')
    
if __name__ == "__main__":
    main()