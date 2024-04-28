import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Process relation predictions for novelty prediction')
    parser.add_argument('--predictions_file', type=str, help='Path to file with relation predictions')
    parser.add_argument('--output_file', type=str, help='Path to output file')
    
    args = parser.parse_args()

    with open(args.output_file, 'w') as o:
      with open(args.predictions_file, 'r') as f:
        for line in f:
          pred = json.loads(line)
          ent_ids = {}
          for i, ner in enumerate(pred['ner']):
            ent_ids[ner['entity_id']] = i
          rels = []
          for rel in pred['relations']:
            rels.append([ent_ids[rel[0]], ent_ids[rel[1]], 'Novel'])
          pred['relations'] = rels
    
          json.dump(pred, o)
          o.write('\n')
    
if __name__ == "__main__":
    main()