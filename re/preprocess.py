import os
import argparse
import json
import pandas as pd
import spacy
import scispacy
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import KFold

def process_data(tokenizer, nlp, text_data, entity_data, relation_data = None, relation_type_column = 'type'):
  processed_data = []

  total_multientities = 0
  total_multientity_relations = 0
  total_actual_relations = 0
  same_types = 0
  same_novelties = 0
  
  for index, row in text_data.iterrows():
      training_sample = {'doc_key' : row['abstract_id'],
                        'sentence_texts' : [],
                        'sentence_spans' : []
                         }

      doc = nlp(row['text'])

      doc_entities = entity_data[entity_data['abstract_id'] == row['abstract_id']]

      if relation_data is not None:
          doc_relations = relation_data[relation_data['abstract_id'] == row['abstract_id']]

      # multi-entities are mentions with multiple entity ids assigned to them (entity ids separated by comma)
      # e.g. 1812,1813 is a multi-entity | we will treat each multi-entity as a separate entity
      # for each multi-entity, get set of all related entity ids (denoted as co-related entities)
      # e.g. if 1812 is related to 1814 and 1813 is related to 1815, then 1812,1813 has co-related entities {1814,1815}
      entity_start_indices = []
      multi_entities = {}
      for index, e in doc_entities.iterrows(): 
        entity_ids = e['entity_ids'].split(',')
        if len(entity_ids) > 1 and e['entity_ids'] not in multi_entities and relation_data is not None:
          co_related_entities = None
          entity_ids_set = set(entity_ids)
          for entity_id in entity_ids:
            if co_related_entities is None:
              co_related_entities = set(doc_relations[doc_relations['entity_1_id'] == entity_id]['entity_2_id'].unique()).union(
                          set(doc_relations[doc_relations['entity_2_id'] == entity_id]['entity_1_id'].unique())) - set(entity_ids)
            else:
              co_related_entities = co_related_entities.intersection(set(doc_relations[doc_relations['entity_1_id'] == entity_id]['entity_2_id'].unique()).union(
                          set(doc_relations[doc_relations['entity_2_id'] == entity_id]['entity_1_id'].unique())) - set(entity_ids))
          if len(co_related_entities) > 0:
            multi_entities[e['entity_ids']] = co_related_entities
        entity_start_indices.append((e['offset_start'], e['offset_finish'], e['type'], e['entity_ids'], e['mention']))

      relations_to_remove = set()
      relations_to_not_remove = set() #trumps relations to remove
      multientity_relations = []

      total_multientities += len(multi_entities)
      total_multientity_relations += sum([len(v) for v in multi_entities.values()])
      total_actual_relations += sum([len(k.split(',')) * len(v) for k, v in multi_entities.items()])
      
      # for each multi-entity/co-related entity pair, decide whether a particular relation comes from this multi-entity or from an individual entity mention (see cases below)
      for multientity, co_related_entities in multi_entities.items():
        same_type = True
        same_novelty = True
        for co_related_entity in co_related_entities:
          novelties = set()
          types = set()
          for entity_id in multientity.split(','):
            novelty = set(doc_relations[((doc_relations['entity_1_id'] == entity_id) & (doc_relations['entity_2_id'] == co_related_entity) |
                                      (doc_relations['entity_2_id'] == entity_id) & (doc_relations['entity_1_id'] == co_related_entity))]['novel'].unique())
            
            rel_types = set(doc_relations[((doc_relations['entity_1_id'] == entity_id) & (doc_relations['entity_2_id'] == co_related_entity) |
                                      (doc_relations['entity_2_id'] == entity_id) & (doc_relations['entity_1_id'] == co_related_entity))]['type'].unique())
            
            novelties = novelties.union(novelty)
            types = types.union(rel_types)

            relations_to_remove.add(tuple(sorted([entity_id, co_related_entity])))

          # case 1: all individual entities in the multientity have same type/novelty relations
          # e.g. 1812 and 1813 both have Association/Novel relations to 1814
          # assume that the relation comes from this mention, add multi-entity and --
          # remove every single entity pair relation (remove 1812-1814 and 1813-1814 relations)
          if len(types) == 1 and len(novelties) == 1:
            multientity_relations.append({'id' : 1,
                                          'abstract_id' : row['abstract_id'],
                                          'type' : list(types)[0],
                                          'entity_1_id' : multientity,
                                          'entity_2_id' : co_related_entity,
                                          'novel' : list(novelties)[0]})
          else:
            # case 2: individual entities in the multientity have different type/novelty relations
            # e.g. 1812 has Association/Novel relation to 1814; 1812 has PosCorrelation/No relation to 1814
            # usually in this case, one of the individual entities is not mentioned by itself in the text 
            # add multi-entity relation with a default type Association and novelty No
            # do not remove the individual entity pair relations (e.g. 1812-1814 and 1813-1814 relations)
            if len(types) > 1 and len(novelties) > 1:
              multientity_relations.append({'id' : 1,
                                          'abstract_id' : row['abstract_id'],
                                          'type' : 'Association',
                                          'entity_1_id' : multientity,
                                          'entity_2_id' : co_related_entity,
                                          'novel' : 'No'})

              for entity_id in multientity.split(','):
                if len(doc_entities[doc_entities['entity_ids'] == entity_id]) != 0: # there is an indv entity mention
                  relations_to_not_remove.add(tuple(sorted([entity_id, co_related_entity])))
              same_type = False
              same_novelty = False
              
            # case 3: individual entities in the multientity have different type relations
            # e.g. 1812 has Association relation to 1814; 1812 has PosCorrelation relation to 1814
            # add multi-entity relation with a default type Association, use the same Novelty
            # do not remove the individual entity pair relations (e.g. 1812-1814 and 1813-1814 relations)
            elif len(types) > 1:
              indv_mentions = 0
              for entity_id in multientity.split(','):
                if len(doc_entities[doc_entities['entity_ids'] == entity_id]) > 0: # there is an indv entity mention
                  relations_to_not_remove.add(tuple(sorted([entity_id, co_related_entity])))
                  indv_mentions += 1

              if indv_mentions < len(multientity.split(',')):
                multientity_relations.append({'id' : 1,
                                            'abstract_id' : row['abstract_id'],
                                            'type' : 'Association',
                                            'entity_1_id' : multientity,
                                            'entity_2_id' : co_related_entity,
                                            'novel' : list(novelties)[0]})
              same_type = False
            
            # case 4: individual entities in the multientity have different type relations
            # e.g. 1812 has No novelty relation to 1814; 1812 has Novel novelty relation to 1814
            # add multi-entity relation with a default novelty No, use the same relation type
            # do not remove the individual entity pair relations (e.g. 1812-1814 and 1813-1814 relations)
            elif len(novelties) > 1:
              indv_mentions = 0
              for entity_id in multientity.split(','):
                if len(doc_entities[doc_entities['entity_ids'] == entity_id]) > 0: # there is an indv entity mention
                  relations_to_not_remove.add(tuple(sorted([entity_id, co_related_entity])))
                  indv_mentions += 1

              if indv_mentions < len(multientity.split(',')):
                multientity_relations.append({'id' : 1,
                                          'abstract_id' : row['abstract_id'],
                                          'type' : list(types)[0],
                                          'entity_1_id' : multientity,
                                          'entity_2_id' : co_related_entity,
                                          'novel' : 'No'})
              same_novelty = False
        
        if same_type:
          same_types += 1
        if same_novelty:
          same_novelties += 1

      # based on the cases above, remove individual relations and add the multi-entity relations
      if relation_data is not None:
        for (entity_1_id, entity_2_id) in relations_to_remove - relations_to_not_remove:
          doc_relations.drop(doc_relations[((doc_relations['entity_1_id'] == entity_1_id) & (doc_relations['entity_2_id'] == entity_2_id)) |
                                          ((doc_relations['entity_1_id'] == entity_2_id) & (doc_relations['entity_2_id'] == entity_1_id))].index, inplace = True)

        doc_relations = pd.concat([doc_relations, pd.DataFrame(multientity_relations)])

      tokens = []
      sentence_token_indices = []
      entities = []
      ordered_entity_ids = []
      relations = []
      
      char_index = 1       
      sent_index = 0
      
      # assign sentence boundaries (in case the best model requires sentence information)
      for sent in doc.sents:
        tokenized = tokenizer(sent.text)
        token_ids = tokenized['input_ids'][1:-1] # remove [CLS] and [SEP]

        sentence_token_indices += token_ids
        tokens += tokenizer.convert_ids_to_tokens(token_ids)
        training_sample['sentence_spans'].append((char_index, char_index  + len(token_ids)))
        training_sample['sentence_texts'].append(sent.text)

        i = 0
        while i < len(token_ids):
          char_span = tokenized.token_to_chars(i + 1) # +1 because of [CLS]
          found_indices = []
          for index, (start, end, ent_type, entity_id, mention) in enumerate(entity_start_indices):
            if start in range(doc[sent.start].idx + char_span.start, doc[sent.start].idx + char_span.end):
              ent_start_index = char_index + i

              # get end index
              j = i
              while j < len(token_ids):
                if doc[sent.start].idx + tokenized.token_to_chars(j + 1).end >= end:
                  ent_end_index = char_index + j
                  break
                j += 1

              found_indices.append(index)

              if entity_id not in ordered_entity_ids:
                ordered_entity_ids.append(entity_id)
                entities.append({
                    'entity_id' : entity_id,
                    'mention_spans' : [],
                    'entity_type' : ent_type,
                    'sentence_mentions' : [sent_index],
                })
              ent_index = ordered_entity_ids.index(entity_id)
              entities[ent_index]['mention_spans'].append((ent_start_index, ent_end_index))
              if sent_index not in entities[ent_index]['sentence_mentions']:
                entities[ent_index]['sentence_mentions'].append(sent_index)

          for index in sorted(found_indices, reverse = True):
            del entity_start_indices[index]

          i += 1
        char_index += i
        sent_index += 1

      tokens = ['[CLS]'] + tokens + ['[SEP]']
      sentence_token_indices = [ tokenizer.convert_tokens_to_ids('[CLS]')] + sentence_token_indices + [ tokenizer.convert_tokens_to_ids('[SEP]')]

      if relation_data is not None:
        for index, r in doc_relations.iterrows():
          
          if r['entity_1_id'] in ordered_entity_ids and r['entity_2_id'] in ordered_entity_ids:
            relation_type = r[relation_type_column]

            rel_tuple = (ordered_entity_ids.index(r['entity_1_id']), 
                          ordered_entity_ids.index(r['entity_2_id']),
                          relation_type)
            relations.append(rel_tuple)

      training_sample['tokens'] = tokens
      training_sample['token_indices'] = sentence_token_indices
      training_sample['ner'] = entities
      training_sample['relations'] = relations
      
      multientities = {}
      for j, entity in enumerate(training_sample['ner']):
        if ',' in entity['entity_id']:
          multientities[entity['entity_id']] = j
      
      # multientity-individual entity relations are not allowed as negative examples
      # if indv entity is part of the multientity set
      # e.g. 1812 to 1812,1813 relation is not allowed even as a negative example
      no_relations = []
      for j, entity in enumerate(training_sample['ner']):
        for multientity, mid in multientities.items():
          if j == mid: continue
          if len(set(entity['entity_id'].split(',')).intersection(set(multientity.split(',')))) > 0:
            no_relations.append(tuple(sorted([mid, j])))
      training_sample['no_relations'] = no_relations

      processed_data.append(training_sample)

  if relation_data is not None:
    print(f'Total multi-entities: {total_multientities}')
    print(f'Total multi-entity relations : {total_multientity_relations}')
    print(f'Total actual relations : {total_actual_relations}')
    print(f'Same types: {same_types} ({same_types/total_multientities})')
    print(f'Same novelties: {same_novelties} ({same_novelties/total_multientities})')

  return processed_data
  
def create_folds(data, output_dir, folds = 5):
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  np_data = np.array(data)

  kf = KFold(n_splits = folds, random_state = 0, shuffle = True)

  j = 0
  for train_index, val_index in kf.split(data):
    train, val = np_data[train_index], np_data[val_index]

    save_json_data(train, os.path.join(output_dir, f'train_{j}.json'))
    save_json_data(val, os.path.join(output_dir, f'val_{j}.json'))

    j += 1 
  
def save_json_data(data, filename):
   with open(filename, 'w') as o:
      for d in data:
        json.dump(d, o)
        o.write('\n')
  
def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--tokenizer', type=str, help='Pretrained tokenizer name or path')
    parser.add_argument('--text_file', type=str, help='Path to tab-delimited file with input text data')
    parser.add_argument('--entity_file', type=str, help='Path to tab-delimited file with entity data')
    parser.add_argument('--relation_file', type=str, default=None, help='Path to tab-delimited file with relation data')
    parser.add_argument('--output_file', type=str, help='Path to output JSON file')
    parser.add_argument('--num_folds', type=int, default=None, help='If provided, split the data into given folds')

    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    nlp = spacy.load('en_core_sci_scibert', exclude = ['transformer', 'tagger', 'attribute_ruler', 'lemmatizer', 'parser', 'ner'])
    nlp.add_pipe('sentencizer')
    
    text_data = pd.read_csv(args.text_file, sep = '\t')
    text_data['text'] = text_data['title'] + ' ' + text_data['abstract']

    entity_data = pd.read_csv(args.entity_file, sep = '\t')
    
    if args.relation_file is not None:
        relation_data = pd.read_csv(args.relation_file, sep = '\t')
        
        for label in ['type', 'novel']:
            processed_data = process_data(tokenizer, nlp, text_data, entity_data, relation_data, label)
            save_json_data(processed_data, f"{args.output_file.split('.')[0]}_{label}.json")
            
            if args.num_folds is not None:
                create_folds(processed_data, f"{args.output_file.split('.')[0]}_{label}", args.num_folds)
    else: 
        processed_data = process_data(tokenizer, nlp, text_data, entity_data, None, None)
        save_json_data(processed_data, args.output_file)
    
if __name__ == "__main__":
    main()