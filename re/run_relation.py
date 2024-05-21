"""
This code is based on the file in SpanBERT repo: https://github.com/facebookresearch/SpanBERT/blob/master/code/run_tacred.py
"""

import argparse
import logging
import os
import random
import time
import json
import sys
import csv

import numpy as np
import torch
# import matplotlib.pyplot as plt
# from math import floor

from torch.utils.data import DataLoader, TensorDataset
# from collections import Counter

from torch.nn import CrossEntropyLoss

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from models import BertForRelationMultiMention, BertForRelationMultiMentionAttention, PGD
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

CLS = "[CLS]"
SEP = "[SEP]"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

task_ner_labels = {
    'litcoin': ['DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct',
       'SequenceVariant', 'OrganismTaxon', 'ChemicalEntity', 'CellLine'],
    'novelty' : ['DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct',
       'SequenceVariant', 'OrganismTaxon', 'ChemicalEntity', 'CellLine'],
}

task_rel_labels = {
    'litcoin': ['Association', 'Positive_Correlation', 'Negative_Correlation',
       'Bind', 'Cotreatment', 'Comparison', 'Drug_Interaction',
       'Conversion'],
    'novelty' : ['No', 'Novel'],
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx

def add_marker_tokens(tokenizer, marker_tokens, ner_labels):
    new_tokens = []
    if marker_tokens == '[ENTITY]':
        new_tokens.append('MARKER_[ENTITY]')
    elif marker_tokens == 'ENT_TYPE':
        for label in ner_labels:
            new_tokens.append('MARKER_[%s]'%label)
    elif marker_tokens == 'STARTEND':
        new_tokens.append('MARKER_[START]')
        new_tokens.append('MARKER_[END]')
    elif marker_tokens == 'STARTEND_TYPE':
        for label in ner_labels:
            new_tokens.append('MARKER_[START_%s]'%label)
            new_tokens.append('MARKER_[END_%s]'%label)
    
    tokenizer.add_tokens(new_tokens)
    logger.info('# vocab after adding markers: %d'%len(tokenizer))
    
def decode_sample_id(sample_id):
    doc = sample_id.split('::')[0]
    pair = sample_id.split('::')[1][1:-1]
    sub = int(pair.split(',')[0])
    obj = int(pair.split(',')[1])

    return doc, sub, obj
    
# remove any sentence w/o entity mentions (until length fits)
# if num tokens still > max length, remove sentences w/ only a single entity mention (until length fits)
# if still doesn't fit, remove the sentences (until length fits)
# when removing, always start from first non-title sentence
def get_truncated_text(doc, entity_1, entity_2, model_max_seq_length):
    tokens = doc['tokens'][:] # make a copy so we don't modify original
    token_indices = doc['token_indices'][:]
    subj_mention_spans = entity_1['mention_spans'][:]
    obj_mention_spans = entity_2['mention_spans'][:]
    

    # remove any sentence w/o entity mentions
    new_sentence_spans = []
    new_entity_1_sent_mentions = entity_1['sentence_mentions'][:]
    new_entity_2_sent_mentions = entity_2['sentence_mentions'][:]
    cur_sent_end_index = 1
    for i, startend in enumerate(doc['sentence_spans']):
        start = startend[0]
        end = startend[1]
        offset = (end - start)
        if i != 0 and len(tokens) > model_max_seq_length and i not in entity_1['sentence_mentions'] and i not in entity_2['sentence_mentions']:
            tokens = tokens[:cur_sent_end_index] + tokens[cur_sent_end_index + offset:]
            token_indices = token_indices[:cur_sent_end_index] + token_indices[cur_sent_end_index + offset:]
            
            subj_mention_spans = [(s - offset, e - offset) if s > cur_sent_end_index else (s, e) for (s, e) in subj_mention_spans]
            obj_mention_spans = [(s - offset, e - offset) if s > cur_sent_end_index else (s, e) for (s, e) in obj_mention_spans]
            
            new_entity_1_sent_mentions = [new_entity_1_sent_mentions[idx] if entity_1['sentence_mentions'][idx]  <= i else new_entity_1_sent_mentions[idx] - 1 for idx in range(len(entity_1['sentence_mentions']))]
            new_entity_2_sent_mentions = [new_entity_2_sent_mentions[idx] if entity_2['sentence_mentions'][idx]  <= i else new_entity_2_sent_mentions[idx] - 1 for idx in range(len(entity_2['sentence_mentions']))]
            
        else:
            new_sentence_spans.append([cur_sent_end_index, cur_sent_end_index + offset])
            cur_sent_end_index += (offset)
    
    # if num tokens still > max length, remove sentences w/ only a single entity mention (until length fits)
    if len(tokens) > model_max_seq_length:
        
        cur_tokens = tokens[:] # make a copy so we don't modify original
        cur_token_indices = token_indices[:]
        
        cur_sentence_spans = new_sentence_spans[:]
        cur_entity_1_sent_mentions = new_entity_1_sent_mentions[:]
        cur_entity_2_sent_mentions = new_entity_2_sent_mentions[:]
        
        new_sentence_spans = []
        num_removed_sents = 0
        cur_sent_end_index = 1
        for i, startend in enumerate(cur_sentence_spans):
            start = startend[0]
            end = startend[1]
            offset = (end - start)
            if i != 0 and len(tokens) > model_max_seq_length and ((i not in cur_entity_1_sent_mentions and len(obj_mention_spans) > 1) \
                                                                or (i not in cur_entity_2_sent_mentions and len(subj_mention_spans) > 1)): # if only remaining entity mention, obv don't remove sentence

                
                tokens = tokens[:cur_sent_end_index] + tokens[cur_sent_end_index + offset:]
                token_indices = token_indices[:cur_sent_end_index] + token_indices[cur_sent_end_index + offset:]
                
                old_subj_mention_spans = subj_mention_spans[:]
                subj_mention_spans = []
                for (s, e) in old_subj_mention_spans:
                    # remove mention span if it is inside remove sentence
                    if s > cur_sent_end_index and s < cur_sent_end_index + offset :
                        pass
                    elif s > cur_sent_end_index:
                        subj_mention_spans.append((s - offset, e - offset))
                    else:
                        subj_mention_spans.append((s, e))
                    
                old_obj_mention_spans = obj_mention_spans[:]
                obj_mention_spans = []
                for (s, e) in old_obj_mention_spans:
                    # remove mention span if it is inside remove sentence
                    if s > cur_sent_end_index and s < cur_sent_end_index + offset :
                        pass
                    elif s > cur_sent_end_index:
                        obj_mention_spans.append((s - offset, e - offset))
                    else:
                        obj_mention_spans.append((s, e))

                
                num_removed_sents += 1
                new_entity_1_sent_mentions = [idx if idx  < num_removed_sents else idx - 1 for idx in new_entity_1_sent_mentions]
                new_entity_2_sent_mentions = [idx if idx  < num_removed_sents else idx - 1 for idx in new_entity_2_sent_mentions]
            else:
                new_sentence_spans.append([cur_sent_end_index, cur_sent_end_index + offset])
                cur_sent_end_index += (offset)

    return tokens, token_indices, subj_mention_spans, obj_mention_spans, new_sentence_spans
    
def generate_relation_data(entity_data, add_negative_label=True, marker_tokens='[UNK]', model_max_seq_length=128):
    """
    Prepare data for the relation model
    """
    logger.info('Generate relation data from %s'%(entity_data))
    
    data = []
    with open(entity_data, 'r') as f:
        for doc in f:
            data.append(json.loads(doc))

    nrel = 0
    max_sample = 0
    samples = []
    num_fit_examples = 0
    
    for doc in data:
        sent_samples = []

        sent_ner = doc['ner']
            
        for j in range(len(sent_ner)):
            sent_ner[j]['mention_spans'] = [tuple(ms) for ms in sent_ner[j]['mention_spans']]
            
        all_entity_pair_samples = set()
        for rel in doc['relations']:
            # if there is an overlap between entity 1 and entity 2 mentions, do not include in samples
            if len(set(sent_ner[rel[0]]['mention_spans']).intersection(set(sent_ner[rel[1]]['mention_spans']))) > 0:
                continue
            sample = {}	
            sample['docid'] = doc['doc_key']	
            sample['id'] = '%s::(%d,%d)'%(doc['doc_key'], rel[0], rel[1])	
            sample['relation'] = rel[2]
            
            if len(doc['tokens']) > model_max_seq_length:
                tokens, token_indices, subj_mention_spans, obj_mention_spans, sentence_spans = get_truncated_text(doc,
                    sent_ner[rel[0]], sent_ner[rel[1]], model_max_seq_length - ((len([sent_ner[rel[0]]['mention_spans']]) + len([sent_ner[rel[1]]['mention_spans']])) * 2))
            else:
                tokens = doc['tokens']
                token_indices = doc['token_indices']
                subj_mention_spans = sent_ner[rel[0]]['mention_spans']
                obj_mention_spans = sent_ner[rel[1]]['mention_spans']
                sentence_spans = doc['sentence_spans']
                
                num_fit_examples += 1
            
            sample['subjs'] = subj_mention_spans
            sample['objs'] = obj_mention_spans
                
            sample['tokens'] = tokens
            sample['token_ids'] = token_indices
            sample['sentence_spans'] = sentence_spans
            
            sample['subj_type'] = sent_ner[rel[0]]['entity_type']	
            sample['obj_type'] = sent_ner[rel[1]]['entity_type']
            sample['subj_id'] = sent_ner[rel[0]]['entity_id']
            sample['obj_id'] = sent_ner[rel[1]]['entity_id']
            
            sent_samples.append(sample)
            
            all_entity_pair_samples.add((rel[0], rel[1]))
            
            nrel += 1
            
        if add_negative_label:
            impossible_relations = [tuple(pair) for pair in doc['no_relations']]
            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if len(set(sent_ner[x]['mention_spans']).intersection(set(sent_ner[y]['mention_spans']))) > 0:
                        continue
                    if (x, y) in all_entity_pair_samples or ('SUBOBJ' not in marker_tokens and (y, x) in all_entity_pair_samples):
                        continue
                    if tuple(sorted([x,  y])) in impossible_relations:
                        continue
                        
                    sample = {}	
                    sample['docid'] = doc['doc_key']	
                    sample['id'] = '%s::(%d,%d)'%(doc['doc_key'], x, y)	
                    sample['relation'] = 'no_relation'	
                    
                    if len(doc['tokens']) > model_max_seq_length:
                        tokens, token_indices, subj_mention_spans, obj_mention_spans, sentence_spans = get_truncated_text(doc,
                            sent_ner[x], sent_ner[y], model_max_seq_length - ((len([sent_ner[x]['mention_spans']]) + len([sent_ner[y]['mention_spans']])) * 2))
                    else:
                        tokens = doc['tokens']
                        token_indices = doc['token_indices']
                        subj_mention_spans = sent_ner[x]['mention_spans']
                        obj_mention_spans = sent_ner[y]['mention_spans']
                        sentence_spans = doc['sentence_spans']
                    
                    sample['subjs'] = subj_mention_spans
                    sample['objs'] = obj_mention_spans
                        
                    sample['tokens'] = tokens
                    sample['token_ids'] = token_indices
                    sample['sentence_spans'] = sentence_spans

                    sample['subj_type'] = sent_ner[x]['entity_type']	
                    sample['obj_type'] = sent_ner[y]['entity_type']	
                    sample['subj_id'] = sent_ner[x]['entity_id']
                    sample['obj_id'] = sent_ner[y]['entity_id']
                    
                    sent_samples.append(sample)	
                    
                    all_entity_pair_samples.add((x, y))

        max_sample = max(max_sample, len(sent_samples))
        samples += sent_samples
    tot = len(samples)
                
    return data, samples, nrel
    
def convert_examples_to_features(examples, label2id, model_max_seq_length, tokenizer, marker_tokens):
    """
    Loads a data file into a list of `InputBatch`s.
    """
    max_sub_idx = 0
    max_obj_idx =0

    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if marker_tokens == '[UNK]':
            SUBJECT_START_NER = OBJECT_START_NER = SUBJECT_END_NER = OBJECT_END_NER = marker_tokens
        elif marker_tokens == '[ENTITY]':
            SUBJECT_START_NER = OBJECT_START_NER = SUBJECT_END_NER = OBJECT_END_NER = 'MARKER_%s'%marker_tokens
        elif marker_tokens == 'ENT_TYPE':
            SUBJECT_START_NER = SUBJECT_END_NER = 'MARKER_[%s]'%example['subj_type']
            OBJECT_START_NER = OBJECT_END_NER = 'MARKER_[%s]'%example['obj_type']
        elif marker_tokens == 'STARTEND':
            SUBJECT_START_NER = OBJECT_START_NER = 'MARKER_[START_%s]'
            SUBJECT_END_NER = OBJECT_END_NER = 'MARKER_[END]'
        elif marker_tokens == 'STARTEND_TYPE':
            SUBJECT_START_NER = 'MARKER_[START_%s]'%example['subj_type']
            OBJECT_START_NER = 'MARKER_[START_%s]'%example['obj_type']
            SUBJECT_END_NER = 'MARKER_[END_%s]'%example['subj_type']
            OBJECT_END_NER = 'MARKER_[END_%s]'%example['obj_type']
            
        subj_starts = [start for (start, end) in example['subjs']]
        subj_ends = [end for (start, end) in example['subjs']]
        obj_starts = [start for (start, end) in example['objs']]
        obj_ends = [end for (start, end) in example['objs']]

        # code doesn't account for overlapping subject and object indices
        # code doesn't account for same subj/obj indices
        tokens = []
        sub_idx = []
        obj_idx = []
        for i, token in enumerate(example['token_ids']):
            if i in subj_starts:
                sub_idx.append(len(tokens))
                tokens.append(tokenizer.convert_tokens_to_ids(SUBJECT_START_NER))
            if i in obj_starts:
                obj_idx.append(len(tokens))
                tokens.append(tokenizer.convert_tokens_to_ids(OBJECT_START_NER))

            tokens.append(token)
            
            if i in subj_ends:
                tokens.append(tokenizer.convert_tokens_to_ids(SUBJECT_END_NER))
            if i in obj_ends:
                tokens.append(tokenizer.convert_tokens_to_ids(OBJECT_END_NER))

        num_tokens += len(tokens)
        max_tokens = max(max_tokens, len(tokens))

        if len(tokens) <= model_max_seq_length:
            num_fit_examples += 1
        
        tokens = tokens[:model_max_seq_length]
        sub_idx = [idx for idx in sub_idx if idx < model_max_seq_length]
        if len(sub_idx) > max_sub_idx:
            max_sub_idx = len(sub_idx)
        if len(sub_idx) == 0:
            sub_idx.append(0)
            print(f"Subject does not fit: {example['id']}, {example['relation']}")
            
        obj_idx = [idx for idx in obj_idx if idx < model_max_seq_length]
        if len(obj_idx) > max_obj_idx:
            max_obj_idx = len(obj_idx)
        if len(obj_idx) == 0:
            obj_idx.append(0)
            print(f"Object does not fit: {example['id']}, {example['relation']}")
        
        
        segment_ids = [0] * len(tokens)
        input_ids = tokens
        input_mask = [1] * len(input_ids)
        padding = [0] * (model_max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label2id[example['relation']]
        assert len(input_ids) == model_max_seq_length
        assert len(input_mask) == model_max_seq_length
        assert len(segment_ids) == model_max_seq_length
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              sub_idx=sub_idx,
                              obj_idx=obj_idx))
    return features, max(max_sub_idx, max_obj_idx)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_f1(preds, labels, e2e_ngold):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0

        if e2e_ngold is not None:
            e2e_recall = n_correct * 1.0 / e2e_ngold
            e2e_f1 = 2.0 * prec * e2e_recall / (prec + e2e_recall)
        else:
            e2e_recall = e2e_f1 = 0.0
        return {'precision': prec, 'recall': e2e_recall, 'f1': e2e_f1, 'task_recall': recall, 'task_f1': f1, 
        'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': e2e_ngold, 'task_ngold': n_gold}


def evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=None, verbose=True):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        sub_idx = sub_idx.to(device)
        obj_idx = obj_idx.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None, sub_idx=sub_idx, obj_idx=obj_idx)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    logits = preds[0]
    preds = np.argmax(preds[0], axis=1)
    result = compute_f1(preds, eval_label_ids.numpy(), e2e_ngold=e2e_ngold)
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return preds, result, logits

def print_pred_json(eval_data, eval_examples, preds, id2label, output_file):
    rels = dict()
    for ex, pred in zip(eval_examples, preds):
        doc, sub, obj = decode_sample_id(ex['id'])
        if doc not in rels:
            rels[doc] = []
        if pred != 0:
            rels[doc].append([ex['subj_id'], ex['obj_id'], id2label[pred]])

    for doc in eval_data:
        doc['relations'] = rels.get(str(doc['doc_key']), [])
    
    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in eval_data))
        
def print_pred_csv(eval_data, eval_examples, preds, id2label, output_file):
    rels = dict()
    i = 0
    with open(output_file[:-5] + '.csv', 'w') as f:
        csvf = csv.writer(f, delimiter = '\t')
        csvf.writerow(['id', 'abstract_id', 'type', 'entity_1_id', 'entity_2_id'])
        for ex, pred in zip(eval_examples, preds):
            doc, sub, obj = decode_sample_id(ex['id'])
            if doc not in rels:
                rels[doc] = []
                
            triple = (ex['subj_id'], ex['obj_id'], id2label[pred])
            if pred != 0 and triple not in rels[doc]:
                rels[doc].append(triple)
                
                csvf.writerow([i, doc, id2label[pred], ex['subj_id'], ex['obj_id']])
                i += 1

def setseed(seed):
    random.seed(seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_trained_model(output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger.info('Saving model to %s'%output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_pretrained(output_dir)
    
def main(args):
    if args.with_mention_attention:
        RelationModel = BertForRelationMultiMentionAttention
    else:
        RelationModel = BertForRelationMultiMention

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    # train set
    if args.do_train:
        train_dataset, train_examples, train_nrel = generate_relation_data(args.train_file,
                                                        add_negative_label = not(args.no_negative_label), marker_tokens = args.marker_tokens,
                                                        model_max_seq_length = args.model_max_seq_length)
    # dev set
    if args.do_eval:
        eval_dataset, eval_examples, eval_nrel = generate_relation_data(args.eval_file, 
                                                        add_negative_label = not(args.no_negative_label), marker_tokens = args.marker_tokens,
                                                        model_max_seq_length = args.model_max_seq_length)
    # test set
    if args.do_predict:
        test_dataset, test_examples, test_nrel = generate_relation_data(args.test_file, 
                                                        add_negative_label = not(args.no_negative_label), marker_tokens = args.marker_tokens, 
                                                        model_max_seq_length = args.model_max_seq_length)

    setseed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(sys.argv)
    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(
        device, n_gpu))

    # get label_list
    if os.path.exists(os.path.join(args.output_dir, 'label_list.json')):
        with open(os.path.join(args.output_dir, 'label_list.json'), 'r') as f:
            label_list = json.load(f)
    else:
        if args.no_negative_label:	
            label_list = task_rel_labels[args.task]	
        else:
            label_list = [args.negative_label] + task_rel_labels[args.task]
        with open(os.path.join(args.output_dir, 'label_list.json'), 'w') as f:
            json.dump(label_list, f)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.marker_tokens != '[UNK]' and args.do_train:
        add_marker_tokens(tokenizer, args.marker_tokens, task_ner_labels[args.task])
        
    model = RelationModel.from_pretrained(
            args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_rel_labels=num_labels)
    model.to(device)

    if args.do_eval:
        eval_features, eval_max_ent_idx = convert_examples_to_features(
            eval_examples, label2id, args.model_max_seq_length, tokenizer, args.marker_tokens)
        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_sub_idx = torch.tensor([[f.sub_idx[i] if i < len(f.sub_idx) else -1 for i in range(eval_max_ent_idx)] for f in eval_features], dtype=torch.long)
        all_obj_idx = torch.tensor([[f.obj_idx[i] if i < len(f.obj_idx) else -1 for i in range(eval_max_ent_idx)] for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        eval_label_ids = all_label_ids

    if args.do_train:
        train_features, train_max_ent_idx = convert_examples_to_features(
            train_examples, label2id, args.model_max_seq_length, tokenizer, args.marker_tokens)
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_sub_idx = torch.tensor([[f.sub_idx[i] if i < len(f.sub_idx) else -1 for i in range(train_max_ent_idx)] for f in train_features], dtype=torch.long)
        all_obj_idx = torch.tensor([[f.obj_idx[i] if i < len(f.obj_idx) else -1 for i in range(train_max_ent_idx)] for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None

        lr = args.learning_rate
        # model = RelationModel.from_pretrained(
        #     args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_rel_labels=num_labels)
        
        if hasattr(model, 'bert'):
            model.bert.resize_token_embeddings(len(tokenizer))
        elif hasattr(model, 'albert'):
            model.albert.resize_token_embeddings(len(tokenizer))
        else:
            raise TypeError("Unknown model class")

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
            
        pgd = PGD(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=not(args.bertadam))
        scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * args.warmup_proportion), num_train_optimization_steps)

        start_time = time.time()
        global_step = 0
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        K = args.pgd_k
        for epoch in range(int(args.num_train_epochs)):
            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)
            for step, batch in enumerate(train_batches):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx = batch
                loss, logits = model(input_ids, segment_ids, input_mask, label_ids, sub_idx, obj_idx)
                if n_gpu > 1:
                    loss = loss.mean()
                
                org_stm_acc = simple_accuracy(label_ids.detach().cpu().numpy(), np.argmax(logits.detach().cpu().numpy(), axis=1))
                    
                loss.backward()

                tr_loss += loss.item()
                
                pgd.backup_grad()
                
                for t in range(K):
                    # Add perturbation to the embeddings, and back up param.data at the first attack
                    pgd.attack(is_first_attack=(t==0))
                    if t != K-1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv, logits_adv = model(input_ids, segment_ids, input_mask, label_ids, sub_idx, obj_idx)

                    new_stm_acc = simple_accuracy(label_ids.detach().cpu().numpy(), np.argmax(logits_adv.detach().cpu().numpy(), axis=1))
                    
                    if new_stm_acc < org_stm_acc:  # Once the accuracy decreases, stop searching for adversarial noise
                        pgd.restore_grad()
                        loss_adv.backward()  # Backpropogate, and add the gradient of adversarial training to the normal gradiant
                        break
                    loss_adv.backward()
                pgd.restore()  # Restore the embedding parameters
                
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                        epoch, step + 1, len(train_batches),
                        time.time() - start_time, tr_loss / nb_tr_steps))
            
            if args.do_eval:
                preds, result, logits = evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=eval_nrel)
                model.train()
                
                if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                    best_result = result
                    logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))
                    
                    save_trained_model(os.path.join(args.output_dir, 'best_model'), model, tokenizer)

            if args.checkpoint is not None and (epoch + 1) %  args.checkpoint == 0:
                save_trained_model(os.path.join(args.output_dir, f'checkpoint-{epoch}'), model, tokenizer)                
    
        save_trained_model(args.output_dir, model, tokenizer)  
    if args.do_eval:
        preds, result, logits = evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=eval_nrel)
        
        # best_result = None
        # if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
        #     best_result = result
        #     logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
        #                 (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))
                                
    if args.do_predict:
        test_features, test_max_ent_idx = convert_examples_to_features(
        test_examples, label2id, args.model_max_seq_length, tokenizer, args.marker_tokens)
        logger.info("***** Test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_sub_idx = torch.tensor([[f.sub_idx[i] if i < len(f.sub_idx) else -1 for i in range(test_max_ent_idx)] for f in test_features], dtype=torch.long)
        all_obj_idx = torch.tensor([[f.obj_idx[i] if i < len(f.obj_idx) else -1 for i in range(test_max_ent_idx)] for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size)
        test_label_ids = all_label_ids
        
        preds, result, logits = evaluate(model, device, test_dataloader, test_label_ids, num_labels, e2e_ngold=test_nrel, verbose = False)

        print_pred_json(test_dataset, test_examples, preds, id2label, os.path.join(args.output_dir, args.prediction_file))
        print_pred_csv(test_dataset, test_examples, preds, id2label, os.path.join(args.output_dir, args.prediction_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--with_mention_attention", action='store_true', help="Whether to calculate attention for pairs of subject/object mentions")
    parser.add_argument('--task', type=str, default=None, required=True, choices=['litcoin', 'novelty'])
    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--checkpoint", default=None, type=int,
                        help="If provided, save model after every x epochs")
    parser.add_argument("--model_max_seq_length", default=128, type=int,
                        help="Maximum total input sequence length \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--no_negative_label", default=False, action="store_true", help = "Whether to add negative labels")
    
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_file", default=None, type=str, help="The path of the training data.")
    parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
    
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_file", default=None, type=str, help="The path of the evaluation data.")
    
    parser.add_argument("--do_predict", action='store_true', help="Whether to run predictions on test set.")
    parser.add_argument("--test_file", default=None, type=str, help="The path of the test data.")

    parser.add_argument("--prediction_file", type=str, default="predictions.json", help="The prediction filename for the relation model")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")

    parser.add_argument('--marker_tokens', default='[UNK]', choices=['[UNK]', '[ENTITY]', 'ENT_TYPE', 'STARTEND', 'STARTEND_TYPE'], 
                        help="Type of entity marker tokens")
                        
    parser.add_argument('--pgd_k', default = 3, type = int, help = 'number of adversarial training iterations')

    args = parser.parse_args()
    main(args)
