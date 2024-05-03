import os
import torch
import math
import gc
import time
import numpy as np
import json

from utils import *
from constants import *
from models import *
from scripts import benchmark_model
from transformers import *
from data import load_data
from data.base import *
from scorer import evaluate
from argparse import ArgumentParser

PRETRAINED_MODEL = None

def train(configs):
    dataset_name = configs['dataset']
    config_name = configs['config_name']

    # For reproducibility
    torch.manual_seed(configs['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(configs['seed'])
        torch.cuda.manual_seed_all(configs['seed'])
    np.random.seed(configs['seed'])

    # Load dataset
    start_time = time.time()
    train, dev, test, ontology, dev_org, test_org = load_data(
        configs['dataset'], configs['use_synthetic_train'], 
        configs['do_predict_dev'], configs['do_predict_test']
    )
    if dataset_name in SEPARATE_ONTOLOGIES:
        train_ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{dataset_name}_train.json'))
        dev_ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{dataset_name}_dev.json'))
        test_ontology = ontology
    else:
        train_ontology = dev_ontology = test_ontology = ontology

    print('Prepared the dataset (%s seconds)'  % (time.time() - start_time))

    # Load model
    if configs['lightweight']:
        print('class LightWeightModel')
        model = LightWeightModel(configs)
    elif not configs['online_kd']:
        print('class DualBertEncodersModel')
        model = DualBertEncodersModel(configs)
    else:
        print('class EncodersModelWithOnlineKD')
        model = EncodersModelWithOnlineKD(configs)
    print('Prepared the model (Nb params: {})'.format(get_n_params(model)), flush=True)
    print(f'Nb tunable params: {get_n_tunable_params(model)}')

    # Reload a pretrained model (if exists)
    if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL):
        print('Reload the pretrained model')
        checkpoint = torch.load(PRETRAINED_MODEL, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    FINETUNED_MODEL = configs['finetuned_model_dir']

    # Reload a fine-tuned model (if exists)
    if (configs['do_predict_dev_gold'] or configs['do_predict_dev'] or configs['do_predict_test']) and \
        FINETUNED_MODEL and os.path.exists(FINETUNED_MODEL):

        print('Reload the fine-tuned model >>', str(FINETUNED_MODEL))
        checkpoint = torch.load(FINETUNED_MODEL, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if configs['do_predict_dev_gold']:
            test_org = dev_org
            dev_ontology.build_index(model, 512)
            with torch.no_grad():
                test_results, _, test_predictions, test_predictions_names = evaluate(
                    model, dev, dev_ontology, configs, return_prediction=True
                )
        else:
            test_ontology.build_index(model, 512)
            with torch.no_grad():
                test_results, _, test_predictions, test_predictions_names = evaluate(
                    model, test, test_ontology, configs, return_prediction=True
                )

        print('Test results: {}'.format(test_results))
        gc.collect()
        torch.cuda.empty_cache()

        prediction_result = []
        print(len(test_org), len(test_predictions))
        for gold, pred, pred_name in zip(test_org, test_predictions, test_predictions_names):
            prediction_result.append(
                {
                    'mention': gold.mention['term'], 
                    'gold_id': gold.mention['entity_id'],
                    'pred_id': pred[0],
                    'pred_name': pred_name[0]
                }
            )
        if configs['do_predict_dev_gold']:
            pred_file_fp = 'predictions_dev_gold' 
        elif configs['do_predict_dev']:
            pred_file_fp = 'predictions_dev' 
        elif configs['do_predict_test']:
            pred_file_fp = 'predictions_test'

        if 'disease' in configs['dataset']:
            pred_file_fp += '_disease.json'
        else:
            pred_file_fp += '_chemical.json'

        with open(os.path.join(configs['result_dir'], pred_file_fp), 'w') as f:
            json.dump(prediction_result, f)
            
        return

    # Evaluate the initial model on the dev set and the test set
    print('Evaluate the initial model on the DEV set')
    if dataset_name in SEPARATE_ONTOLOGIES:
        train_ontology.build_index(model, 512)
        dev_ontology.build_index(model, 512)
        # test_ontology.build_index(model, 512)
    else:
        ontology.build_index(model, 512)
    with torch.no_grad():
        if configs['hard_negatives_training']:
            train_results = evaluate(model, train, train_ontology, configs)
            print('Train results: {}'.format(train_results))
        dev_results = evaluate(model, dev, dev_ontology, configs)
        print('Dev results: {}'.format(dev_results))
        # test_results = evaluate(model, test, test_ontology, configs)
        # print('Test results: {}'.format(test_results))
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare the optimizer and the scheduler
    optimizer = model.get_optimizer(len(train))
    num_epoch_steps = math.ceil(len(train) / configs['batch_size'])
    print('Prepared the optimizer and the scheduler', flush=True)

    # Start Training
    accumulated_loss = RunningAverage()
    iters, batch_loss, best_dev_score, final_test_results = 0, 0, 0, None
    gradient_accumulation_steps = configs['gradient_accumulation_steps']
    curr_patience = 0
    for epoch_ix in range(configs['epochs']):

        if curr_patience == configs['max_patience']:
            print('Reached at Max Patience. Now End Training.')
            break

        print('Starting epoch {}'.format(epoch_ix+1), flush=True)
        for i in range(num_epoch_steps):
            iters += 1
            instances = train.next_items(configs['batch_size'])

            # Compute iter_loss
            iter_loss = model(instances, train_ontology, is_training=True)[0]
            iter_loss = iter_loss / gradient_accumulation_steps
            iter_loss.backward()
            batch_loss += iter_loss.data.item()

            # Update params
            if iters % gradient_accumulation_steps == 0:
                accumulated_loss.update(batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = 0

            # Report loss
            if iters % configs['report_frequency'] == 0:
                print('{} Average Loss = {}'.format(iters, round(accumulated_loss(), 3)), flush=True)
                accumulated_loss = RunningAverage()

        if (epoch_ix + 1) % configs['epoch_evaluation_frequency'] > 0: continue

        # Build the index of the ontology
        print('Starting building the index of the ontology')
        if dataset_name in SEPARATE_ONTOLOGIES:
            train_ontology.build_index(model, 512)
            dev_ontology.build_index(model, 512)
            # test_ontology.build_index(model, 512)
        else:
            ontology.build_index(model, 512)

        # Evaluation after each epoch
        with torch.no_grad():
            if configs['hard_negatives_training']:
                train_results = evaluate(model, train, train_ontology, configs)
                print('Train results: {}'.format(train_results))

            start_dev_eval_time = time.time()
            print('Evaluation on the dev set')
            if (dataset_name in SEPARATE_ONTOLOGIES) and USE_TRAINDEV:
                dev_results, _, dev_predictions, dev_predictions_names = evaluate(
                    model, test, test_ontology, configs, return_prediction=True
                )
            else:
                dev_results, _, dev_predictions, dev_predictions_names = evaluate(
                    model, dev, dev_ontology, configs, return_prediction=True
                )
            print(dev_results)
            dev_score = dev_results['top1_accuracy']
            print('Evaluation on the dev set took %s seconds'  % (time.time() - start_dev_eval_time))

            # # if online_kd is enabled
            # if configs['online_kd']:
            #     print('Evaluation using only the first 3 layers')
            #     model.enable_child_branch_exit(3)
            #     dev_score_3_layers = benchmark_model(model, 128, [configs['dataset']], 'dev')
            #     print(dev_score_3_layers)
            #     model.disable_child_branch_exit()

            #     dev_score = (dev_score + dev_score_3_layers) / 2.0

        # Save model if it has better dev score
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            best_dev_results = dev_results
            # print('Evaluation on the test set')
            # test_results = evaluate(model, test, test_ontology, configs)
            # final_test_results = test_results
            # print(test_results)

            # Save the model
            save_path = join(configs['save_dir'], 'model.pt')
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print('Saved the model', flush=True)

            # Save the prediction result based on GOLD NER
            prediction_result = []
            for gold, pred, pred_name in zip(dev_org, dev_predictions, dev_predictions_names):
                prediction_result.append(
                    {
                        'mention': gold.mention['term'], 
                        'gold_id': gold.mention['entity_id'],
                        'pred_id': pred[0],
                        'pred_name': pred_name[0]
                    }
                )
            # result_path = os.path.join(
            #     BASE_RESULT_PATH, dataset_name, config_name, 
            # )
            # os.makedirs(configs['result_dir'], exist_ok=True)

            if 'disease' in configs['dataset']:
                pred_file_fp = 'predictions_dev_gold_disease.json'
            else:
                pred_file_fp = 'predictions_dev_gold_chemical.json'
                
            with open(os.path.join(configs['result_dir'], pred_file_fp), 'w') as f:
                json.dump(prediction_result, f)
            print('Saved the prediction result with gold NER', flush=True)

            curr_patience = 0  # reset
        else:
            curr_patience += 1

        # Free memory of the index
        if not dataset_name in SEPARATE_ONTOLOGIES:
            del ontology.namevecs_index
            ontology.namevecs_index = None
        gc.collect()
        torch.cuda.empty_cache()

    print(best_dev_results)
    return best_dev_results

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--cg_config', default='lightweight_cnn_text')
    parser.add_argument('-d', '--dataset', default=BC8BIORED_D, choices=DATASETS)

    parser.add_argument('-lr', '--task_learning_rate', default=1e-3, type=float)
    parser.add_argument('-depth', '--cnn_text_depth', default=4, type=int)
    parser.add_argument('-fs', '--feature_size', default=256, type=int)
    parser.add_argument('-dropout', '--cnn_text_dropout', default=0.25, type=float)
    parser.add_argument('--seed', default=42, type=int)

    # parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict_dev", action='store_true', help="Whether to run prediction on dev set.")
    parser.add_argument("--do_predict_dev_gold", action='store_true', help="Whether to run prediction on dev set with gold NER.")
    parser.add_argument("--do_predict_test", action='store_true', help="Whether to run prediction on test set.")    

    args = parser.parse_args()

    # Prepare config
    configs = prepare_configs(args)

    # Train
    train(configs)
